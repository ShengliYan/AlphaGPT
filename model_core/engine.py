import json
import math
from pathlib import Path

import torch
from torch.distributions import Categorical
from tqdm import tqdm

from .alphagpt import AlphaGPT, NewtonSchulzLowRankDecay, StableRankMonitor
from .backtest import SpotBacktest
from .config import ModelConfig
from .data_loader import CryptoDataLoader
from .report_md import write_oos_markdown
from .vocab import decode_formula, FORMULA_VOCAB
from .vm import StackVM


def _json_num(value):
    if value is None:
        return None
    if torch.is_tensor(value):
        value = value.detach().item()
    value = float(value)
    if not math.isfinite(value):
        return None
    return value


def _slice_times(times, sl):
    if times is None:
        return None
    return times[sl]


def _slice_raw(raw, sl):
    return {key: tensor[:, sl] for key, tensor in raw.items()}


def _fold_slices(n_bars: int, n_folds: int) -> list[slice]:
    n_folds = max(1, min(int(n_folds), n_bars))
    fold = max(n_bars // n_folds, 1)
    slices = []
    for k in range(n_folds):
        start = k * fold
        end = n_bars if k == n_folds - 1 else min((k + 1) * fold, n_bars)
        if end - start < 8:
            continue
        slices.append(slice(start, end))
    return slices or [slice(0, n_bars)]


class AlphaEngine:
    def __init__(self, use_lord_regularization=True, lord_decay_rate=1e-3, lord_num_iterations=5):
        """
        Initialize AlphaGPT training engine.

        Args:
            use_lord_regularization: Enable Low-Rank Decay (LoRD) regularization
            lord_decay_rate: Strength of LoRD regularization
            lord_num_iterations: Number of Newton-Schulz iterations per step
        """
        self.loader = CryptoDataLoader()
        self.loader.load_data()

        self.model = AlphaGPT().to(ModelConfig.DEVICE)

        self.opt = torch.optim.AdamW(self.model.parameters(), lr=1e-3)

        self.use_lord = use_lord_regularization
        if self.use_lord:
            self.lord_opt = NewtonSchulzLowRankDecay(
                self.model.named_parameters(),
                decay_rate=lord_decay_rate,
                num_iterations=lord_num_iterations,
                target_keywords=["q_proj", "k_proj", "attention", "qk_norm"]
            )
            self.rank_monitor = StableRankMonitor(
                self.model,
                target_keywords=["q_proj", "k_proj"]
            )
        else:
            self.lord_opt = None
            self.rank_monitor = None

        self.vm = StackVM()
        self.bt = SpotBacktest()

        self.best_score = -float("inf")
        self.best_formula = None
        self.best_train_score = None
        self.best_valid_score = None
        self.best_fold_scores = []
        self.training_history = {
            "step": [],
            "avg_reward": [],
            "best_score": [],
            "stable_rank": []
        }

        train_sl = self.loader.split["train"]
        self.train_feat = self.loader.slice_feat(train_sl)
        self.train_raw = self.loader.slice_raw(train_sl)
        self.train_target = self.loader.slice_target(train_sl)
        self.valid_feat = None
        self.valid_raw = None
        self.valid_target = None
        if "valid" in self.loader.split:
            valid_sl = self.loader.split["valid"]
            if valid_sl.stop > valid_sl.start:
                self.valid_feat = self.loader.slice_feat(valid_sl)
                self.valid_raw = self.loader.slice_raw(valid_sl)
                self.valid_target = self.loader.slice_target(valid_sl)

    def _eval_formula(self, formula, feat, raw, target):
        res = self.vm.execute(formula, feat)
        if res is None:
            return None, None, None
        if res.std() < 1e-4:
            return res, None, None
        score, ret_val = self.bt.evaluate(res, raw, target)
        if not torch.isfinite(score):
            return res, None, None
        return res, score, ret_val

    def _fold_scores(self, res, raw, target):
        slices = _fold_slices(int(res.shape[1]), ModelConfig.N_FOLDS)
        scores = []
        for sl in slices:
            score, _ret = self.bt.evaluate(res[:, sl], _slice_raw(raw, sl), target[:, sl])
            scores.append(float(score.item()) if torch.is_tensor(score) else float(score))
        mean = sum(scores) / max(len(scores), 1)
        var = sum((s - mean) ** 2 for s in scores) / max(len(scores), 1)
        std = var ** 0.5
        return scores, mean, std

    def _selection_score(self, train_mean, train_std, valid_score):
        gap_pen = float(ModelConfig.GAP_PENALTY)
        stable_train = train_mean - gap_pen * train_std
        if valid_score is None:
            return stable_train, stable_train
        reward = stable_train
        king = min(train_mean, valid_score) - gap_pen * abs(train_mean - valid_score)
        return reward, king

    def train(self):
        print("Starting TradFi perp alpha mining with LoRD regularization..." if self.use_lord else "Starting TradFi perp alpha mining...")
        if self.use_lord:
            print("   LoRD regularization enabled")
            print("   Target keywords: ['q_proj', 'k_proj', 'attention', 'qk_norm']")
        print(
            f"   fee={ModelConfig.BASE_FEE:.4f} impact={ModelConfig.IMPACT_COEFF:g} "
            f"side={ModelConfig.POSITION_MODE} z_thresh={ModelConfig.LS_Z_THRESH:g} "
            f"batch={ModelConfig.BATCH_SIZE} steps={ModelConfig.TRAIN_STEPS} "
            f"formula_len={ModelConfig.MAX_FORMULA_LEN} interval={ModelConfig.BAR_INTERVAL} "
            f"min_qv={ModelConfig.MIN_QUOTE_VOLUME:g} test_days={ModelConfig.TEST_DAYS} "
            f"valid_days={ModelConfig.VALID_DAYS} oos_days={ModelConfig.OOS_DAYS} "
            f"folds={ModelConfig.N_FOLDS} gap_pen={ModelConfig.GAP_PENALTY:g} "
            f"turn_pen={ModelConfig.TURNOVER_PENALTY:g}"
        )

        pbar = tqdm(range(ModelConfig.TRAIN_STEPS))

        for step in pbar:
            bs = ModelConfig.BATCH_SIZE
            inp = torch.zeros((bs, 1), dtype=torch.long, device=ModelConfig.DEVICE)

            log_probs = []
            tokens_list = []

            for _ in range(ModelConfig.MAX_FORMULA_LEN):
                logits, _, _ = self.model(inp)
                dist = Categorical(logits=logits)
                action = dist.sample()

                log_probs.append(dist.log_prob(action))
                tokens_list.append(action)
                inp = torch.cat([inp, action.unsqueeze(1)], dim=1)

            seqs = torch.stack(tokens_list, dim=1)
            rewards = torch.zeros(bs, device=ModelConfig.DEVICE)

            with torch.no_grad():
                for i in range(bs):
                    formula = seqs[i].tolist()
                    res = self.vm.execute(formula, self.train_feat)
                    if res is None:
                        rewards[i] = -5.0
                        continue
                    if res.std() < 1e-4:
                        rewards[i] = -2.0
                        continue
                    fold_scores, fold_mean, fold_std = self._fold_scores(
                        res, self.train_raw, self.train_target
                    )
                    valid_score = None
                    if self.valid_feat is not None:
                        _vres, vscore, _vret = self._eval_formula(
                            formula, self.valid_feat, self.valid_raw, self.valid_target
                        )
                        valid_score = float(vscore.item()) if vscore is not None else -5.0
                    reward, king = self._selection_score(fold_mean, fold_std, valid_score)
                    if not math.isfinite(reward):
                        rewards[i] = -5.0
                        continue
                    rewards[i] = reward

                    if king > self.best_score:
                        self.best_score = king
                        self.best_formula = formula
                        self.best_train_score = fold_mean
                        self.best_valid_score = valid_score
                        self.best_fold_scores = fold_scores
                        folds_txt = ",".join(f"{s:.2f}" for s in fold_scores)
                        tqdm.write(
                            f"[!] New King: sel={king:.2f} train_folds={fold_mean:.2f} "
                            f"(std={fold_std:.2f} [{folds_txt}]) valid={valid_score} "
                            f"| Formula {formula}"
                        )

            adv = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

            loss = 0
            for t in range(len(log_probs)):
                loss += -log_probs[t] * adv
            loss = loss.mean()

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            if self.use_lord:
                self.lord_opt.step()

            avg_reward = rewards.mean().item()
            postfix_dict = {"AvgRew": f"{avg_reward:.3f}", "BestScore": f"{self.best_score:.3f}"}

            if self.use_lord and step % 100 == 0:
                stable_rank = self.rank_monitor.compute()
                postfix_dict["Rank"] = f"{stable_rank:.2f}"
                self.training_history["stable_rank"].append(stable_rank)

            self.training_history["step"].append(step)
            self.training_history["avg_reward"].append(avg_reward)
            self.training_history["best_score"].append(self.best_score)
            pbar.set_postfix(postfix_dict)

        report = self.evaluate_windows()
        payload = {
            "formula": self.best_formula,
            "formula_decoded": decode_formula(self.best_formula),
            "score": _json_num(self.best_score) if self.best_formula is not None else None,
            "features": list(FORMULA_VOCAB.feature_names),
            "base_fee": ModelConfig.BASE_FEE,
            "min_quote_volume": ModelConfig.MIN_QUOTE_VOLUME,
            "bar_interval": ModelConfig.BAR_INTERVAL,
            "position_mode": ModelConfig.POSITION_MODE,
            "ls_z_thresh": ModelConfig.LS_Z_THRESH,
            "impact_coeff": ModelConfig.IMPACT_COEFF,
            "valid_days": ModelConfig.VALID_DAYS,
            "n_folds": ModelConfig.N_FOLDS,
            "gap_penalty": ModelConfig.GAP_PENALTY,
            "turnover_penalty": ModelConfig.TURNOVER_PENALTY,
            "train_fold_scores": self.best_fold_scores,
            "train_fold_mean": _json_num(self.best_train_score),
            "valid_score": _json_num(self.best_valid_score),
            "symbols": self.loader.symbols,
            "split": {name: self.loader.window_meta(name) for name in self.loader.split},
        }
        with open(ModelConfig.STRATEGY_FILE, "w") as f:
            json.dump(payload, f, indent=2)
        with open(ModelConfig.HISTORY_FILE, "w") as f:
            json.dump(
                {
                    "step": self.training_history["step"],
                    "avg_reward": [_json_num(x) for x in self.training_history["avg_reward"]],
                    "best_score": [_json_num(x) for x in self.training_history["best_score"]],
                    "stable_rank": [_json_num(x) for x in self.training_history["stable_rank"]],
                },
                f,
            )
        report_path = Path(ModelConfig.REPORT_FILE)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        slim = dict(report)
        slim.pop("symbols", None)
        with open(report_path, "w") as f:
            json.dump(slim, f, indent=2)
        md_path = report_path.with_suffix(".md")
        write_oos_markdown(slim, md_path)

        print("\nTraining completed.")
        print(f"  Best selection score: {_json_num(self.best_score)}")
        print(f"  Train fold mean: {_json_num(self.best_train_score)} folds={self.best_fold_scores}")
        print(f"  Valid score: {_json_num(self.best_valid_score)}")
        print(f"  Best formula: {self.best_formula} {decode_formula(self.best_formula)}")
        print(f"  Wrote {ModelConfig.STRATEGY_FILE}")
        print(f"  Wrote {report_path}")
        print(f"  Wrote {md_path}")
        self._print_report(report)
        return report

    def evaluate_windows(self):
        formula = self.best_formula if self.best_formula is not None else [0]
        windows = {}
        extra_formulas = {"best": formula, "baseline_RET": [0]}
        for idx, feat_name in enumerate(FORMULA_VOCAB.feature_names):
            if feat_name == "RET":
                continue
            extra_formulas[f"feature_{feat_name}"] = [idx]
        with torch.no_grad():
            for name, sl in self.loader.split.items():
                if sl.stop <= sl.start:
                    continue
                feat = self.loader.slice_feat(sl)
                raw = self.loader.slice_raw(sl)
                target = self.loader.slice_target(sl)
                times = _slice_times(self.loader.times, sl)
                fallback = feat[:, 0, :]
                window = {**self.loader.window_meta(name)}
                for key, tokens in extra_formulas.items():
                    res = self.vm.execute(tokens, feat)
                    window[key] = self.bt.summarize(
                        res if res is not None else fallback,
                        raw,
                        target,
                        symbols=self.loader.symbols,
                        times=times,
                    )
                windows[name] = window
        consistency = self._consistency_table(windows)
        return {
            "bar_interval": ModelConfig.BAR_INTERVAL,
            "n_symbols": len(self.loader.symbols),
            "symbols": self.loader.symbols,
            "formula": formula,
            "formula_decoded": decode_formula(formula),
            "train_fitness": _json_num(self.best_score) if self.best_formula is not None else None,
            "train_fold_mean": _json_num(self.best_train_score),
            "valid_score": _json_num(self.best_valid_score),
            "train_fold_scores": self.best_fold_scores,
            "selection": "min(train_fold_mean, valid) - GAP_PENALTY*|train-valid|",
            "fee_bps": ModelConfig.BASE_FEE * 10000.0,
            "min_quote_volume": ModelConfig.MIN_QUOTE_VOLUME,
            "trade_size_usd": ModelConfig.TRADE_SIZE_USD,
            "position_mode": ModelConfig.POSITION_MODE,
            "ls_z_thresh": ModelConfig.LS_Z_THRESH,
            "impact_coeff": ModelConfig.IMPACT_COEFF,
            "train_steps": ModelConfig.TRAIN_STEPS,
            "batch_size": ModelConfig.BATCH_SIZE,
            "test_days": ModelConfig.TEST_DAYS,
            "valid_days": ModelConfig.VALID_DAYS,
            "oos_days": ModelConfig.OOS_DAYS,
            "n_folds": ModelConfig.N_FOLDS,
            "gap_penalty": ModelConfig.GAP_PENALTY,
            "turnover_penalty": ModelConfig.TURNOVER_PENALTY,
            "consistency": consistency,
            "windows": windows,
        }

    def _consistency_table(self, windows):
        keys = ["best", "baseline_RET"] + [
            f"feature_{feat}" for feat in FORMULA_VOCAB.feature_names if feat != "RET"
        ]
        rows = []
        for key in keys:
            sharpes = {}
            for name, window in windows.items():
                stats = window.get(key) or {}
                sharpes[name] = stats.get("ew_ann_sharpe")
            train_s = sharpes.get("train")
            valid_s = sharpes.get("valid")
            test_s = sharpes.get("test_30d") or sharpes.get(f"test_{ModelConfig.TEST_DAYS}d")
            def _sign(value):
                if value is None:
                    return 0
                if value > 0:
                    return 1
                if value < 0:
                    return -1
                return 0
            rows.append(
                {
                    "variant": key,
                    "sharpes": sharpes,
                    "train_valid_same_sign": _sign(train_s) == _sign(valid_s) and _sign(train_s) != 0,
                    "valid_test_same_sign": _sign(valid_s) == _sign(test_s) and _sign(valid_s) != 0,
                    "train_test_same_sign": _sign(train_s) == _sign(test_s) and _sign(train_s) != 0,
                }
            )
        return rows

    @staticmethod
    def _print_report(report):
        print("\n=== OOS backtest ===")
        print(f"formula: {report.get('formula_decoded')} {report.get('formula')}")
        print(f"selection: {report.get('selection')}  folds={report.get('train_fold_scores')}")
        for row in report.get("consistency") or []:
            sharpes = row.get("sharpes") or {}
            bits = " ".join(f"{k}={v:.2f}" for k, v in sharpes.items() if v is not None)
            flags = []
            if row.get("train_valid_same_sign"):
                flags.append("train=valid")
            if row.get("valid_test_same_sign"):
                flags.append("valid=test")
            print(f"  {row.get('variant'):16s} {bits}  {' '.join(flags)}")
        for window_name, window in report.get("windows", {}).items():
            print(
                f"{window_name:8s} {window.get('start')} → {window.get('end')}  bars={window.get('n_bars')}"
            )
            keys = ["best", "baseline_RET"] + [
                f"feature_{feat}" for feat in FORMULA_VOCAB.feature_names if feat != "RET"
            ]
            for key in keys:
                stats = window.get(key) or {}
                if not stats:
                    continue
                print(
                    f"  {key:16s} med_ann_sharpe={stats.get('median_ann_sharpe'):.3f}  "
                    f"ew_ann_sharpe={stats.get('ew_ann_sharpe'):.3f}  "
                    f"ew_ann_ret={stats.get('ew_ann_return'):.3f}  "
                    f"long={stats.get('long_bars'):.0f} short={stats.get('short_bars'):.0f}  "
                    f"trades={stats.get('total_trades'):.0f}  "
                    f"hit={stats.get('ew_hit_rate'):.3f}"
                )


if __name__ == "__main__":
    eng = AlphaEngine(use_lord_regularization=True)
    eng.train()
