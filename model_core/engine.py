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
    if times is None:
        return None
    return times[sl]


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

    def train(self):
        print("Starting TradFi perp alpha mining with LoRD regularization..." if self.use_lord else "Starting TradFi perp alpha mining...")
        if self.use_lord:
            print("   LoRD regularization enabled")
            print("   Target keywords: ['q_proj', 'k_proj', 'attention', 'qk_norm']")
        print(
            f"   fee={ModelConfig.BASE_FEE:.4f} batch={ModelConfig.BATCH_SIZE} "
            f"steps={ModelConfig.TRAIN_STEPS} formula_len={ModelConfig.MAX_FORMULA_LEN} "
            f"interval={ModelConfig.BAR_INTERVAL} min_qv={ModelConfig.MIN_QUOTE_VOLUME:g} "
            f"test_days={ModelConfig.TEST_DAYS} oos_days={ModelConfig.OOS_DAYS}"
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
                    _res, score, ret_val = self._eval_formula(
                        formula, self.train_feat, self.train_raw, self.train_target
                    )
                    if score is None:
                        rewards[i] = -5.0 if _res is None else -2.0
                        continue
                    rewards[i] = score

                    if score.item() > self.best_score:
                        self.best_score = score.item()
                        self.best_formula = formula
                        tqdm.write(
                            f"[!] New King: Score {score:.2f} | Ret {ret_val:.2%} | Formula {formula}"
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
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        print("\nTraining completed.")
        print(f"  Best score: {_json_num(self.best_score)}")
        print(f"  Best formula: {self.best_formula} {decode_formula(self.best_formula)}")
        print(f"  Wrote {ModelConfig.STRATEGY_FILE}")
        print(f"  Wrote {report_path}")
        self._print_report(report)
        return report

    def evaluate_windows(self):
        formula = self.best_formula if self.best_formula is not None else [0]
        windows = {}
        with torch.no_grad():
            for name, sl in self.loader.split.items():
                if sl.stop <= sl.start:
                    continue
                feat = self.loader.slice_feat(sl)
                raw = self.loader.slice_raw(sl)
                target = self.loader.slice_target(sl)
                times = _slice_times(self.loader.times, sl)
                best_res = self.vm.execute(formula, feat)
                baseline_res = self.vm.execute([0], feat)
                fallback = feat[:, 0, :]
                windows[name] = {
                    **self.loader.window_meta(name),
                    "best": self.bt.summarize(
                        best_res if best_res is not None else fallback,
                        raw,
                        target,
                        symbols=self.loader.symbols,
                        times=times,
                    ),
                    "baseline_RET": self.bt.summarize(
                        baseline_res if baseline_res is not None else fallback,
                        raw,
                        target,
                        symbols=self.loader.symbols,
                        times=times,
                    ),
                }
        return {
            "bar_interval": ModelConfig.BAR_INTERVAL,
            "n_symbols": len(self.loader.symbols),
            "symbols": self.loader.symbols,
            "formula": formula,
            "formula_decoded": decode_formula(formula),
            "train_fitness": _json_num(self.best_score) if self.best_formula is not None else None,
            "fee_bps": ModelConfig.BASE_FEE * 10000.0,
            "min_quote_volume": ModelConfig.MIN_QUOTE_VOLUME,
            "trade_size_usd": ModelConfig.TRADE_SIZE_USD,
            "train_steps": ModelConfig.TRAIN_STEPS,
            "batch_size": ModelConfig.BATCH_SIZE,
            "test_days": ModelConfig.TEST_DAYS,
            "oos_days": ModelConfig.OOS_DAYS,
            "windows": windows,
        }

    @staticmethod
    def _print_report(report):
        print("\n=== OOS backtest ===")
        print(f"formula: {report.get('formula_decoded')} {report.get('formula')}")
        for name, window in report.get("windows", {}).items():
            best = window.get("best") or {}
            base = window.get("baseline_RET") or {}
            print(
                f"{name:8s} {window.get('start')} → {window.get('end')}  bars={window.get('n_bars')}"
            )
            print(
                f"  best     median_ann_sharpe={best.get('median_ann_sharpe'):.3f}  "
                f"ew_ann_sharpe={best.get('ew_ann_sharpe'):.3f}  "
                f"ew_ret={best.get('ew_total_return'):.4f}  "
                f"maxDD={best.get('ew_max_drawdown'):.4f}  "
                f"trades={best.get('total_trades'):.0f}  "
                f"hit={best.get('ew_hit_rate'):.3f}"
            )
            print(
                f"  baseline median_ann_sharpe={base.get('median_ann_sharpe'):.3f}  "
                f"ew_ann_sharpe={base.get('ew_ann_sharpe'):.3f}  "
                f"ew_ret={base.get('ew_total_return'):.4f}  "
                f"maxDD={base.get('ew_max_drawdown'):.4f}  "
                f"trades={base.get('total_trades'):.0f}  "
                f"hit={base.get('ew_hit_rate'):.3f}"
            )


if __name__ == "__main__":
    eng = AlphaEngine(use_lord_regularization=True)
    eng.train()
