import json

import torch
from torch.distributions import Categorical
from tqdm import tqdm

from .alphagpt import AlphaGPT, NewtonSchulzLowRankDecay, StableRankMonitor
from .backtest import SpotBacktest
from .config import ModelConfig
from .data_loader import CryptoDataLoader
from .vocab import FORMULA_VOCAB
from .vm import StackVM


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

    def train(self):
        print("Starting bStocks alpha mining with LoRD regularization..." if self.use_lord else "Starting bStocks alpha mining...")
        if self.use_lord:
            print("   LoRD regularization enabled")
            print("   Target keywords: ['q_proj', 'k_proj', 'attention', 'qk_norm']")
        print(f"   fee={ModelConfig.BASE_FEE:.4f} batch={ModelConfig.BATCH_SIZE} steps={ModelConfig.TRAIN_STEPS} formula_len={ModelConfig.MAX_FORMULA_LEN}")

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

            for i in range(bs):
                formula = seqs[i].tolist()
                res = self.vm.execute(formula, self.loader.feat_tensor)

                if res is None:
                    rewards[i] = -5.0
                    continue

                if res.std() < 1e-4:
                    rewards[i] = -2.0
                    continue

                score, ret_val = self.bt.evaluate(res, self.loader.raw_data_cache, self.loader.target_ret)
                if not torch.isfinite(score):
                    rewards[i] = -5.0
                    continue
                rewards[i] = score

                if score.item() > self.best_score:
                    self.best_score = score.item()
                    self.best_formula = formula
                    tqdm.write(f"[!] New King: Score {score:.2f} | Ret {ret_val:.2%} | Formula {formula}")

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

        payload = {
            "formula": self.best_formula,
            "score": self.best_score if self.best_formula is not None else None,
            "features": list(FORMULA_VOCAB.feature_names),
            "base_fee": ModelConfig.BASE_FEE,
            "symbols": self.loader.symbols,
        }
        with open(ModelConfig.STRATEGY_FILE, "w") as f:
            json.dump(payload, f, indent=2)
        with open(ModelConfig.HISTORY_FILE, "w") as f:
            json.dump(self.training_history, f)

        print("\nTraining completed.")
        print(f"  Best score: {self.best_score:.4f}")
        print(f"  Best formula: {self.best_formula}")
        print(f"  Wrote {ModelConfig.STRATEGY_FILE}")


if __name__ == "__main__":
    eng = AlphaEngine(use_lord_regularization=True)
    eng.train()
