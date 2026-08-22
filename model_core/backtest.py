import math

import torch

from .config import ModelConfig


class SpotBacktest:
    def __init__(self):
        self.trade_size = ModelConfig.TRADE_SIZE_USD
        self.min_quote_volume = ModelConfig.MIN_QUOTE_VOLUME
        self.base_fee = ModelConfig.BASE_FEE

    def evaluate(self, factors, raw_data, target_ret):
        quote_volume = raw_data.get("quote_volume", raw_data.get("liquidity"))
        signal = torch.sigmoid(factors)
        is_liquid = (quote_volume > self.min_quote_volume).float()
        position = (signal > 0.7).float() * is_liquid

        impact_slippage = self.trade_size / (quote_volume + 1e-9)
        impact_slippage = torch.clamp(impact_slippage, 0.0, 0.02)
        total_slippage_one_way = self.base_fee + impact_slippage

        prev_pos = torch.roll(position, 1, dims=1)
        prev_pos[:, 0] = 0
        turnover = torch.abs(position - prev_pos)
        tx_cost = turnover * total_slippage_one_way

        gross_pnl = position * target_ret
        net_pnl = torch.nan_to_num(gross_pnl - tx_cost, nan=0.0, posinf=0.0, neginf=0.0)

        n_symbols, n_bars = net_pnl.shape
        mean_pnl = net_pnl.mean(dim=1)
        std_pnl = net_pnl.std(dim=1) + 1e-8
        sharpe = mean_pnl / std_pnl * math.sqrt(max(n_bars, 1))
        sharpe = torch.nan_to_num(sharpe, nan=0.0, posinf=0.0, neginf=0.0)

        activity = position.sum(dim=1)
        min_trades = max(1.0, 5.0 * n_symbols / 300.0)
        inactive = activity < min_trades
        score = torch.where(inactive, sharpe * 0.5, sharpe)
        final_fitness = torch.median(score)
        if not torch.isfinite(final_fitness):
            final_fitness = torch.zeros((), device=score.device)
        return final_fitness, float(mean_pnl.mean().item())


MemeBacktest = SpotBacktest
