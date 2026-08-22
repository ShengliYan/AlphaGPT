import math

import torch

from .config import ModelConfig


def _to_float(value) -> float:
    if torch.is_tensor(value):
        value = value.detach()
        if value.numel() != 1:
            value = value.reshape(-1)[0]
        value = value.item()
    return float(value)


class SpotBacktest:
    def __init__(self):
        self.trade_size = ModelConfig.TRADE_SIZE_USD
        self.min_quote_volume = ModelConfig.MIN_QUOTE_VOLUME
        self.base_fee = ModelConfig.BASE_FEE

    def _net_pnl(self, factors, raw_data, target_ret):
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
        return net_pnl, position, turnover

    def evaluate(self, factors, raw_data, target_ret):
        net_pnl, position, _turnover = self._net_pnl(factors, raw_data, target_ret)
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

    def summarize(self, factors, raw_data, target_ret, symbols=None, times=None):
        net_pnl, position, turnover = self._net_pnl(factors, raw_data, target_ret)
        n_symbols, n_bars = net_pnl.shape
        bars_per_year = float(ModelConfig.bars_per_year())
        ann = math.sqrt(bars_per_year)

        mean_pnl = net_pnl.mean(dim=1)
        std_pnl = net_pnl.std(dim=1) + 1e-8
        period_sharpe = torch.nan_to_num(
            mean_pnl / std_pnl * math.sqrt(max(n_bars, 1)), nan=0.0, posinf=0.0, neginf=0.0
        )
        ann_sharpe = torch.nan_to_num(mean_pnl / std_pnl * ann, nan=0.0, posinf=0.0, neginf=0.0)
        trades = turnover.sum(dim=1) / 2.0
        activity = position.sum(dim=1)

        ew = net_pnl.mean(dim=0)
        ew_mean = ew.mean()
        ew_std = ew.std() + 1e-8
        ew_period_sharpe = _to_float(ew_mean / ew_std * math.sqrt(max(n_bars, 1)))
        ew_ann_sharpe = _to_float(ew_mean / ew_std * ann)
        ew_mean_bar = _to_float(ew_mean)
        ew_ann_return = ew_mean_bar * bars_per_year
        equity = ew.cumsum(0)
        peak = torch.cummax(equity, 0).values
        max_dd = _to_float((equity - peak).min())
        total_ret = _to_float(equity[-1]) if n_bars else 0.0
        hit_rate = _to_float((ew > 0).float().mean()) if n_bars else 0.0

        names = list(symbols or [str(i) for i in range(n_symbols)])
        per_symbol = []
        for i, name in enumerate(names):
            per_symbol.append(
                {
                    "symbol": name,
                    "ann_sharpe": _to_float(ann_sharpe[i]),
                    "period_sharpe": _to_float(period_sharpe[i]),
                    "mean_pnl": _to_float(mean_pnl[i]),
                    "trades": _to_float(trades[i]),
                    "active_bars": _to_float(activity[i]),
                }
            )
        traded = [row for row in per_symbol if row["trades"] > 0]
        ranked = sorted(traded or per_symbol, key=lambda row: row["ann_sharpe"], reverse=True)

        return {
            "n_symbols": int(n_symbols),
            "n_bars": int(n_bars),
            "fitness_median_period_sharpe": _to_float(torch.median(period_sharpe)),
            "median_ann_sharpe": _to_float(torch.median(ann_sharpe)),
            "mean_ann_sharpe": _to_float(ann_sharpe.mean()),
            "ew_ann_sharpe": ew_ann_sharpe,
            "ew_period_sharpe": ew_period_sharpe,
            "ew_mean_bar": ew_mean_bar,
            "ew_ann_return": ew_ann_return,
            "ew_sum_bar": total_ret,
            "ew_max_drawdown": max_dd,
            "ew_hit_rate": hit_rate,
            "mean_pnl": _to_float(mean_pnl.mean()),
            "total_trades": _to_float(trades.sum()),
            "mean_trades_per_symbol": _to_float(trades.mean()),
            "top5": ranked[:5],
            "bottom5": ranked[-5:][::-1],
            "start": str(times[0]) if times is not None and len(times) else None,
            "end": str(times[-1]) if times is not None and len(times) else None,
        }


MemeBacktest = SpotBacktest
