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
        self.position_mode = ModelConfig.POSITION_MODE
        self.ls_z_thresh = ModelConfig.LS_Z_THRESH
        self.impact_coeff = ModelConfig.IMPACT_COEFF

    def _positions(self, factors, quote_volume):
        finite = torch.isfinite(factors)
        is_liquid = (quote_volume > self.min_quote_volume) & finite
        if self.position_mode == "long_only":
            signal = torch.sigmoid(torch.nan_to_num(factors, nan=0.0))
            return (signal > 0.7).float() * is_liquid.float()

        x = torch.where(is_liquid, factors, torch.full_like(factors, float("nan")))
        mu = torch.nanmean(x, dim=0, keepdim=True)
        var = torch.nanmean((x - mu) ** 2, dim=0, keepdim=True)
        sd = torch.sqrt(var.clamp_min(0.0)) + 1e-6
        z = (x - mu) / sd
        z = torch.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
        enough = is_liquid.sum(dim=0, keepdim=True) >= 2
        live = (z.abs() > self.ls_z_thresh) & enough
        return torch.sign(z) * live.float() * is_liquid.float()

    def _net_pnl(self, factors, raw_data, target_ret):
        quote_volume = raw_data.get("quote_volume", raw_data.get("liquidity"))
        position = self._positions(factors, quote_volume)

        impact_slippage = self.impact_coeff * self.trade_size / (quote_volume + 1e-9)
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
        net_pnl, position, turnover = self._net_pnl(factors, raw_data, target_ret)
        n_symbols, n_bars = net_pnl.shape
        if float(position.abs().sum()) < 1.0:
            return torch.zeros((), device=net_pnl.device), 0.0
        ew = net_pnl.mean(dim=0)
        sharpe = ew.mean() / (ew.std() + 1e-8) * math.sqrt(max(n_bars, 1))
        sharpe = torch.nan_to_num(sharpe, nan=0.0, posinf=0.0, neginf=0.0)
        turn = float(turnover.mean().item())
        score = sharpe - float(ModelConfig.TURNOVER_PENALTY) * turn
        return score, float(ew.mean().item())

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
        activity = position.abs().sum(dim=1)

        ew = net_pnl.mean(dim=0)
        ew_mean = ew.mean()
        ew_std = ew.std() + 1e-8
        ew_period_sharpe = _to_float(ew_mean / ew_std * math.sqrt(max(n_bars, 1)))
        ew_ann_sharpe = _to_float(ew_mean / ew_std * ann)
        ew_mean_bar = _to_float(ew_mean)
        ew_ann_return = ew_mean_bar * bars_per_year
        equity = ew.cumsum(0)
        peak = torch.cummax(equity, 0).values
        max_dd = _to_float((equity - peak).min()) if n_bars else 0.0
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
        ranked = sorted(traded, key=lambda row: row["ann_sharpe"], reverse=True)
        top5 = ranked[:5]
        bottom5 = ranked[-5:][::-1] if ranked else []

        return {
            "n_symbols": int(n_symbols),
            "n_bars": int(n_bars),
            "position_mode": self.position_mode,
            "fitness_ew_period_sharpe": _to_float(
                ew_mean / ew_std * math.sqrt(max(n_bars, 1))
            ),
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
            "long_bars": _to_float((position > 0).sum()),
            "short_bars": _to_float((position < 0).sum()),
            "top5": top5,
            "bottom5": bottom5,
            "start": str(times[0]) if times is not None and len(times) else None,
            "end": str(times[-1]) if times is not None and len(times) else None,
        }


MemeBacktest = SpotBacktest
