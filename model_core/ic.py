"""Cross-sectional IC / RankIC vs multi-horizon forward returns.

Factor at research bar t is scored against log(open[t+1+h] / open[t+1]), matching
the Sharpe backtest's next-open execution. Exit prices come from 1m opens so
5m / 10m / 30m horizons exist even when the research clock is 1h.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import ModelConfig
from .data_loader import _to_naive_utc


DEFAULT_IC_HORIZONS = ("5m", "10m", "30m", "1h", "1d")
_HORIZON_MINUTES = {
    "5m": 5,
    "10m": 10,
    "30m": 30,
    "1h": 60,
    "1d": 1440,
}
MIN_IC_NAMES = 10
_PRICE_EPS = 1e-12
_MATCH_TOL = np.timedelta64(1, "s")


def parse_ic_horizons(raw=None) -> list[str]:
    text = str(raw if raw is not None else ModelConfig.IC_HORIZONS)
    names = [part.strip() for part in text.split(",") if part.strip()]
    unknown = [name for name in names if name not in _HORIZON_MINUTES]
    if unknown:
        raise ValueError(f"Unknown IC horizon(s) {unknown}; use {list(_HORIZON_MINUTES)}")
    return names or list(DEFAULT_IC_HORIZONS)


def horizon_minutes(name: str) -> int:
    if name not in _HORIZON_MINUTES:
        raise ValueError(f"Unknown IC horizon {name!r}")
    return _HORIZON_MINUTES[name]


def _as_numpy(value) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _cs_pearson(x: np.ndarray, y: np.ndarray, min_names: int = MIN_IC_NAMES) -> np.ndarray:
    valid = np.isfinite(x) & np.isfinite(y)
    n = valid.sum(axis=0)
    x2 = np.where(valid, x, np.nan)
    y2 = np.where(valid, y, np.nan)
    xm = x2 - np.nanmean(x2, axis=0)
    ym = y2 - np.nanmean(y2, axis=0)
    xm = np.where(valid, xm, 0.0)
    ym = np.where(valid, ym, 0.0)
    num = (xm * ym).sum(axis=0)
    den = np.sqrt((xm * xm).sum(axis=0) * (ym * ym).sum(axis=0))
    out = np.full(x.shape[1], np.nan, dtype=np.float64)
    ok = (n >= min_names) & (den > 1e-18)
    out[ok] = num[ok] / den[ok]
    return out


def _cs_ranks(x: np.ndarray) -> np.ndarray:
    return pd.DataFrame(x).rank(axis=0, method="average").to_numpy(dtype=np.float64)


def _series_stats(values: np.ndarray) -> dict | None:
    sample = np.asarray(values, dtype=np.float64)
    sample = sample[np.isfinite(sample)]
    n = int(sample.size)
    if n < 5:
        return None
    mean = float(sample.mean())
    std = float(sample.std(ddof=1)) if n > 1 else 0.0
    ir = mean / std if std > 1e-12 else None
    tstat = mean / (std / math.sqrt(n)) if std > 1e-12 else None
    return {
        "mean": mean,
        "median": float(np.median(sample)),
        "std": std,
        "ir": ir,
        "tstat": tstat,
        "pos_share": float((sample > 0).mean()),
        "n_periods": n,
    }


def summarize_ic(factor, fwd_ret, mask, min_names: int = MIN_IC_NAMES) -> dict | None:
    x = np.asarray(_as_numpy(factor), dtype=np.float64)
    y = np.asarray(_as_numpy(fwd_ret), dtype=np.float64)
    m = np.asarray(_as_numpy(mask), dtype=bool)
    if x.shape != y.shape or x.shape != m.shape:
        raise ValueError(f"IC shape mismatch factor={x.shape} fwd={y.shape} mask={m.shape}")
    x = np.where(m, x, np.nan)
    y = np.where(m, y, np.nan)
    names = np.isfinite(x) & np.isfinite(y)
    ic = _cs_pearson(x, y, min_names=min_names)
    rank_ic = _cs_pearson(_cs_ranks(x), _cs_ranks(y), min_names=min_names)
    ic_stats = _series_stats(ic)
    rank_stats = _series_stats(rank_ic)
    if ic_stats is None and rank_stats is None:
        return None
    used = np.isfinite(ic) | np.isfinite(rank_ic)
    if names.any() and used.any():
        mean_names = float(names.sum(axis=0)[used].mean())
        if not math.isfinite(mean_names):
            mean_names = None
    else:
        mean_names = None
    return {
        "ic": ic_stats,
        "rank_ic": rank_stats,
        "mean_names": mean_names,
        "min_names": int(min_names),
    }


def _flatten_stats(bundle: dict | None) -> dict:
    if not bundle:
        return {
            "ic": None,
            "rank_ic": None,
            "ic_ir": None,
            "rank_ic_ir": None,
            "ic_tstat": None,
            "rank_ic_tstat": None,
            "ic_pos_share": None,
            "rank_ic_pos_share": None,
            "n_periods": None,
            "mean_names": None,
        }
    ic = bundle.get("ic") or {}
    rank = bundle.get("rank_ic") or {}
    return {
        "ic": ic.get("mean"),
        "rank_ic": rank.get("mean"),
        "ic_ir": ic.get("ir"),
        "rank_ic_ir": rank.get("ir"),
        "ic_tstat": ic.get("tstat"),
        "rank_ic_tstat": rank.get("tstat"),
        "ic_pos_share": ic.get("pos_share"),
        "rank_ic_pos_share": rank.get("pos_share"),
        "n_periods": ic.get("n_periods") or rank.get("n_periods"),
        "mean_names": bundle.get("mean_names"),
    }


class MinuteOpenIndex:
    """Sparse 1m open panel at the timestamps needed for IC exit prices."""

    def __init__(self, times: np.ndarray, opens: np.ndarray):
        self.times = np.asarray(times).astype("datetime64[ns]")
        self.opens = np.asarray(opens, dtype=np.float32)
        if self.opens.ndim != 2 or self.opens.shape[1] != self.times.shape[0]:
            raise ValueError(f"Minute opens shape {self.opens.shape} != times {self.times.shape}")

    def price_at(self, query_times: np.ndarray) -> np.ndarray:
        query = np.asarray(query_times).astype("datetime64[ns]")
        prices = np.full((self.opens.shape[0], query.shape[0]), np.nan, dtype=np.float64)
        valid = np.array(query != np.datetime64("NaT"), dtype=bool)
        if not valid.any() or self.times.size == 0:
            return prices
        idx = np.searchsorted(self.times, query)
        idx = np.clip(idx, 0, len(self.times) - 1)
        match = np.abs(self.times[idx] - query) <= _MATCH_TOL
        ok = valid & match
        if ok.any():
            taken = self.opens[:, idx[ok]].astype(np.float64, copy=False)
            taken = np.where(taken > _PRICE_EPS, taken, np.nan)
            prices[:, ok] = taken
        return prices


def _needed_times(research_times: np.ndarray, horizons: list[str]) -> np.ndarray:
    times = np.asarray(research_times).astype("datetime64[ns]")
    if times.size < 2:
        return times
    entry = times[1:]
    parts = [entry]
    for name in horizons:
        parts.append(entry + np.timedelta64(horizon_minutes(name), "m"))
    needed = np.unique(np.concatenate(parts))
    return needed[needed != np.datetime64("NaT")]


def load_minute_opens(symbols: list[str], needed_times: np.ndarray, db_path: str | None = None) -> MinuteOpenIndex:
    path = Path(db_path or ModelConfig.DB_PATH)
    if not path.exists():
        raise FileNotFoundError(f"DuckDB file not found: {path}")
    needed = pd.DatetimeIndex(_to_naive_utc(needed_times)).unique().sort_values()
    if needed.empty:
        return MinuteOpenIndex(np.array([], dtype="datetime64[ns]"), np.zeros((len(symbols), 0), dtype=np.float32))

    import duckdb

    placeholders = ",".join(["?"] * len(symbols))
    need_df = pd.DataFrame({"time": needed})
    con = duckdb.connect(str(path), read_only=True)
    try:
        con.register("need_times", need_df)
        df = con.execute(
            f"""
            SELECT o.time, o.symbol, o.open
            FROM ohlcv o
            WHERE o.symbol IN ({placeholders})
              AND o.time IN (SELECT time FROM need_times)
            """,
            list(symbols),
        ).fetchdf()
        if df.empty:
            t0 = needed[0].to_pydatetime()
            t1 = needed[-1].to_pydatetime()
            df = con.execute(
                f"""
                SELECT time, symbol, open
                FROM ohlcv
                WHERE symbol IN ({placeholders})
                  AND time BETWEEN ? AND ?
                """,
                list(symbols) + [t0, t1],
            ).fetchdf()
    finally:
        con.close()

    if df.empty:
        return MinuteOpenIndex(np.array([], dtype="datetime64[ns]"), np.zeros((len(symbols), 0), dtype=np.float32))

    df = df.drop_duplicates(subset=["time", "symbol"], keep="last")
    df["time"] = _to_naive_utc(df["time"])
    pivot = df.pivot(index="time", columns="symbol", values="open")
    pivot = pivot.reindex(index=needed, columns=symbols)
    times = pivot.index.to_numpy().astype("datetime64[ns]")
    opens = np.ascontiguousarray(pivot.to_numpy(dtype=np.float32).T)
    return MinuteOpenIndex(times, opens)


def build_forward_returns(
    research_times: np.ndarray,
    research_open: np.ndarray,
    minute_index: MinuteOpenIndex,
    horizon: str,
) -> np.ndarray:
    """log(open[t+1+h] / open[t+1]) aligned to factor column t. Last bar is NaN."""
    times = np.asarray(research_times).astype("datetime64[ns]")
    opens = np.asarray(_as_numpy(research_open), dtype=np.float64)
    n_sym, n_t = opens.shape
    fwd = np.full((n_sym, n_t), np.nan, dtype=np.float64)
    if n_t < 2:
        return fwd
    entry_px = opens[:, 1:]
    entry_times = times[1:]
    exit_times = entry_times + np.timedelta64(horizon_minutes(horizon), "m")
    exit_px = minute_index.price_at(exit_times)
    safe = (entry_px > _PRICE_EPS) & (exit_px > _PRICE_EPS)
    with np.errstate(divide="ignore", invalid="ignore"):
        ret = np.where(safe, np.log(exit_px / entry_px), np.nan)
    fwd[:, :-1] = ret
    return fwd


def ic_mask(factors, quote_volume, fwd_ret, min_quote_volume: float | None = None) -> np.ndarray:
    factor = np.asarray(_as_numpy(factors), dtype=np.float64)
    qv = np.asarray(_as_numpy(quote_volume), dtype=np.float64)
    fwd = np.asarray(_as_numpy(fwd_ret), dtype=np.float64)
    min_qv = float(ModelConfig.MIN_QUOTE_VOLUME if min_quote_volume is None else min_quote_volume)
    return np.isfinite(factor) & np.isfinite(fwd) & (qv > min_qv)


def ic_by_horizon(
    factors,
    quote_volume,
    fwd_by_horizon: dict[str, np.ndarray],
    min_quote_volume: float | None = None,
) -> dict[str, dict]:
    out = {}
    for name, fwd in fwd_by_horizon.items():
        mask = ic_mask(factors, quote_volume, fwd, min_quote_volume=min_quote_volume)
        out[name] = _flatten_stats(summarize_ic(factors, fwd, mask))
    return out


def build_horizon_panels(loader, horizons: list[str] | None = None) -> dict[str, np.ndarray]:
    names = list(horizons or parse_ic_horizons())
    times = np.asarray(loader.times)
    opens = _as_numpy(loader.raw_data_cache["open"])
    needed = _needed_times(times, names)
    print(f"Loading 1m opens for IC exits ({len(needed):,} timestamps, {len(loader.symbols)} symbols)...")
    minute_index = load_minute_opens(loader.symbols, needed, db_path=loader.db_path)
    print(f"  1m IC index times={minute_index.times.size:,} hits={int(np.isfinite(minute_index.opens).sum()):,}")
    return {name: build_forward_returns(times, opens, minute_index, name) for name in names}


def slice_horizon_panels(panels: dict[str, np.ndarray], sl: slice) -> dict[str, np.ndarray]:
    return {name: arr[:, sl] for name, arr in panels.items()}


def self_check() -> None:
    rng = np.random.default_rng(0)
    n, t = 40, 80
    factor = rng.normal(size=(n, t))
    noise = rng.normal(scale=0.05, size=(n, t))
    aligned = summarize_ic(factor, factor + noise, np.ones((n, t), dtype=bool))
    flipped = summarize_ic(factor, -(factor + noise), np.ones((n, t), dtype=bool))
    shuffled = factor[:, rng.permutation(t)]
    independent = summarize_ic(factor, shuffled, np.ones((n, t), dtype=bool))
    assert aligned is not None and aligned["ic"]["mean"] > 0.8
    assert aligned["rank_ic"]["mean"] > 0.8
    assert flipped["ic"]["mean"] < -0.8
    assert abs(independent["ic"]["mean"]) < 0.2
    times = np.arange("2026-01-01", "2026-01-01T08:00", dtype="datetime64[h]").astype("datetime64[ns]")
    def _px(ts):
        return 100.0 + float((ts - times[0]) / np.timedelta64(1, "h"))

    research_open = np.array([[_px(ts) for ts in times]], dtype=np.float64)
    minute_times = np.arange(
        times[0],
        times[-1] + np.timedelta64(2, "h"),
        np.timedelta64(1, "m"),
    ).astype("datetime64[ns]")
    minute_open = np.array([[_px(ts) for ts in minute_times]], dtype=np.float32)
    index = MinuteOpenIndex(minute_times, minute_open)
    fwd_1h = build_forward_returns(times, research_open, index, "1h")
    sample = fwd_1h[0, 0]
    expected = math.log(_px(times[1] + np.timedelta64(1, "h")) / _px(times[1]))
    assert np.isfinite(sample) and abs(sample - expected) < 1e-6, (sample, expected)
    print("IC self-check passed.")


if __name__ == "__main__":
    self_check()
