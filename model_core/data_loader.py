import gc
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import ModelConfig
from .factors import FeatureEngineer


def _to_naive_utc(values) -> pd.DatetimeIndex:
    converted = pd.to_datetime(values, utc=True)
    if isinstance(converted, pd.Series):
        converted = converted.dt.tz_convert("UTC").dt.tz_localize(None)
        return pd.DatetimeIndex(converted)
    return pd.DatetimeIndex(converted.tz_convert("UTC").tz_localize(None))


def _rss_mb() -> float:
    try:
        with open("/proc/self/status", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    return 0.0


class CryptoDataLoader:
    def __init__(self, db_path: str | None = None):
        self.db_path = str(db_path or ModelConfig.DB_PATH)
        self.feat_tensor = None
        self.raw_data_cache = None
        self.target_ret = None
        self.symbols = []
        self.addresses = []
        self.times = None
        self.train_end = None
        self.oos_start = None
        self.split = {}

    def load_data(self, limit_tokens=None):
        if limit_tokens is None:
            limit_tokens = ModelConfig.LIMIT_SYMBOLS
        path = Path(self.db_path)
        if not path.exists():
            raise FileNotFoundError(
                f"DuckDB file not found: {path}. Run `python -m data_pipeline.run_pipeline` first."
            )
        print(f"Loading {ModelConfig.BAR_INTERVAL} bars from DuckDB ({path})...")
        table = ModelConfig.ohlcv_table()
        import duckdb

        con = duckdb.connect(str(path), read_only=True)
        try:
            if limit_tokens and int(limit_tokens) > 0:
                self.symbols = con.execute(
                    "SELECT symbol FROM symbols ORDER BY symbol LIMIT ?",
                    [int(limit_tokens)],
                ).fetchdf()["symbol"].tolist()
            else:
                self.symbols = con.execute(
                    "SELECT symbol FROM symbols ORDER BY symbol"
                ).fetchdf()["symbol"].tolist()
            if not self.symbols:
                raise ValueError("No symbols found in DuckDB. Run the data pipeline first.")
            placeholders = ",".join(["?"] * len(self.symbols))
            times_df = con.execute(
                f"""
                SELECT DISTINCT time FROM {table}
                WHERE symbol IN ({placeholders})
                ORDER BY time
                """,
                self.symbols,
            ).fetchdf()
            df = con.execute(
                f"""
                SELECT time, symbol, open, high, low, close, volume, quote_volume, n_trades
                FROM {table}
                WHERE symbol IN ({placeholders})
                ORDER BY time ASC
                """,
                self.symbols,
            ).fetchdf()
        finally:
            con.close()

        if df.empty or times_df.empty:
            raise ValueError("OHLCV table is empty. Run the data pipeline first.")

        df = df.drop_duplicates(subset=["time", "symbol"], keep="last")
        df["time"] = _to_naive_utc(df["time"])
        df["symbol"] = pd.Categorical(df["symbol"], categories=self.symbols)
        times = _to_naive_utc(times_df["time"])
        self.times = times.to_numpy().astype("datetime64[ns]")
        self.addresses = list(self.symbols)

        def to_tensor(col):
            pivot = df.pivot(index="time", columns="symbol", values=col)
            pivot = pivot.reindex(index=pd.DatetimeIndex(self.times), columns=self.symbols)
            pivot = pivot.ffill().fillna(0.0)
            arr = np.ascontiguousarray(pivot.to_numpy(dtype=np.float32).T)
            del pivot
            return torch.from_numpy(arr).to(ModelConfig.DEVICE)

        quote_volume = to_tensor("quote_volume")
        close_t = to_tensor("close")
        self.raw_data_cache = {
            "open": to_tensor("open"),
            "high": to_tensor("high"),
            "low": to_tensor("low"),
            "close": close_t,
            "volume": to_tensor("volume"),
            "quote_volume": quote_volume,
            "n_trades": to_tensor("n_trades"),
            "liquidity": quote_volume,
        }
        del df, times_df
        gc.collect()

        self._apply_time_split()
        self.feat_tensor = FeatureEngineer.compute_features(
            self.raw_data_cache, norm_end=self.train_end
        )
        self.feat_tensor = torch.nan_to_num(self.feat_tensor, nan=0.0, posinf=5.0, neginf=-5.0)
        op = self.raw_data_cache["open"]
        t1 = torch.roll(op, -1, dims=1)
        t2 = torch.roll(op, -2, dims=1)
        safe = (t1 > 1e-12) & (t2 > 1e-12)
        self.target_ret = torch.where(safe, torch.log(t2 / t1), torch.zeros_like(op))
        self.target_ret[:, -2:] = 0.0
        self.target_ret = torch.nan_to_num(self.target_ret, nan=0.0, posinf=0.0, neginf=0.0)
        # Training labels must not peek at the first two test opens.
        if self.train_end is not None and self.train_end >= 2:
            self.target_ret[:, self.train_end - 2 : self.train_end] = 0.0

        n_sym, n_ch, n_t = self.feat_tensor.shape
        print(
            f"Data Ready. Interval={ModelConfig.BAR_INTERVAL} Symbols={n_sym} "
            f"Shape={tuple(self.feat_tensor.shape)} RSS={_rss_mb():.0f}MB"
        )
        parts = [f"range={self.times[0]} → {self.times[-1]}"]
        for name, sl in self.split.items():
            parts.append(f"{name}={sl.stop - sl.start}")
        print("  " + "  ".join(parts))

    def _apply_time_split(self):
        n_t = int(self.raw_data_cache["close"].shape[1])
        test_days = int(ModelConfig.TEST_DAYS)
        oos_days = int(ModelConfig.OOS_DAYS)
        if test_days <= 0 or self.times is None or len(self.times) == 0:
            self.train_end = n_t
            self.oos_start = n_t
            self.split = {
                "train": slice(0, n_t),
                "test_30d": slice(n_t, n_t),
                "test_14d": slice(n_t, n_t),
            }
            return

        t_max = self.times[-1]
        cutoff_test = t_max - np.timedelta64(test_days, "D")
        cutoff_oos = t_max - np.timedelta64(oos_days, "D")
        train_end = int(np.searchsorted(self.times, cutoff_test, side="left"))
        oos_start = int(np.searchsorted(self.times, cutoff_oos, side="left"))
        train_end = max(2, min(train_end, n_t - 1))
        oos_start = max(train_end, min(oos_start, n_t))
        self.train_end = train_end
        self.oos_start = oos_start
        holdout_name = f"test_{test_days}d"
        recent_name = f"test_{oos_days}d"
        self.split = {"train": slice(0, train_end), holdout_name: slice(train_end, n_t)}
        if recent_name != holdout_name:
            self.split[recent_name] = slice(oos_start, n_t)

    def slice_raw(self, sl):
        return {key: tensor[:, sl] for key, tensor in self.raw_data_cache.items()}

    def slice_feat(self, sl):
        return self.feat_tensor[:, :, sl]

    def slice_target(self, sl):
        return self.target_ret[:, sl]

    def window_meta(self, name: str) -> dict:
        sl = self.split[name]
        start, end = sl.start, sl.stop
        n = end - start
        if n <= 0 or self.times is None:
            return {"name": name, "n_bars": 0, "start": None, "end": None}
        return {
            "name": name,
            "n_bars": int(n),
            "start": str(self.times[start]),
            "end": str(self.times[end - 1]),
        }
