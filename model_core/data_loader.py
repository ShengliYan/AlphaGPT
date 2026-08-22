from pathlib import Path

import duckdb
import torch

from .config import ModelConfig
from .factors import FeatureEngineer


class CryptoDataLoader:
    def __init__(self, db_path: str | None = None):
        self.db_path = str(db_path or ModelConfig.DB_PATH)
        self.feat_tensor = None
        self.raw_data_cache = None
        self.target_ret = None
        self.symbols = []
        self.addresses = []

    def load_data(self, limit_tokens=50):
        path = Path(self.db_path)
        if not path.exists():
            raise FileNotFoundError(
                f"DuckDB file not found: {path}. Run `python -m data_pipeline.run_pipeline` first."
            )
        print(f"Loading data from DuckDB ({path})...")
        con = duckdb.connect(str(path), read_only=True)
        try:
            self.symbols = con.execute(
                "SELECT symbol FROM symbols ORDER BY symbol LIMIT ?",
                [int(limit_tokens)],
            ).fetchdf()["symbol"].tolist()
            if not self.symbols:
                raise ValueError("No symbols found in DuckDB. Run the data pipeline first.")
            placeholders = ",".join(["?"] * len(self.symbols))
            df = con.execute(
                f"""
                SELECT time, symbol, open, high, low, close, volume, quote_volume, n_trades
                FROM ohlcv
                WHERE symbol IN ({placeholders})
                ORDER BY time ASC
                """,
                self.symbols,
            ).fetchdf()
        finally:
            con.close()

        if df.empty:
            raise ValueError("OHLCV table is empty. Run the data pipeline first.")

        df = df.drop_duplicates(subset=["time", "symbol"], keep="last")
        self.addresses = list(self.symbols)

        def to_tensor(col):
            pivot = df.pivot(index="time", columns="symbol", values=col)
            pivot = pivot.reindex(columns=self.symbols)
            pivot = pivot.ffill().fillna(0.0)
            return torch.tensor(pivot.values.T, dtype=torch.float32, device=ModelConfig.DEVICE)

        quote_volume = to_tensor("quote_volume")
        self.raw_data_cache = {
            "open": to_tensor("open"),
            "high": to_tensor("high"),
            "low": to_tensor("low"),
            "close": to_tensor("close"),
            "volume": to_tensor("volume"),
            "quote_volume": quote_volume,
            "n_trades": to_tensor("n_trades"),
            "liquidity": quote_volume,
        }
        self.feat_tensor = FeatureEngineer.compute_features(self.raw_data_cache)
        op = self.raw_data_cache["open"]
        t1 = torch.roll(op, -1, dims=1)
        t2 = torch.roll(op, -2, dims=1)
        self.target_ret = torch.log(t2 / (t1 + 1e-9))
        self.target_ret[:, -2:] = 0.0
        print(f"Data Ready. Symbols={len(self.symbols)} Shape={tuple(self.feat_tensor.shape)}")
