from pathlib import Path

import duckdb
import pandas as pd
from loguru import logger

from .config import Config


class DBManager:
    def __init__(self, db_path: str | None = None):
        self.db_path = str(db_path or Config.DB_PATH)
        self.con: duckdb.DuckDBPyConnection | None = None

    def connect(self):
        if self.con is not None:
            return self.con
        path = Path(self.db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.con = duckdb.connect(str(path))
        logger.info(f"DuckDB connected: {path}")
        return self.con

    def close(self):
        if self.con is not None:
            self.con.close()
            self.con = None

    def init_schema(self):
        assert self.con is not None, "Call connect() first"
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS symbols (
                symbol VARCHAR PRIMARY KEY,
                base VARCHAR,
                quote VARCHAR,
                underlying VARCHAR,
                name VARCHAR,
                status VARCHAR,
                last_updated TIMESTAMP
            );
        """)
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS ohlcv (
                time TIMESTAMP NOT NULL,
                symbol VARCHAR NOT NULL,
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,
                close DOUBLE,
                volume DOUBLE,
                quote_volume DOUBLE,
                n_trades INTEGER,
                source VARCHAR,
                PRIMARY KEY (time, symbol)
            );
        """)
        self.con.execute("CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol ON ohlcv (symbol);")
        logger.info("DuckDB schema ready (symbols, ohlcv).")

    def upsert_symbols(self, rows):
        if rows is None:
            return
        df = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(list(rows))
        if df.empty:
            return
        if "last_updated" not in df.columns:
            df = df.copy()
            df["last_updated"] = pd.Timestamp.utcnow()
        cols = ["symbol", "base", "quote", "underlying", "name", "status", "last_updated"]
        incoming = df[cols]
        self.con.register("_incoming_symbols", incoming)
        self.con.execute("""
            INSERT OR REPLACE INTO symbols
            BY NAME
            SELECT symbol, base, quote, underlying, name, status, last_updated
            FROM _incoming_symbols
        """)
        self.con.unregister("_incoming_symbols")
        logger.info(f"Upserted {len(incoming)} symbols.")

    def batch_insert_ohlcv(self, records):
        if records is None:
            return 0
        df = records if isinstance(records, pd.DataFrame) else pd.DataFrame(list(records))
        if df.empty:
            return 0
        cols = [
            "time", "symbol", "open", "high", "low", "close",
            "volume", "quote_volume", "n_trades", "source",
        ]
        incoming = df[cols].copy()
        incoming["time"] = pd.to_datetime(incoming["time"], utc=True).dt.tz_localize(None)
        self.con.register("_incoming_ohlcv", incoming)
        self.con.execute("""
            INSERT OR REPLACE INTO ohlcv
            BY NAME
            SELECT time, symbol, open, high, low, close, volume, quote_volume, n_trades, source
            FROM _incoming_ohlcv
        """)
        self.con.unregister("_incoming_ohlcv")
        return len(incoming)

    def query_df(self, sql: str, params=None) -> pd.DataFrame:
        assert self.con is not None, "Call connect() first"
        if params is None:
            return self.con.execute(sql).df()
        return self.con.execute(sql, params).df()
