from pathlib import Path

import duckdb
import pandas as pd
from loguru import logger

from .config import Config

_AGG_SQL = """
SELECT
    time_bucket(INTERVAL '{bucket}', time) AS time,
    symbol,
    arg_min(open, time) AS open,
    max(high) AS high,
    min(low) AS low,
    arg_max(close, time) AS close,
    sum(volume) AS volume,
    sum(quote_volume) AS quote_volume,
    CAST(sum(n_trades) AS INTEGER) AS n_trades,
    any_value(source) AS source
FROM ohlcv
GROUP BY 1, 2
"""


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
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS meta (
                key VARCHAR PRIMARY KEY,
                value VARCHAR
            );
        """)
        self.con.execute("CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol ON ohlcv (symbol);")
        logger.info("DuckDB schema ready (symbols, ohlcv 1m, meta).")

    def prepare_storage_interval(self, interval: str = "1m"):
        assert self.con is not None, "Call connect() first"
        row = self.con.execute("SELECT value FROM meta WHERE key = 'storage_interval'").fetchone()
        current = row[0] if row else None
        n_rows = self.con.execute("SELECT COUNT(*) FROM ohlcv").fetchone()[0]
        if current != interval:
            if n_rows:
                logger.warning(f"OHLCV storage {current!r} -> {interval!r}; dropping {n_rows} rows")
                self.con.execute("DELETE FROM ohlcv")
                self.con.execute("DROP TABLE IF EXISTS ohlcv_5m")
                self.con.execute("DROP TABLE IF EXISTS ohlcv_1h")
            self.con.execute(
                "INSERT OR REPLACE INTO meta VALUES ('storage_interval', ?)",
                [interval],
            )

    def last_bar_times_ms(self) -> dict[str, int]:
        df = self.con.execute("SELECT symbol, max(time) AS t FROM ohlcv GROUP BY 1").df()
        out: dict[str, int] = {}
        for row in df.itertuples(index=False):
            ts = pd.Timestamp(row.t)
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            else:
                ts = ts.tz_convert("UTC")
            out[str(row.symbol)] = int(ts.timestamp() * 1000)
        return out

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

    def rebuild_resampled_bars(self):
        assert self.con is not None, "Call connect() first"
        logger.info("Resampling 1m -> 5m and 1h...")
        self.con.execute(f"CREATE OR REPLACE TABLE ohlcv_5m AS {_AGG_SQL.format(bucket='5 minutes')}")
        self.con.execute(f"CREATE OR REPLACE TABLE ohlcv_1h AS {_AGG_SQL.format(bucket='1 hour')}")
        self.con.execute("CREATE INDEX IF NOT EXISTS idx_ohlcv_5m_symbol ON ohlcv_5m (symbol)")
        self.con.execute("CREATE INDEX IF NOT EXISTS idx_ohlcv_1h_symbol ON ohlcv_1h (symbol)")
        n1m = self.con.execute("SELECT COUNT(*) FROM ohlcv").fetchone()[0]
        n5 = self.con.execute("SELECT COUNT(*) FROM ohlcv_5m").fetchone()[0]
        n1h = self.con.execute("SELECT COUNT(*) FROM ohlcv_1h").fetchone()[0]
        logger.info(f"Bars stored: 1m={n1m:,} 5m={n5:,} 1h={n1h:,}")

    def query_df(self, sql: str, params=None) -> pd.DataFrame:
        assert self.con is not None, "Call connect() first"
        if params is None:
            return self.con.execute(sql).df()
        return self.con.execute(sql, params).df()
