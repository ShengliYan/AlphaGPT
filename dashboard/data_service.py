import os
from pathlib import Path

import duckdb
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

_ROOT = Path(__file__).resolve().parents[1]


class DashboardService:
    def __init__(self):
        self.db_path = os.getenv("DUCKDB_PATH", str(_ROOT / "data" / "alphagpt.duckdb"))
        self.strategy_file = os.getenv("STRATEGY_FILE", "best_tradfi_strategy.json")

    def _connect(self):
        path = Path(self.db_path)
        if not path.exists():
            return None
        return duckdb.connect(str(path), read_only=True)

    def get_db_status(self):
        con = self._connect()
        if con is None:
            return {"path": self.db_path, "exists": False, "n_symbols": 0, "n_bars": 0, "n_1m": 0, "n_5m": 0, "n_1h": 0, "last_time": None}
        try:
            n_symbols = con.execute("SELECT COUNT(*) FROM symbols").fetchone()[0]
            n_1m = con.execute("SELECT COUNT(*) FROM ohlcv").fetchone()[0]
            n_5m = 0
            n_1h = 0
            try:
                n_5m = con.execute("SELECT COUNT(*) FROM ohlcv_5m").fetchone()[0]
                n_1h = con.execute("SELECT COUNT(*) FROM ohlcv_1h").fetchone()[0]
            except Exception:
                pass
            last_time = con.execute("SELECT MAX(time) FROM ohlcv").fetchone()[0]
            return {
                "path": self.db_path,
                "exists": True,
                "n_symbols": int(n_symbols),
                "n_bars": int(n_1m),
                "n_1m": int(n_1m),
                "n_5m": int(n_5m),
                "n_1h": int(n_1h),
                "last_time": last_time,
            }
        except Exception:
            return {"path": self.db_path, "exists": True, "n_symbols": 0, "n_bars": 0, "n_1m": 0, "n_5m": 0, "n_1h": 0, "last_time": None}
        finally:
            con.close()

    def load_strategy_info(self):
        try:
            import json
            with open(self.strategy_file, "r") as f:
                return json.load(f)
        except (FileNotFoundError, OSError, ValueError):
            return {"formula": "Not Trained Yet"}

    def get_market_overview(self, limit=50):
        con = self._connect()
        if con is None:
            return pd.DataFrame()
        query = """
        SELECT s.symbol, s.underlying, s.name, s.status,
               o.close, o.volume, o.quote_volume, o.n_trades, o.time
        FROM ohlcv o
        JOIN symbols s ON o.symbol = s.symbol
        WHERE o.time = (SELECT MAX(time) FROM ohlcv)
        ORDER BY o.quote_volume DESC
        LIMIT ?
        """
        try:
            return con.execute(query, [int(limit)]).df()
        except Exception:
            return pd.DataFrame()
        finally:
            con.close()

    def get_recent_logs(self, n=50):
        log_file = "strategy.log"
        if not os.path.exists(log_file):
            return []
        with open(log_file, "r") as f:
            lines = f.readlines()
            return lines[-n:]
