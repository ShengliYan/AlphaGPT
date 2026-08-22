import os
from pathlib import Path

import torch
from dotenv import load_dotenv

from .vocab import FORMULA_VOCAB

load_dotenv()

_ROOT = Path(__file__).resolve().parents[1]


class ModelConfig:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DB_PATH = os.getenv("DUCKDB_PATH", str(_ROOT / "data" / "alphagpt.duckdb"))
    BAR_INTERVAL = os.getenv("BAR_INTERVAL", "1h")
    # 0 = all symbols in DuckDB
    LIMIT_SYMBOLS = int(os.getenv("LIMIT_SYMBOLS", "0"))
    TEST_DAYS = int(os.getenv("TEST_DAYS", "30"))
    OOS_DAYS = int(os.getenv("OOS_DAYS", "14"))
    # Days immediately before the test holdout, used only to select formulas.
    VALID_DAYS = int(os.getenv("VALID_DAYS", "30"))
    N_FOLDS = int(os.getenv("N_FOLDS", "4"))
    # Penalize unstable fold Sharpes and train/valid gaps: score - GAP_PENALTY * std/gap.
    GAP_PENALTY = float(os.getenv("GAP_PENALTY", "1.0"))
    # Subtract TURNOVER_PENALTY * mean(|Δposition|) from the training Sharpe.
    TURNOVER_PENALTY = float(os.getenv("TURNOVER_PENALTY", "1.0"))
    _INTERVAL_DEFAULTS = {
        "1m": {"batch": "32", "steps": "80", "min_qv": "100"},
        "5m": {"batch": "32", "steps": "100", "min_qv": "300"},
        "1h": {"batch": "256", "steps": "200", "min_qv": "1000"},
    }
    _d = _INTERVAL_DEFAULTS.get(BAR_INTERVAL, _INTERVAL_DEFAULTS["1h"])
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", _d["batch"]))
    TRAIN_STEPS = int(os.getenv("TRAIN_STEPS", _d["steps"]))
    MAX_FORMULA_LEN = int(os.getenv("MAX_FORMULA_LEN", "8"))
    TRADE_SIZE_USD = float(os.getenv("TRADE_SIZE_USD", "1000"))
    # 1m/5m quote volume is a fraction of hourly; 1000 would filter out most bars.
    MIN_QUOTE_VOLUME = float(os.getenv("MIN_QUOTE_VOLUME", _d["min_qv"]))
    # Maker fills on these perps are 0 fee; set FEE_BPS=5 to replay the old taker book.
    BASE_FEE = float(os.getenv("FEE_BPS", os.getenv("SPOT_FEE_BPS", "0"))) / 10000.0
    # long_short = cross-sectional sign vs mean (perps). long_only = old sigmoid>0.7 book.
    POSITION_MODE = os.getenv("POSITION_MODE", "long_short")
    LS_Z_THRESH = float(os.getenv("LS_Z_THRESH", "0"))
    # 0 = fill at mid (maker). 1 = old taker impact = trade_size / quote_volume, capped at 2%.
    IMPACT_COEFF = float(os.getenv("IMPACT_COEFF", "0"))
    INPUT_DIM = FORMULA_VOCAB.feature_count
    STRATEGY_FILE = os.getenv("STRATEGY_FILE", "best_tradfi_strategy.json")
    HISTORY_FILE = os.getenv("HISTORY_FILE", "training_history.json")
    REPORT_FILE = os.getenv(
        "REPORT_FILE",
        str(_ROOT / "reports" / f"tradfi_{BAR_INTERVAL}_oos.json"),
    )
    INTERVAL_TABLES = {
        "1m": "ohlcv",
        "5m": "ohlcv_5m",
        "1h": "ohlcv_1h",
    }
    BARS_PER_YEAR = {
        "1m": 365 * 24 * 60,
        "5m": 365 * 24 * 12,
        "1h": 365 * 24,
    }

    @classmethod
    def ohlcv_table(cls) -> str:
        table = cls.INTERVAL_TABLES.get(cls.BAR_INTERVAL)
        if not table:
            raise ValueError(f"Unsupported BAR_INTERVAL={cls.BAR_INTERVAL!r}; use 1m, 5m, or 1h")
        return table

    @classmethod
    def bars_per_year(cls) -> int:
        return cls.BARS_PER_YEAR.get(cls.BAR_INTERVAL, 365 * 24)
