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
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "256"))
    TRAIN_STEPS = int(os.getenv("TRAIN_STEPS", "200"))
    MAX_FORMULA_LEN = int(os.getenv("MAX_FORMULA_LEN", "8"))
    TRADE_SIZE_USD = float(os.getenv("TRADE_SIZE_USD", "1000"))
    MIN_QUOTE_VOLUME = float(os.getenv("MIN_QUOTE_VOLUME", "1000"))
    BASE_FEE = float(os.getenv("SPOT_FEE_BPS", "10")) / 10000.0
    INPUT_DIM = FORMULA_VOCAB.feature_count
    STRATEGY_FILE = os.getenv("STRATEGY_FILE", "best_bstock_strategy.json")
    HISTORY_FILE = os.getenv("HISTORY_FILE", "training_history.json")
