import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

_ROOT = Path(__file__).resolve().parents[1]


def _csv_env(name: str, default: str) -> list[str]:
    raw = os.getenv(name, default)
    return [item.strip().upper() for item in raw.split(",") if item.strip()]


class Config:
    DB_PATH = os.getenv("DUCKDB_PATH", str(_ROOT / "data" / "alphagpt.duckdb"))

    # Official market-data host; api.binance.com is geo-blocked (HTTP 451) in some regions.
    BINANCE_BASE_URL = os.getenv("BINANCE_BASE_URL", "https://data-api.binance.vision")
    BINANCE_FALLBACK_URLS = (
        "https://data-api.binance.vision",
        "https://www.binance.com",
        "https://api.binance.com",
    )
    BINANCE_INTERVAL = os.getenv("BINANCE_INTERVAL", "1h")
    BINANCE_KLINE_LIMIT = int(os.getenv("BINANCE_KLINE_LIMIT", "1000"))
    BINANCE_START = os.getenv("BINANCE_START", "2026-06-01")

    SEED_SYMBOLS = tuple(_csv_env(
        "BSTOCK_SEED_SYMBOLS",
        "TSLABUSDT,NVDABUSDT,CRCLBUSDT,SNDKBUSDT,MUBUSDT",
    ))
    CRYPTO_DENYLIST = tuple(_csv_env(
        "BSTOCK_CRYPTO_DENYLIST",
        ",".join((
            "BTC", "ETH", "BNB", "SOL", "XRP", "DOGE", "ADA", "AVAX", "TRX", "TON",
            "SUI", "LINK", "DOT", "SHIB", "PEPE", "WIF", "BONK", "LTC", "BCH", "ATOM",
            "NEAR", "APT", "ARB", "OP", "FIL", "UNI", "AAVE", "MKR", "LDO", "INJ",
            "TIA", "SEI", "FET", "RNDR", "RENDER", "TAO", "WLD", "JUP", "JTO", "PYTH",
            "HYPE", "BOME", "FLOKI", "NOT", "MEME", "ORDI", "SATS", "WBTC", "WETH",
            "USDC", "USDT", "FDUSD", "TUSD", "DAI", "BUSD", "USDE", "DGB", "TRB",
            "CKB", "BB", "YB", "CAKE", "GMX", "DYDX", "CRV", "SNX", "COMP", "YFI",
            "SUSHI", "ENS", "GRT", "FTM", "IMX", "STX", "CFX", "QNT", "ETC", "XLM",
            "HBAR", "VET", "ALGO", "ICP", "SAND", "MANA", "AXS", "GALA", "CHZ",
            "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT",
            "SHIBUSDT", "ARBUSDT",
        )),
    ))

    MIN_QUOTE_VOLUME_24H = float(os.getenv("MIN_QUOTE_VOLUME_24H", "10000"))
    MAX_SYMBOLS = int(os.getenv("MAX_SYMBOLS", "0"))
    SPOT_FEE_BPS = float(os.getenv("SPOT_FEE_BPS", "10"))
    CONCURRENCY = int(os.getenv("BINANCE_CONCURRENCY", "8"))
    TIMEFRAME = BINANCE_INTERVAL
