import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

_ROOT = Path(__file__).resolve().parents[1]


def _csv_env(name: str, default: str) -> list[str]:
    raw = os.getenv(name, default)
    return [item.strip().upper() for item in raw.split(",") if item.strip()]


def _flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


class Config:
    DB_PATH = os.getenv("DUCKDB_PATH", str(_ROOT / "data" / "alphagpt.duckdb"))

    # USD-M futures live on www.binance.com; fapi.binance.com is geo-blocked (HTTP 451) here.
    BINANCE_BASE_URL = os.getenv("BINANCE_BASE_URL", "https://www.binance.com")
    BINANCE_FALLBACK_URLS = (
        "https://www.binance.com",
        "https://fapi.binance.com",
    )
    BINANCE_INTERVAL = os.getenv("BINANCE_INTERVAL", "1h")
    BINANCE_KLINE_LIMIT = int(os.getenv("BINANCE_KLINE_LIMIT", "1000"))
    BINANCE_START = os.getenv("BINANCE_START", "2026-01-01")
    BINANCE_FAPI_PREFIX = os.getenv("BINANCE_FAPI_PREFIX", "/fapi/v1")

    CONTRACT_TYPE = os.getenv("TRADFI_CONTRACT_TYPE", "TRADIFI_PERPETUAL")
    UNDERLYING_TYPES = tuple(_csv_env(
        "TRADFI_UNDERLYING_TYPES",
        "EQUITY,KR_EQUITY,HK_EQUITY,CN_EQUITY,PREMARKET",
    ))
    INCLUDE_COMMODITIES = _flag("TRADFI_INCLUDE_COMMODITIES", "0")

    SEED_SYMBOLS = tuple(_csv_env(
        "TRADFI_SEED_SYMBOLS",
        "MUUSDT,LITEUSDT,TSLAUSDT,NVDAUSDT,CRCLUSDT,SNDKUSDT",
    ))
    CRYPTO_DENYLIST = tuple(_csv_env(
        "TRADFI_CRYPTO_DENYLIST",
        ",".join((
            "BTC", "ETH", "BNB", "SOL", "XRP", "DOGE", "ADA", "AVAX", "TRX", "TON",
            "SUI", "LINK", "DOT", "SHIB", "PEPE", "WIF", "BONK", "LTC", "BCH", "ATOM",
            "NEAR", "APT", "ARB", "OP", "FIL", "UNI", "AAVE", "MKR", "LDO", "INJ",
            "TIA", "SEI", "FET", "RNDR", "RENDER", "TAO", "WLD", "JUP", "JTO", "PYTH",
            "HYPE", "BOME", "FLOKI", "NOT", "MEME", "ORDI", "SATS", "WBTC", "WETH",
            "USDC", "USDT", "FDUSD", "TUSD", "DAI", "BUSD", "USDE",
            "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT",
            "SHIBUSDT", "ARBUSDT",
        )),
    ))

    MIN_QUOTE_VOLUME_24H = float(os.getenv("MIN_QUOTE_VOLUME_24H", "10000"))
    MAX_SYMBOLS = int(os.getenv("MAX_SYMBOLS", "0"))
    FEE_BPS = float(os.getenv("FEE_BPS", os.getenv("SPOT_FEE_BPS", "5")))
    CONCURRENCY = int(os.getenv("BINANCE_CONCURRENCY", "8"))
    TIMEFRAME = BINANCE_INTERVAL
