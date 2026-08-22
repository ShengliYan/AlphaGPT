import asyncio
from datetime import datetime, timezone
from typing import Any

import aiohttp
from loguru import logger

from ..config import Config

CRYPTO_LEAK_SYMBOLS = ("BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT")


def is_tradfi_usdt_perp(
    info: dict,
    seed_symbols: tuple[str, ...] | None = None,
    denylist: tuple[str, ...] | None = None,
) -> bool:
    symbol = str(info.get("symbol") or "").upper()
    base = str(info.get("baseAsset") or "").upper()
    quote = str(info.get("quoteAsset") or info.get("marginAsset") or "").upper()
    status = str(info.get("status") or "").upper()
    contract_type = str(info.get("contractType") or "").upper()
    underlying_type = str(info.get("underlyingType") or "").upper()
    seeds = set(seed_symbols if seed_symbols is not None else Config.SEED_SYMBOLS)
    denied = set(denylist if denylist is not None else Config.CRYPTO_DENYLIST)

    if status != "TRADING" or quote != "USDT":
        return False
    if symbol in denied or base in denied:
        return False
    if contract_type != Config.CONTRACT_TYPE:
        return False
    if symbol in seeds:
        return True
    if underlying_type == "COMMODITY" and not Config.INCLUDE_COMMODITIES:
        return False
    if Config.UNDERLYING_TYPES and underlying_type not in Config.UNDERLYING_TYPES:
        if underlying_type == "COMMODITY" and Config.INCLUDE_COMMODITIES:
            return True
        return False
    return True


class BinanceTradFiProvider:
    def __init__(self):
        self.base_url = Config.BINANCE_BASE_URL.rstrip("/")
        self.fallback_urls = [u.rstrip("/") for u in Config.BINANCE_FALLBACK_URLS]
        self.semaphore = asyncio.Semaphore(Config.CONCURRENCY)
        self.headers = {"User-Agent": "AlphaGPT-tradfi-research/1.0", "accept": "application/json"}
        self.prefix = Config.BINANCE_FAPI_PREFIX.rstrip("/")

    def _url(self, path: str, root: str | None = None) -> str:
        if not path.startswith("/"):
            path = "/" + path
        return f"{(root or self.base_url).rstrip('/')}{path}"

    def _fapi(self, endpoint: str) -> str:
        if not endpoint.startswith("/"):
            endpoint = "/" + endpoint
        return f"{self.prefix}{endpoint}"

    async def _get_json(self, session: aiohttp.ClientSession, path: str, params: dict | None = None) -> Any:
        urls = [self.base_url] + [u for u in self.fallback_urls if u != self.base_url]
        last_error = None
        for root in urls:
            url = self._url(path, root)
            try:
                async with session.get(url, params=params, headers=self.headers) as resp:
                    if resp.status == 200:
                        self.base_url = root
                        return await resp.json()
                    if resp.status in (418, 429):
                        retry_after = float(resp.headers.get("Retry-After", "2"))
                        logger.warning(f"Binance {resp.status} on {path}, retrying in {retry_after}s")
                        await asyncio.sleep(retry_after)
                        async with session.get(url, params=params, headers=self.headers) as retry:
                            retry.raise_for_status()
                            self.base_url = root
                            return await retry.json()
                    if resp.status in (403, 451):
                        logger.warning(f"Binance {resp.status} at {root}, trying fallback host")
                        last_error = RuntimeError(f"HTTP {resp.status} from {root}")
                        continue
                    body = await resp.text()
                    raise RuntimeError(f"Binance HTTP {resp.status} {path}: {body[:200]}")
            except aiohttp.ClientError as exc:
                last_error = exc
                logger.warning(f"Binance request failed at {root}: {exc}")
        raise RuntimeError(f"All Binance hosts failed for {path}: {last_error}")

    async def get_exchange_info(self, session: aiohttp.ClientSession) -> list[dict]:
        data = await self._get_json(session, self._fapi("/exchangeInfo"))
        return data.get("symbols") or []

    async def get_ticker_quote_volumes(self, session: aiohttp.ClientSession) -> dict[str, float]:
        data = await self._get_json(session, self._fapi("/ticker/24hr"))
        volumes: dict[str, float] = {}
        if isinstance(data, dict):
            data = [data]
        for row in data:
            try:
                volumes[row["symbol"]] = float(row.get("quoteVolume") or 0.0)
            except (TypeError, ValueError, KeyError):
                continue
        return volumes

    def _symbol_record(self, info: dict) -> dict:
        base = info.get("baseAsset", "")
        quote = info.get("quoteAsset") or info.get("marginAsset") or ""
        onboard = info.get("onboardDate")
        try:
            onboard_ms = int(onboard) if onboard else None
        except (TypeError, ValueError):
            onboard_ms = None
        return {
            "symbol": info.get("symbol", ""),
            "base": base,
            "quote": quote,
            "underlying": base,
            "name": f"{base} {info.get('underlyingType') or 'TradFi'}".strip(),
            "status": info.get("status", ""),
            "contract_type": info.get("contractType", ""),
            "underlying_type": info.get("underlyingType", ""),
            "onboard_ms": onboard_ms,
        }

    async def discover_universe(self, session: aiohttp.ClientSession) -> list[dict]:
        infos = await self.get_exchange_info(session)
        volumes = await self.get_ticker_quote_volumes(session)
        selected: dict[str, dict] = {}

        for info in infos:
            if not is_tradfi_usdt_perp(info):
                continue
            record = self._symbol_record(info)
            symbol = record["symbol"]
            quote_vol = volumes.get(symbol, 0.0)
            record["quote_volume_24h"] = quote_vol
            is_seed = symbol in Config.SEED_SYMBOLS
            if (not is_seed) and quote_vol < Config.MIN_QUOTE_VOLUME_24H:
                continue
            selected[symbol] = record

        for seed in Config.SEED_SYMBOLS:
            if seed in selected:
                continue
            match = next((info for info in infos if info.get("symbol") == seed), None)
            if not match:
                logger.warning(f"Seed symbol {seed} not found on USD-M futures")
                continue
            if not is_tradfi_usdt_perp(match):
                logger.warning(
                    f"Seed {seed} is not a TRADING {Config.CONTRACT_TYPE} USDT contract "
                    f"(type={match.get('contractType')} status={match.get('status')})"
                )
                continue
            record = self._symbol_record(match)
            record["quote_volume_24h"] = volumes.get(seed, 0.0)
            selected[seed] = record

        universe = sorted(selected.values(), key=lambda r: r.get("quote_volume_24h", 0.0), reverse=True)
        blocked = [s for s in CRYPTO_LEAK_SYMBOLS if s in selected]
        if blocked:
            raise RuntimeError(f"Crypto pairs leaked into universe: {blocked}")
        logger.info(f"TradFi USDT perps: {len(universe)} symbols")
        return universe

    @staticmethod
    def _parse_kline(symbol: str, row: list) -> dict:
        open_ms = int(row[0])
        return {
            "time": datetime.fromtimestamp(open_ms / 1000.0, tz=timezone.utc).replace(tzinfo=None),
            "symbol": symbol,
            "open": float(row[1]),
            "high": float(row[2]),
            "low": float(row[3]),
            "close": float(row[4]),
            "volume": float(row[5]),
            "quote_volume": float(row[7]),
            "n_trades": int(row[8]),
            "source": "binance_fapi",
        }

    def _default_start_ms(self) -> int:
        start_dt = datetime.strptime(Config.BINANCE_START, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        return int(start_dt.timestamp() * 1000)

    async def get_klines(
        self,
        session: aiohttp.ClientSession,
        symbol: str,
        interval: str | None = None,
        start_ms: int | None = None,
        end_ms: int | None = None,
    ) -> list[dict]:
        interval = interval or Config.BINANCE_INTERVAL
        cursor = start_ms if start_ms is not None else self._default_start_ms()

        candles: list[dict] = []
        while True:
            params = {
                "symbol": symbol,
                "interval": interval,
                "limit": Config.BINANCE_KLINE_LIMIT,
                "startTime": cursor,
            }
            if end_ms is not None:
                params["endTime"] = end_ms
            async with self.semaphore:
                data = await self._get_json(session, self._fapi("/klines"), params=params)
            if not data:
                break
            candles.extend(self._parse_kline(symbol, row) for row in data)
            if len(data) < Config.BINANCE_KLINE_LIMIT:
                break
            next_open = int(data[-1][0]) + 1
            if next_open <= cursor:
                break
            cursor = next_open
            await asyncio.sleep(0.05)
        return candles
