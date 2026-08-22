import asyncio
import re
from datetime import datetime, timezone
from typing import Any

import aiohttp
from loguru import logger

from ..config import Config

BSTOCK_BASE_RE = re.compile(r"^[A-Z]{1,5}B$")
CRYPTO_TRADING_GROUP = "TRD_GRP_004"


def underlying_from_base(base: str) -> str:
    if base.endswith("B") and len(base) >= 2:
        return base[:-1]
    return base


def _permission_groups(info: dict) -> set[str]:
    groups: set[str] = set()
    for item in info.get("permissions") or []:
        groups.add(str(item))
    for perm_set in info.get("permissionSets") or []:
        groups.update(str(x) for x in perm_set)
    return groups


def is_bstock_candidate(
    symbol: str,
    base: str,
    quote: str,
    status: str = "TRADING",
    permission_groups: set[str] | None = None,
    seed_symbols: tuple[str, ...] | None = None,
    denylist: tuple[str, ...] | None = None,
) -> bool:
    symbol = (symbol or "").upper()
    base = (base or "").upper()
    quote = (quote or "").upper()
    status = (status or "").upper()
    seeds = set(seed_symbols if seed_symbols is not None else Config.SEED_SYMBOLS)
    denied = set(denylist if denylist is not None else Config.CRYPTO_DENYLIST)

    if quote != "USDT" or status != "TRADING":
        return False
    if symbol in denied or base in denied:
        return False
    if symbol in seeds:
        return True
    if not BSTOCK_BASE_RE.match(base):
        return False
    underlying = underlying_from_base(base)
    if underlying in denied:
        return False
    if permission_groups and CRYPTO_TRADING_GROUP in permission_groups:
        return False
    return True


class BinanceBStocksProvider:
    def __init__(self):
        self.base_url = Config.BINANCE_BASE_URL.rstrip("/")
        self.fallback_urls = [u.rstrip("/") for u in Config.BINANCE_FALLBACK_URLS]
        self.semaphore = asyncio.Semaphore(Config.CONCURRENCY)
        self.headers = {"User-Agent": "AlphaGPT-bStocks-research/1.0", "accept": "application/json"}

    def _url(self, path: str, root: str | None = None) -> str:
        return f"{(root or self.base_url).rstrip('/')}{path}"

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
        data = await self._get_json(session, "/api/v3/exchangeInfo")
        return data.get("symbols") or []

    async def get_ticker_quote_volumes(self, session: aiohttp.ClientSession) -> dict[str, float]:
        data = await self._get_json(session, "/api/v3/ticker/24hr")
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
        quote = info.get("quoteAsset", "")
        underlying = underlying_from_base(base)
        return {
            "symbol": info.get("symbol", ""),
            "base": base,
            "quote": quote,
            "underlying": underlying,
            "name": underlying,
            "status": info.get("status", ""),
        }

    async def discover_universe(self, session: aiohttp.ClientSession) -> list[dict]:
        infos = await self.get_exchange_info(session)
        volumes = await self.get_ticker_quote_volumes(session)
        selected: dict[str, dict] = {}

        for info in infos:
            if not info.get("isSpotTradingAllowed", True):
                continue
            symbol = info.get("symbol", "")
            record = self._symbol_record(info)
            groups = _permission_groups(info)
            if not is_bstock_candidate(
                record["symbol"],
                record["base"],
                record["quote"],
                record["status"],
                permission_groups=groups,
            ):
                continue
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
                logger.warning(f"Seed symbol {seed} not found in exchangeInfo")
                continue
            record = self._symbol_record(match)
            if record["quote"] != "USDT" or record["status"] != "TRADING":
                logger.warning(f"Seed symbol {seed} is not a TRADING USDT pair")
                continue
            record["quote_volume_24h"] = volumes.get(seed, 0.0)
            selected[seed] = record

        universe = sorted(selected.values(), key=lambda r: r.get("quote_volume_24h", 0.0), reverse=True)
        blocked = [s for s in ("BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT") if s in selected]
        if blocked:
            raise RuntimeError(f"Crypto pairs leaked into universe: {blocked}")
        logger.info(f"bStocks universe: {len(universe)} symbols")
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
            "source": "binance",
        }

    async def get_klines(
        self,
        session: aiohttp.ClientSession,
        symbol: str,
        interval: str | None = None,
        start_ms: int | None = None,
        end_ms: int | None = None,
    ) -> list[dict]:
        interval = interval or Config.BINANCE_INTERVAL
        if start_ms is None:
            start_dt = datetime.strptime(Config.BINANCE_START, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            start_ms = int(start_dt.timestamp() * 1000)

        candles: list[dict] = []
        cursor = start_ms
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
                data = await self._get_json(session, "/api/v3/klines", params=params)
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
