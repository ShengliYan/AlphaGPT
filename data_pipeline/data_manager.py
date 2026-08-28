import asyncio

import aiohttp
from loguru import logger

from .config import Config
from .db_manager import DBManager
from .providers.binance_tradfi import BinanceTradFiProvider


class DataManager:
    def __init__(self):
        self.db = DBManager()
        self.provider = BinanceTradFiProvider()
        self._write_lock = asyncio.Lock()

    async def initialize(self):
        self.db.connect()
        self.db.init_schema()
        self.db.prepare_storage_interval(Config.STORAGE_INTERVAL)

    async def close(self):
        self.db.close()

    async def pipeline_sync_daily(self):
        logger.info("Discovering Binance TradFi USDT perpetual universe...")
        timeout = aiohttp.ClientTimeout(total=300)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            universe = await self.provider.discover_universe(session)
            if not universe:
                logger.warning("No TradFi perps passed the filter. Relax MIN_QUOTE_VOLUME_24H or check BINANCE_BASE_URL.")
                return

            crypto_leaks = [s["symbol"] for s in universe if s["symbol"] in ("BTCUSDT", "ETHUSDT")]
            if crypto_leaks:
                raise RuntimeError(f"Refusing to ingest crypto pairs: {crypto_leaks}")

            if Config.MAX_SYMBOLS > 0 and len(universe) > Config.MAX_SYMBOLS:
                seeds = [s for s in universe if s["symbol"] in Config.SEED_SYMBOLS]
                rest = [s for s in universe if s["symbol"] not in Config.SEED_SYMBOLS]
                universe = (seeds + rest)[: Config.MAX_SYMBOLS]
                logger.info(f"Truncated universe to MAX_SYMBOLS={Config.MAX_SYMBOLS}")

            self.db.upsert_symbols(universe)
            last_ms = self.db.last_bar_times_ms()
            logger.info(
                f"Fetching {Config.STORAGE_INTERVAL} klines for {len(universe)} TradFi perps "
                f"({len(last_ms)} symbols already have bars)..."
            )

            total_candles = 0

            async def ingest_one(item: dict) -> int:
                symbol = item["symbol"]
                start_ms = last_ms.get(symbol)
                if start_ms is None:
                    start_ms = item.get("onboard_ms")
                inserted = 0

                async def on_page(page: list[dict]):
                    nonlocal inserted
                    async with self._write_lock:
                        inserted += self.db.batch_insert_ohlcv(page)

                await self.provider.get_klines(
                    session,
                    symbol,
                    interval=Config.STORAGE_INTERVAL,
                    start_ms=start_ms,
                    on_page=on_page,
                )
                return inserted

            batch_size = max(1, Config.CONCURRENCY)
            for i in range(0, len(universe), batch_size):
                batch = universe[i:i + batch_size]
                results = await asyncio.gather(
                    *[ingest_one(item) for item in batch],
                    return_exceptions=True,
                )
                batch_n = 0
                for item, result in zip(batch, results):
                    if isinstance(result, Exception):
                        logger.error(f"Kline fetch failed for {item['symbol']}: {result}")
                        continue
                    batch_n += int(result)
                total_candles += batch_n
                logger.info(f"Processed {i + len(batch)}/{len(universe)} symbols, +{batch_n} 1m candles.")

        self.db.rebuild_resampled_bars()
        logger.success(f"Pipeline complete. New/updated 1m candles: {total_candles}")
