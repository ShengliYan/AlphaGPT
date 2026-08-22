import asyncio

import aiohttp
from loguru import logger

from .config import Config
from .db_manager import DBManager
from .providers.binance_bstocks import BinanceBStocksProvider


class DataManager:
    def __init__(self):
        self.db = DBManager()
        self.provider = BinanceBStocksProvider()

    async def initialize(self):
        self.db.connect()
        self.db.init_schema()

    async def close(self):
        self.db.close()

    async def pipeline_sync_daily(self):
        logger.info("Discovering Binance bStocks universe...")
        timeout = aiohttp.ClientTimeout(total=300)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            universe = await self.provider.discover_universe(session)
            if not universe:
                logger.warning("No bStocks passed the filter. Relax MIN_QUOTE_VOLUME_24H or check BINANCE_BASE_URL.")
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
            logger.info(f"Fetching {Config.BINANCE_INTERVAL} klines for {len(universe)} symbols...")

            total_candles = 0
            batch_size = max(1, Config.CONCURRENCY)
            for i in range(0, len(universe), batch_size):
                batch = universe[i:i + batch_size]
                tasks = [self.provider.get_klines(session, item["symbol"]) for item in batch]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                records = []
                for item, result in zip(batch, results):
                    if isinstance(result, Exception):
                        logger.error(f"Kline fetch failed for {item['symbol']}: {result}")
                        continue
                    records.extend(result)
                inserted = self.db.batch_insert_ohlcv(records)
                total_candles += inserted
                logger.info(f"Processed {i + len(batch)}/{len(universe)} symbols, +{inserted} candles.")

        logger.success(f"Pipeline complete. Total candles stored: {total_candles}")
