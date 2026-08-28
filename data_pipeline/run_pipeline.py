import asyncio

from loguru import logger

from .data_manager import DataManager


async def main():
    manager = DataManager()
    try:
        await manager.initialize()
        await manager.pipeline_sync_daily()
    except Exception as e:
        logger.exception(f"Pipeline crashed: {e}")
        raise
    finally:
        await manager.close()


if __name__ == "__main__":
    asyncio.run(main())
