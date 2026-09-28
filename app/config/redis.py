import structlog
import redis.asyncio as aioredis
from app.config.settings import get_settings

logger = structlog.get_logger("vyaparsathi.ai.redis")

_redis_client: aioredis.Redis | None = None


def get_clean_redis_url() -> str | None:
    settings = get_settings()
    url = settings.redis_url
    if not url:
        return None
    if url.startswith("redis-cli -u "):
        url = url.replace("redis-cli -u ", "").strip()
    return url


async def init_redis() -> aioredis.Redis | None:
    global _redis_client
    redis_url = get_clean_redis_url()
    if not redis_url:
        logger.warning("redis_url_not_configured")
        return None

    try:
        _redis_client = aioredis.from_url(
            redis_url,
            encoding="utf-8",
            decode_responses=True,
            socket_timeout=5.0,
            socket_connect_timeout=5.0,
        )
        await _redis_client.ping()
        logger.info("redis_connected_successfully")
        return _redis_client
    except Exception as e:
        logger.error("redis_connection_failed", error=str(e))
        _redis_client = None
        return None


async def get_redis() -> aioredis.Redis | None:
    global _redis_client
    if _redis_client is None:
        await init_redis()
    return _redis_client


async def close_redis() -> None:
    global _redis_client
    if _redis_client is not None:
        await _redis_client.aclose()
        _redis_client = None
        logger.info("redis_connection_closed")
