from app.config.redis import get_redis


async def root() -> dict[str, str]:
    redis_client = await get_redis()
    redis_status = "disconnected"
    if redis_client:
        try:
            if await redis_client.ping():
                redis_status = "connected"
        except Exception:
            redis_status = "disconnected"

    return {
        "status": "AI service running",
        "redis": redis_status,
    }

