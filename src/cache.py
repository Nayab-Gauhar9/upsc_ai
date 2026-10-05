import os
import json
import redis
from typing import Any, Optional
from dotenv import load_dotenv

load_dotenv()

redis_url = os.getenv("REDIS_URL")

try:
    if redis_url:
        # Production Upstash / cloud TLS connection
        redis_client = redis.from_url(
            redis_url,
            decode_responses=True,
            socket_connect_timeout=2,
            protocol=2,
        )
    else:
        # Local development fallback
        redis_client = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", 6380)),
            db=int(os.getenv("REDIS_DB", 0)),
            decode_responses=True,
            socket_connect_timeout=1,
            protocol=2,
        )
except Exception as e:
    print(f"[CACHE INIT WARNING] Failed to initialize Redis: {e}")
    redis_client = None


def get_cached_json(key: str) -> Optional[Any]:
    """Retrieve and deserialize JSON data from Redis."""
    try:
        if redis_client:
            val = redis_client.get(key)
            if val:
                return json.loads(val)
    except Exception as e:
        print(f"[CACHE GET ERROR] {key}: {e}")
    return None


def set_cached_json(key: str, data: Any, ex_seconds: Optional[int] = 86400) -> None:
    """Serialize and write data to Redis with an explicit TTL."""
    try:
        if redis_client:
            redis_client.set(key, json.dumps(data), ex=ex_seconds)
    except Exception as e:
        print(f"[CACHE SET ERROR] {key}: {e}")


def get_cached_chapter(chapter_id: int) -> dict | None:
    return get_cached_json(f"laxmikanth:chapter:{chapter_id}")


def set_cached_chapter(chapter_id: int, data: dict, expire_days: int = 30) -> None:
    set_cached_json(f"laxmikanth:chapter:{chapter_id}", data, ex_seconds=expire_days * 86400)
