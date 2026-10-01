import os
import json
import redis
from dotenv import load_dotenv

load_dotenv()

redis_url = os.getenv("REDIS_URL")

try:
    if redis_url:
        # Connect via full connection URL (e.g. rediss://default:... for Upstash cloud)
        redis_client = redis.from_url(
            redis_url,
            decode_responses=True,
            socket_connect_timeout=2,
        )
    else:
        # Fallback to local host & port
        redis_client = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", 6379)),
            db=int(os.getenv("REDIS_DB", 0)),
            decode_responses=True,
            socket_connect_timeout=1,
        )
except Exception as e:
    print(f"[CACHE INIT WARNING] Failed to initialize Redis: {e}")
    redis_client = None


def get_cached_chapter(chapter_id: int) -> dict | None:
    try:
        if redis_client:
            cached = redis_client.get(f"laxmikanth:chapter:{chapter_id}")
            if cached:
                print(f"[CACHE HIT] Loaded Ch.{chapter_id} from Redis")
                return json.loads(cached)
    except Exception as e:
        print(f"[CACHE ERROR GET] {e}")
    return None


def set_cached_chapter(chapter_id: int, data: dict, expire_days: int = 30) -> None:
    try:
        if redis_client:
            redis_client.set(
                f"laxmikanth:chapter:{chapter_id}",
                json.dumps(data),
                ex=expire_days * 86400,
            )
            print(f"[CACHE SET] Stored Ch.{chapter_id} in Redis")
    except Exception as e:
        print(f"[CACHE ERROR SET] {e}")
