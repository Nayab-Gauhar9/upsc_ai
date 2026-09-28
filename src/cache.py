import os
import json
import redis
from dotenv import load_dotenv

load_dotenv()

try:
    redis_client = redis.Redis(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=6380,
        db=int(os.getenv("REDIS_DB", 0)),
        decode_responses=True,
        socket_connect_timeout=1,
    )
except Exception:
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
