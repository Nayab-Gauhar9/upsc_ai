import os
import time
import redis


# Connects to your WSL Redis on port 6380
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6380))

redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=1, decode_responses=True)

KEY_POOL_LIST = "pool:groq:keys"
COOLDOWN_PREFIX = "cooldown:groq:"


def init_api_key_pool(keys: list[str]):
    """Initialize Redis key pool if empty."""
    if not redis_client.exists(KEY_POOL_LIST):
        for k in keys:
            redis_client.lpush(KEY_POOL_LIST, k.strip())
        print(f"Initialized API key pool with {len(keys)} keys.")


def get_available_key() -> str | None:
    """
    Round-robin pop and push to cycle keys.
    Skips keys currently marked in cooldown.
    """
    total_keys = redis_client.llen(KEY_POOL_LIST)
    if total_keys == 0:
        return None

    for _ in range(total_keys):
        # Rotate key atomically: move tail item to head
        key = redis_client.rpoplpush(KEY_POOL_LIST, KEY_POOL_LIST)
        print(f"[ROUND-ROBIN] Dispatched Key: ...{key[-6:]}")
        
        # Check if key is currently rate-limited/in cooldown
        cooldown_key = f"{COOLDOWN_PREFIX}{key[-8:]}"
        if not redis_client.exists(cooldown_key):
            return key
            
    return None  # All keys are currently rate-limited


def mark_key_rate_limited(key: str, cooldown_seconds: int = 60):
    """Mark a key as rate-limited for cooldown_seconds."""
    key_tag = key[-8:]
    redis_client.set(f"{COOLDOWN_PREFIX}{key_tag}", "rate_limited", ex=cooldown_seconds)
    print(f"Key ending with '...{key_tag}' marked in cooldown for {cooldown_seconds}s.")

raw_keys = os.getenv("GROQ_API_KEYS", "")
GROQ_KEYS = [k.strip() for k in raw_keys.split(",") if k.strip()]

if GROQ_KEYS:
  init_api_key_pool(GROQ_KEYS)