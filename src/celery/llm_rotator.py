from src.celery.api_key_pool import get_available_key, mark_key_rate_limited

def execute_with_key_failover(func, *args, **kwargs):
    max_attempts = 5
    
    for attempt in range(max_attempts):
        api_key = get_available_key()
        if not api_key:
            raise RuntimeError("ALL_KEYS_EXHAUSTED")

        try:
            # Pass api_key to the function or set it dynamically
            return func(*args, api_key=api_key, **kwargs)

        except Exception as exc:
            err_msg = str(exc).lower()
            if "429" in err_msg or "rate limit" in err_msg or "resource_exhausted" in err_msg:
                print(f"[KEY ROTATOR] Rate limit hit on key ending ...{api_key[-6:]}. Quarantining for 60s...")
                mark_key_rate_limited(api_key, cooldown_seconds=60)
                continue
            # Re-raise non-rate-limit exceptions
            raise exc

    raise RuntimeError("MAX_ROTATION_ATTEMPTS_EXCEEDED")
