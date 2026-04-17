import json
import os
import time
from collections import OrderedDict

try:
    import redis
except Exception:  # pragma: no cover - optional dependency fallback
    redis = None


class CacheClient:
    def __init__(self):
        self._memory = OrderedDict()
        self._memory_max_items = max(128, int(os.getenv("IN_MEMORY_CACHE_MAX_ITEMS", "2048")))
        self._redis = None
        redis_url = os.getenv("REDIS_URL")
        if redis and redis_url:
            try:
                self._redis = redis.Redis.from_url(redis_url, decode_responses=True)
                self._redis.ping()
            except Exception:
                self._redis = None

    def get_json(self, key):
        if self._redis is not None:
            raw = self._redis.get(key)
            return json.loads(raw) if raw else None

        item = self._memory.get(key)
        if not item:
            return None
        self._memory.move_to_end(key)
        expires_at, value = item
        if expires_at and expires_at < time.time():
            self._memory.pop(key, None)
            return None
        return value

    def set_json(self, key, value, ttl_seconds):
        if self._redis is not None:
            self._redis.setex(key, int(ttl_seconds), json.dumps(value))
            return
        expires_at = time.time() + int(ttl_seconds) if ttl_seconds else None
        self._memory[key] = (expires_at, value)
        self._memory.move_to_end(key)
        while len(self._memory) > self._memory_max_items:
            self._memory.popitem(last=False)


_CACHE = CacheClient()


def get_cache():
    return _CACHE
