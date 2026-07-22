"""Re-ID embedding feature cache with TTL and automatic cleanup."""

import time
from threading import Lock
from typing import Any, Dict, Optional

from monitoring.logging_config import get_logger


class ReIDCache:
    """Thread-safe embedding cache with TTL expiration and memory optimization."""

    def __init__(self, ttl_seconds: float = 300.0, max_entries: int = 1000) -> None:
        self.ttl = ttl_seconds
        self.max_entries = max_entries
        self._logger = get_logger("reid_cache")
        self._lock = Lock()

        # key -> {"embedding": Any, "timestamp": float, "metadata": dict}
        self._cache: Dict[str, Dict[str, Any]] = {}

    def put(self, key: str, embedding: Any, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Store an embedding feature in cache."""
        now = time.monotonic()
        with self._lock:
            if len(self._cache) >= self.max_entries and key not in self._cache:
                self._evict_oldest_unlocked()

            self._cache[key] = {
                "embedding": embedding,
                "timestamp": now,
                "metadata": metadata or {},
            }

    def get(self, key: str) -> Optional[Any]:
        """Retrieve an embedding if present and not expired."""
        now = time.monotonic()
        with self._lock:
            entry = self._cache.get(key)
            if not entry:
                return None

            if (now - entry["timestamp"]) > self.ttl:
                del self._cache[key]
                return None

            return entry["embedding"]

    def cleanup_expired(self) -> int:
        """Purge expired cache entries."""
        now = time.monotonic()
        removed = 0
        with self._lock:
            expired_keys = [
                k for k, v in self._cache.items()
                if (now - v["timestamp"]) > self.ttl
            ]
            for k in expired_keys:
                del self._cache[k]
                removed += 1
        return removed

    def _evict_oldest_unlocked(self) -> None:
        if not self._cache:
            return
        oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k]["timestamp"])
        del self._cache[oldest_key]

    def clear(self) -> None:
        """Clear all entries."""
        with self._lock:
            self._cache.clear()

    def size(self) -> int:
        """Get current size of valid cache entries."""
        self.cleanup_expired()
        with self._lock:
            return len(self._cache)
