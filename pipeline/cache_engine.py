from storage.cache_manager import CacheManager


class CacheEngine:
    def __init__(self, max_items: int = 500) -> None:
        self.cache = CacheManager(max_items=max_items)

    def get_embedding(self, key):
        return self.cache.get(key)

    def save_embedding(self, key, embedding) -> None:
        self.cache.put(key, embedding)

    def has_embedding(self, key) -> bool:
        return self.cache.contains(key)

    def clear(self) -> None:
        self.cache.clear()

    def size(self) -> int:
        return self.cache.size()