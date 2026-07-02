from collections import OrderedDict


class CacheManager:
    def __init__(self, max_items: int = 1000):
        self.max_items = max_items
        self.cache = OrderedDict()

    def put(self, key, value):
        if key in self.cache:
            self.cache.pop(key)

        self.cache[key] = value

        if len(self.cache) > self.max_items:
            self.cache.popitem(last=False)

    def get(self, key, default=None):
        return self.cache.get(key, default)

    def contains(self, key):
        return key in self.cache

    def clear(self):
        self.cache.clear()

    def size(self):
        return len(self.cache)