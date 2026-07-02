class SystemContext:

    def __init__(self):

        self._state = {}

    def set(self, key, value):

        self._state[key] = value

    def get(self, key, default=None):

        return self._state.get(key, default)

    def exists(self, key):

        return key in self._state

    def remove(self, key):

        if key in self._state:
            del self._state[key]

    def clear(self):

        self._state.clear()

    def keys(self):

        return list(self._state.keys())

    def as_dict(self):

        return dict(self._state)