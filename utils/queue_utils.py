from queue import Empty, Queue


class SafeQueue:
    def __init__(self, maxsize: int = 0) -> None:
        self.queue = Queue(maxsize=maxsize)

    def put(self, item) -> None:
        self.queue.put(item)

    def get(self, timeout: float | None = None):
        try:
            return self.queue.get(timeout=timeout)
        except Empty:
            return None

    def size(self) -> int:
        return self.queue.qsize()

    def empty(self) -> bool:
        return self.queue.empty()

    def clear(self) -> None:
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except Empty:
                break