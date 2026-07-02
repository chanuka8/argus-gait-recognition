from queue import Empty, Full, Queue
from threading import Lock


class BufferQueue:
    def __init__(
        self,
        maxsize: int = 5,
    ) -> None:
        self.queue = Queue(maxsize=maxsize)
        self.lock = Lock()

    def push(self, frame) -> bool:
        with self.lock:
            try:
                self.queue.put_nowait(frame)
                return True
            except Full:
                return False

    def pop(self):
        with self.lock:
            try:
                return self.queue.get_nowait()
            except Empty:
                return None

    def size(self) -> int:
        return self.queue.qsize()

    def is_empty(self) -> bool:
        return self.queue.empty()

    def is_full(self) -> bool:
        return self.queue.full()

    def clear(self) -> None:
        with self.lock:
            while not self.queue.empty():
                try:
                    self.queue.get_nowait()
                except Empty:
                    break