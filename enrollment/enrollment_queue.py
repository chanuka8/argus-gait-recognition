from queue import Queue


class EnrollmentQueue:
    def __init__(self) -> None:
        self._queue = Queue()

    def add(self, person_folder: str) -> None:
        self._queue.put(person_folder)

    def get(self):
        if self._queue.empty():
            return None

        return self._queue.get()

    def size(self) -> int:
        return self._queue.qsize()

    def is_empty(self) -> bool:
        return self._queue.empty()