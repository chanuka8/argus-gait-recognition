class FrameDropper:
    def __init__(
        self,
        max_queue_size: int = 5,
    ) -> None:
        self.max_queue_size = max_queue_size
        self.dropped_frames = 0

    def should_drop(
        self,
        queue_size: int,
    ) -> bool:
        return queue_size >= self.max_queue_size

    def record_drop(self) -> None:
        self.dropped_frames += 1

    def reset(self) -> None:
        self.dropped_frames = 0

    def stats(self) -> dict:
        return {
            "dropped_frames": self.dropped_frames,
            "max_queue_size": self.max_queue_size,
        }