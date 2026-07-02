import time


class SpeedController:
    def __init__(
        self,
        target_fps: int = 15,
    ) -> None:
        self.target_fps = target_fps
        self.frame_interval = 1.0 / max(target_fps, 1)
        self.last_time = 0.0

    def should_process(self) -> bool:
        now = time.time()

        if now - self.last_time >= self.frame_interval:
            self.last_time = now
            return True

        return False

    def wait(self) -> None:
        now = time.time()
        elapsed = now - self.last_time
        remaining = self.frame_interval - elapsed

        if remaining > 0:
            time.sleep(remaining)

        self.last_time = time.time()