"""
Multi-stream engine for ARGUS multi-camera support.

Manages multiple camera feeds with independent capture threads.
Each camera runs in its own daemon thread with FPS limiting.
Frames are placed into per-camera thread-safe queues for
consumption by the main recognition pipeline.
"""

import time
import threading
from queue import Queue, Full, Empty

import cv2


class CameraStream:
    """Thread-safe single camera stream with FPS limiting."""

    def __init__(
        self,
        camera_id: str,
        source=0,
        width: int = 640,
        height: int = 480,
        target_fps: int = 5,
        queue_max_size: int = 10,
    ) -> None:
        self.camera_id = camera_id
        self.source = source
        self.width = width
        self.height = height
        self.target_fps = max(1, target_fps)
        self.frame_interval = 1.0 / self.target_fps

        self.queue: Queue = Queue(maxsize=queue_max_size)
        self.cap = None
        self.running = False
        self.thread = None
        self.error = None
        self.frames_read = 0
        self.frames_dropped = 0

    def start(self) -> bool:
        """Open the video source and start the capture thread."""
        self.cap = cv2.VideoCapture(self.source)

        if not self.cap.isOpened():
            self.error = (
                f"Camera {self.camera_id}: "
                f"Failed to open source {self.source}"
            )
            return False

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        self.running = True
        self.thread = threading.Thread(
            target=self._capture_loop,
            daemon=True,
            name=f"cam-{self.camera_id}",
        )
        self.thread.start()

        return True

    def _capture_loop(self) -> None:
        """Background capture loop that reads frames at target FPS."""
        while self.running:
            try:
                ret, frame = self.cap.read()

                if not ret:
                    self.error = (
                        f"Camera {self.camera_id}: "
                        f"Failed to read frame"
                    )
                    self.running = False
                    break

                self.frames_read += 1

                try:
                    self.queue.put_nowait(frame)
                except Full:
                    # Drop oldest frame and push new one
                    try:
                        self.queue.get_nowait()
                    except Empty:
                        pass

                    try:
                        self.queue.put_nowait(frame)
                    except Full:
                        pass

                    self.frames_dropped += 1

                time.sleep(self.frame_interval)

            except Exception as e:
                self.error = (
                    f"Camera {self.camera_id}: {e}"
                )
                self.running = False
                break

    def read(self):
        """
        Non-blocking read of the latest frame.

        Returns:
            (success: bool, frame or None)
        """
        try:
            return True, self.queue.get_nowait()
        except Empty:
            return False, None

    def is_opened(self) -> bool:
        """Check if the camera stream is still running."""
        return (
            self.running
            and self.cap is not None
            and self.cap.isOpened()
        )

    def stop(self) -> None:
        """Stop the capture thread and release the camera."""
        self.running = False

        if self.thread is not None:
            self.thread.join(timeout=3)

        if self.cap is not None:
            self.cap.release()

    def stats(self) -> dict:
        """Return stream statistics."""
        return {
            "camera_id": self.camera_id,
            "source": self.source,
            "frames_read": self.frames_read,
            "frames_dropped": self.frames_dropped,
            "is_running": self.running,
            "error": self.error,
        }


class MultiStreamEngine:
    """
    Manages multiple CameraStream instances.

    Each camera feed runs in its own background thread.
    The engine provides a unified interface for reading
    frames and checking status across all cameras.
    """

    def __init__(
        self,
        camera_configs: list[dict],
        queue_max_size: int = 10,
    ) -> None:
        self.streams: dict[str, CameraStream] = {}

        for cam_cfg in camera_configs:
            camera_id = str(cam_cfg["id"])

            self.streams[camera_id] = CameraStream(
                camera_id=camera_id,
                source=cam_cfg.get("source", 0),
                width=cam_cfg.get("width", 640),
                height=cam_cfg.get("height", 480),
                target_fps=cam_cfg.get("target_fps", 5),
                queue_max_size=queue_max_size,
            )

    def start_all(self) -> dict[str, bool]:
        """
        Start all camera streams.

        Returns:
            dict mapping camera_id to start success boolean.
        """
        results = {}

        for camera_id, stream in self.streams.items():
            success = stream.start()
            results[camera_id] = success

            if not success:
                print(
                    f"[WARNING] {stream.error}"
                )

        return results

    def read(self, camera_id: str):
        """
        Non-blocking read from a specific camera.

        Returns:
            (success: bool, frame or None)
        """
        if camera_id not in self.streams:
            return False, None

        return self.streams[camera_id].read()

    def active_cameras(self) -> list[str]:
        """Return list of camera IDs with active streams."""
        return [
            cid
            for cid, stream in self.streams.items()
            if stream.is_opened()
        ]

    def stop_all(self) -> None:
        """Stop all camera streams and release resources."""
        for stream in self.streams.values():
            stream.stop()

    def stats(self) -> dict[str, dict]:
        """Return statistics for all camera streams."""
        return {
            cid: stream.stats()
            for cid, stream in self.streams.items()
        }
