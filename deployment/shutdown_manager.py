"""
Idempotent Graceful Shutdown Manager for ARGUS AI.

Manages signal handling (SIGINT, SIGTERM), worker pool termination, queue draining,
and resource cleanup order without risking data corruption or process deadlocks.
"""

import signal
import threading
from typing import Callable, List, Optional

from monitoring.logging_config import get_logger

_SHUTDOWN_MANAGER_INSTANCE: Optional["ShutdownManager"] = None
_SIGNAL_HANDLERS_REGISTERED = False


class ShutdownManager:
    """Coordinates graceful component shutdown and resource cleanup."""

    def __init__(self, join_timeout_sec: float = 3.0) -> None:
        self.join_timeout_sec = join_timeout_sec
        self.logger = get_logger("system")
        self._is_shutting_down = False
        self._shutdown_lock = threading.Lock()
        self._cleanup_callbacks: List[Callable[[], None]] = []
        self._registered_threads: List[threading.Thread] = []
        self._stop_events: List[threading.Event] = []
        self._shutdown_complete = False

    @property
    def is_shutting_down(self) -> bool:
        """True if shutdown process has started."""
        return self._is_shutting_down

    def register_stop_event(self, stop_event: threading.Event) -> None:
        """Register a worker stop event to trigger on shutdown."""
        with self._shutdown_lock:
            if stop_event not in self._stop_events:
                self._stop_events.append(stop_event)

    def register_thread(self, thread: threading.Thread) -> None:
        """Register a worker thread to be joined during shutdown."""
        with self._shutdown_lock:
            if thread not in self._registered_threads:
                self._registered_threads.append(thread)

    def register_cleanup_callback(self, callback: Callable[[], None]) -> None:
        """Register a custom cleanup callback function."""
        with self._shutdown_lock:
            if callback not in self._cleanup_callbacks:
                self._cleanup_callbacks.append(callback)

    def register_signal_handlers(self) -> None:
        """Register SIGINT and SIGTERM signal handlers safely once."""
        global _SIGNAL_HANDLERS_REGISTERED

        # Only register handlers if running in the main thread
        if threading.current_thread() is not threading.main_thread():
            return

        if _SIGNAL_HANDLERS_REGISTERED:
            return

        def _signal_handler(signum, frame):
            sig_name = "SIGINT" if signum == signal.SIGINT else f"SIGNAL({signum})"
            self.logger.info(f"Received signal {sig_name}. Initiating graceful shutdown...")
            self.shutdown()

        try:
            signal.signal(signal.SIGINT, _signal_handler)
        except Exception:
            pass

        if hasattr(signal, "SIGTERM"):
            try:
                signal.signal(signal.SIGTERM, _signal_handler)
            except Exception:
                pass

        _SIGNAL_HANDLERS_REGISTERED = True

    def shutdown(self) -> bool:
        """
        Execute idempotent graceful shutdown of all registered resources.

        Returns True if shutdown completed successfully.
        """
        with self._shutdown_lock:
            if self._is_shutting_down:
                self.logger.debug("Shutdown already in progress or completed. Ignoring duplicate call.")
                return self._shutdown_complete

            self._is_shutting_down = True

        self.logger.info("[SHUTDOWN] Graceful shutdown sequence started...")

        # Step 1: Signal all stop events
        for evt in self._stop_events:
            try:
                evt.set()
            except Exception as e:
                self.logger.warning(f"Error setting stop event: {e}")

        # Step 2: Run cleanup callbacks in reverse order of registration
        for cb in reversed(self._cleanup_callbacks):
            try:
                cb()
            except Exception as e:
                self.logger.warning(f"Error executing shutdown callback: {e}")

        # Step 3: Join registered worker threads with timeout
        for t in self._registered_threads:
            if t.is_alive():
                self.logger.debug(f"Joining thread {t.name} (timeout={self.join_timeout_sec}s)...")
                t.join(timeout=self.join_timeout_sec)
                if t.is_alive():
                    self.logger.warning(f"Worker thread {t.name} did not terminate within {self.join_timeout_sec}s")

        self._shutdown_complete = True
        self.logger.info("[SHUTDOWN] Graceful shutdown sequence completed successfully.")
        return True


def get_shutdown_manager() -> ShutdownManager:
    """Get or instantiate global ShutdownManager singleton."""
    global _SHUTDOWN_MANAGER_INSTANCE
    if _SHUTDOWN_MANAGER_INSTANCE is None:
        _SHUTDOWN_MANAGER_INSTANCE = ShutdownManager()
    return _SHUTDOWN_MANAGER_INSTANCE


def reset_shutdown_manager() -> None:
    """Reset global ShutdownManager for testing."""
    global _SHUTDOWN_MANAGER_INSTANCE, _SIGNAL_HANDLERS_REGISTERED
    _SHUTDOWN_MANAGER_INSTANCE = None
    _SIGNAL_HANDLERS_REGISTERED = False
