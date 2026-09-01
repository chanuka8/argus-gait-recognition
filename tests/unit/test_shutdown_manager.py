import signal
import threading
import time
from unittest.mock import MagicMock

from deployment.shutdown_manager import ShutdownManager, reset_shutdown_manager


def test_shutdown_manager_idempotent():
    reset_shutdown_manager()
    sm = ShutdownManager()

    callback_count = [0]

    def cleanup():
        callback_count[0] += 1

    sm.register_cleanup_callback(cleanup)

    assert sm.shutdown() is True
    assert callback_count[0] == 1

    assert sm.shutdown() is True
    assert callback_count[0] == 1


def test_shutdown_manager_signals_stop_events_and_joins_threads():
    reset_shutdown_manager()
    sm = ShutdownManager(join_timeout_sec=1.0)

    stop_evt = threading.Event()
    sm.register_stop_event(stop_evt)

    thread_ran = [False]

    def worker():
        stop_evt.wait(timeout=2.0)
        thread_ran[0] = True

    t = threading.Thread(target=worker, name="TestWorker")
    t.start()
    sm.register_thread(t)

    assert sm.shutdown() is True
    assert stop_evt.is_set() is True
    assert thread_ran[0] is True
    assert t.is_alive() is False


def test_shutdown_manager_handles_join_timeout():
    reset_shutdown_manager()
    sm = ShutdownManager(join_timeout_sec=0.1)

    def stubborn_worker():
        time.sleep(0.5)

    t = threading.Thread(target=stubborn_worker, name="StubbornWorker")
    t.start()
    sm.register_thread(t)

    assert sm.shutdown() is True
    t.join(timeout=1.0)


def test_shutdown_manager_signal_registration(monkeypatch):
    reset_shutdown_manager()
    sm = ShutdownManager()

    mock_signal = MagicMock()
    monkeypatch.setattr("signal.signal", mock_signal)

    sm.register_signal_handlers()

    assert mock_signal.called
    sig_args = [call[0][0] for call in mock_signal.call_args_list]
    assert signal.SIGINT in sig_args
