import argparse
import os
import signal
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from monitoring.logging_config import get_logger, init_logging
from pipeline.steps.live_gei import LiveGEI
from services.camera_service import CameraService


class ArgusService:
    def __init__(self, config_path: str = "configs/system.yaml") -> None:
        init_logging()
        self._logger = get_logger("system")
        self._config = self._load_config(config_path)

        self._camera: CameraService | None = None
        self._recognition_thread: threading.Thread | None = None
        self._watchdog_thread: threading.Thread | None = None

        self._stop_event = threading.Event()
        self._recognition_alive = False
        self._restart_count: int = 0
        self._start_time: float = 0.0

    @staticmethod
    def _load_config(config_path: str) -> dict:
        path = Path(config_path)

        if not path.exists():
            return {}

        try:
            with open(path, "r", encoding="utf-8") as file:
                return yaml.safe_load(file) or {}
        except Exception:
            return {}

    def _build_pipeline(self):
        from pipeline.live_recognition import LiveRecognitionPipeline

        rec_cfg = self._config.get("recognition", {})

        pipeline = LiveRecognitionPipeline(
            model_path=rec_cfg.get("model_path", "runs/exp_001/best_model.pth"),
            threshold=float(rec_cfg.get("threshold", 0.85)),
            alert_threshold=float(rec_cfg.get("alert_threshold", 0.90)),
            security_threshold=float(rec_cfg.get("security_threshold", 0.90)),
            gei_frames=int(rec_cfg.get("gei_frames", 15)),
            recognition_interval=int(rec_cfg.get("recognition_interval", 10)),
            gallery_dir=rec_cfg.get("gallery_dir", "models/live_gallery"),
        )

        return pipeline

    def _recognition_worker(self) -> None:
        detection_logger = get_logger("detection")
        error_logger = get_logger("error")

        self._logger.info("Recognition worker starting...")
        self._recognition_alive = True

        try:
            pipeline = self._build_pipeline()
            self._logger.info("Recognition pipeline built successfully.")

            headless = self._config.get("service", {}).get("headless", False)
            frame_count = 0

            while not self._stop_event.is_set():
                if self._camera is None or not self._camera.is_alive():
                    self._stop_event.wait(0.1)
                    continue

                frame = self._camera.get_frame()

                if frame is None:
                    self._stop_event.wait(0.01)
                    continue

                frame_count += 1

                try:
                    detections = pipeline.tracker.track(frame)

                    xyxy = detections.xyxy
                    tracker_ids = detections.tracker_id

                    raw_detections = []

                    if tracker_ids is not None:
                        confidences = getattr(detections, "confidence", None)

                        for i in range(len(tracker_ids)):
                            raw_detections.append((
                                int(tracker_ids[i]),
                                xyxy[i],
                                float(confidences[i]) if confidences is not None else 1.0,
                            ))

                    stable_results = pipeline.box_stabilizer.update(
                        raw_detections,
                        frame.shape,
                    )

                    pipeline.current_frame_index += 1

                    for track_id, (stable_box, is_valid, is_predicted) in stable_results.items():
                        if not is_valid:
                            continue

                        raw_detected = (tracker_ids is not None) and (track_id in tracker_ids)
                        raw_box = None

                        if raw_detected:
                            raw_box = xyxy[list(tracker_ids).index(track_id)]

                        if raw_detected:
                            box_to_crop = (
                                stable_box
                                if pipeline.box_stability_config.get("use_stable_box_for_silhouette", True)
                                else raw_box
                            )

                            crop = pipeline._crop_person(frame, box_to_crop)

                            if crop is not None:
                                silhouette = pipeline.silhouette_step.extract_from_crop(crop)

                                if silhouette is not None:
                                    if track_id not in pipeline.buffers:
                                        pipeline.buffers[track_id] = LiveGEI(
                                            max_frames=pipeline.gei_frames,
                                        )

                                    pipeline.buffers[track_id].add(silhouette)

                        if (
                            track_id in pipeline.buffers
                            and pipeline.buffers[track_id].ready()
                            and pipeline._should_recognize(track_id)
                        ):
                            pipeline._recognize_track(track_id, is_predicted)

                        if not headless:
                            box_to_draw = (
                                stable_box
                                if pipeline.box_stability_config.get("use_stable_box_for_display", True)
                                else (raw_box if raw_detected else stable_box)
                            )
                            pipeline._draw_track(frame, box_to_draw, track_id, is_predicted)

                    if not headless:
                        cv2.imshow("ARGUS AI Service", frame)

                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            self._logger.info("Quit signal received from GUI.")
                            self._stop_event.set()
                            break

                except Exception as error:
                    error_logger.error(
                        f"Frame processing error (frame #{frame_count}): {error}",
                        exc_info=True,
                    )

        except Exception as error:
            error_logger.error(f"Recognition worker fatal error: {error}", exc_info=True)
        finally:
            self._recognition_alive = False
            self._logger.info("Recognition worker stopped.")

    def _start_watchdog(self) -> None:
        watchdog_cfg = self._config.get("watchdog", {})

        if not watchdog_cfg.get("enabled", True):
            self._logger.info("Watchdog is disabled in config.")
            return

        try:
            from monitoring.watchdog import Watchdog

            watchdog = Watchdog(
                service=self,
                config=watchdog_cfg,
            )

            self._watchdog_thread = threading.Thread(
                target=watchdog.run,
                name="ARGUS-Watchdog",
                daemon=True,
            )
            self._watchdog_thread.start()
            self._logger.info("Watchdog started.")
        except Exception as error:
            self._logger.error(f"Failed to start watchdog: {error}")

    def restart_recognition(self) -> None:
        self._logger.warning("Restarting recognition worker...")
        self._restart_count += 1

        self._recognition_thread = threading.Thread(
            target=self._recognition_worker,
            name="ARGUS-RecognitionWorker",
            daemon=True,
        )
        self._recognition_thread.start()
        self._logger.info("Recognition worker restarted.")

    def restart_camera(self) -> None:
        self._logger.warning("Restarting camera service...")

        if self._camera is not None:
            self._camera.stop()

        cam_cfg = self._config.get("camera", {})
        self._camera = CameraService(config=cam_cfg)
        self._camera.start()
        self._logger.info("Camera service restarted.")

    def _write_pid(self) -> None:
        pid_path = Path(
            self._config.get("service", {}).get("pid_file", "outputs/argus.pid")
        )
        pid_path.parent.mkdir(parents=True, exist_ok=True)
        pid_path.write_text(str(os.getpid()), encoding="utf-8")
        self._logger.info(f"PID {os.getpid()} written to {pid_path}")

    def _remove_pid(self) -> None:
        pid_path = Path(
            self._config.get("service", {}).get("pid_file", "outputs/argus.pid")
        )

        try:
            if pid_path.exists():
                pid_path.unlink()
        except Exception:
            pass

    def _handle_signal(self, signum: int, _frame) -> None:
        sig_name = signal.Signals(signum).name
        self._logger.info(f"Received {sig_name}. Initiating shutdown...")
        self._stop_event.set()

    def start(self) -> None:
        self._start_time = time.monotonic()
        service_name = self._config.get("service", {}).get(
            "name", "ARGUS AI Gait Recognition"
        )

        self._logger.info("=" * 60)
        self._logger.info(f"{service_name} — Starting")
        self._logger.info("=" * 60)

        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

        self._write_pid()

        try:
            cam_cfg = self._config.get("camera", {})
            self._camera = CameraService(config=cam_cfg)
            self._camera.start()

            self._recognition_thread = threading.Thread(
                target=self._recognition_worker,
                name="ARGUS-RecognitionWorker",
                daemon=True,
            )
            self._recognition_thread.start()

            self._start_watchdog()

            self._logger.info("All services started. Running continuously...")

            while not self._stop_event.is_set():
                self._stop_event.wait(1.0)

        except Exception as error:
            self._logger.error(f"Service error: {error}", exc_info=True)
        finally:
            self.stop()

    def stop(self) -> None:
        self._logger.info("Shutting down ARGUS service...")
        self._stop_event.set()

        timeout = self._config.get("service", {}).get("shutdown_timeout", 10)

        if self._camera is not None:
            self._camera.stop()

        if self._recognition_thread is not None and self._recognition_thread.is_alive():
            self._recognition_thread.join(timeout=timeout)

        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

        self._remove_pid()

        uptime = time.monotonic() - self._start_time if self._start_time else 0
        self._logger.info(
            f"ARGUS service stopped. Uptime: {uptime:.1f}s, Restarts: {self._restart_count}"
        )

    def get_status(self) -> dict:
        uptime = time.monotonic() - self._start_time if self._start_time else 0

        return {
            "running": not self._stop_event.is_set(),
            "uptime_seconds": round(uptime, 1),
            "restart_count": self._restart_count,
            "camera": self._camera.get_status() if self._camera else {},
            "recognition_alive": self._recognition_alive,
            "pid": os.getpid(),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="ARGUS AI Gait Recognition Service")

    parser.add_argument(
        "--config",
        type=str,
        default="configs/system.yaml",
        help="Path to system configuration YAML.",
    )

    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without GUI windows.",
    )

    args = parser.parse_args()

    service = ArgusService(config_path=args.config)

    if args.headless:
        service._config.setdefault("service", {})["headless"] = True

    service.start()


if __name__ == "__main__":
    main()
