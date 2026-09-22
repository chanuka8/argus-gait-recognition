import csv
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml


class AlertManager:
    def __init__(
        self,
        alert_file="outputs/logs/events/alerts.csv",
        confidence_threshold=0.75,
    ):

        self.confidence_threshold = confidence_threshold

        self.alert_file = Path(alert_file)

        self.alert_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._lock = threading.Lock()
        self.alert_cooldown_seconds = 5.0
        try:
            config_path = Path("configs/inference.yaml")
            if config_path.exists():
                with open(config_path, encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
                self.alert_cooldown_seconds = float(data.get("crowd_control", {}).get("alert_cooldown_seconds", 5.0))
        except (yaml.YAMLError, OSError, ValueError, KeyError):
            pass
        self.last_alert_times = {}

        if not self.alert_file.exists():
            with open(
                self.alert_file,
                "w",
                newline="",
                encoding="utf-8",
            ) as f:
                writer = csv.writer(f)

                writer.writerow(
                    [
                        "timestamp",
                        "track_id",
                        "alert_type",
                        "identity",
                        "score",
                        "camera_id",
                    ]
                )

    def create_alert(
        self,
        track_id,
        identity,
        score,
        alert_type,
        camera_id="default",
    ):

        with self._lock:
            key = (camera_id, identity, alert_type)
            now = time.time()
            if key in self.last_alert_times and now - self.last_alert_times[key] < self.alert_cooldown_seconds:
                return
            self.last_alert_times[key] = now

            with open(
                self.alert_file,
                "a",
                newline="",
                encoding="utf-8",
            ) as f:
                writer = csv.writer(f)

                writer.writerow(
                    [
                        datetime.now(timezone.utc).isoformat(),
                        track_id,
                        alert_type,
                        identity,
                        round(score, 4),
                        camera_id,
                    ]
                )

        print(f"[ALERT] {alert_type} | Camera={camera_id} | Track={track_id} | Identity={identity} | Score={score:.2f}")

    def evaluate(
        self,
        track_id,
        identity,
        score,
        source=None,
        decision=None,
        camera_id="default",
    ):

        if identity == "UNKNOWN":
            self.create_alert(
                track_id,
                identity,
                score,
                "UNKNOWN_PERSON",
                camera_id=camera_id,
            )

            return

        if score < self.confidence_threshold:
            self.create_alert(
                track_id,
                identity,
                score,
                "LOW_CONFIDENCE",
                camera_id=camera_id,
            )

            return

        self.create_alert(
            track_id,
            identity,
            score,
            "CONFIRMED_MATCH",
            camera_id=camera_id,
        )
