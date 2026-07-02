from pathlib import Path
from datetime import datetime
import csv
import threading


class SecurityLogger:

    def __init__(
        self,
        log_file="outputs/security_logs/security_events.csv",
    ):

        self.log_file = Path(log_file)

        self.log_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._lock = threading.Lock()

        if not self.log_file.exists():

            with open(
                self.log_file,
                "w",
                newline="",
                encoding="utf-8",
            ) as f:

                writer = csv.writer(f)

                writer.writerow(
                    [
                        "timestamp",
                        "track_id",
                        "identity",
                        "score",
                        "severity",
                        "decision",
                        "camera_id",
                    ]
                )

    def log(
        self,
        track_id,
        identity,
        score,
        severity,
        decision,
        camera_id="default",
    ):

        with self._lock:
            with open(
                self.log_file,
                "a",
                newline="",
                encoding="utf-8",
            ) as f:

                writer = csv.writer(f)

                writer.writerow(
                    [
                        datetime.now().isoformat(),
                        track_id,
                        identity,
                        round(score, 4),
                        severity,
                        decision,
                        camera_id,
                    ]
                )