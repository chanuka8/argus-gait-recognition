from datetime import datetime
from pathlib import Path
import json


class LineageTracker:
    def __init__(
        self,
        output_file="outputs/lineage/lineage.json"
    ):
        self.output_file = Path(output_file)

        self.output_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if not self.output_file.exists():
            self._save([])

    def _load(self):
        with open(
            self.output_file,
            "r",
            encoding="utf-8",
        ) as f:
            return json.load(f)

    def _save(self, data):
        with open(
            self.output_file,
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                data,
                f,
                indent=4,
            )

    def add_record(
        self,
        operation,
        details,
    ):
        records = self._load()

        records.append(
            {
                "timestamp": datetime.now().isoformat(),
                "operation": operation,
                "details": details,
            }
        )

        self._save(records)

    def history(self):
        return self._load()