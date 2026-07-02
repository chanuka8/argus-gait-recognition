import json
from pathlib import Path

import torch


class Checkpointer:
    def __init__(self, run_dir: str = "runs/exp_001") -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def save_model(self, model, filename: str) -> None:
        path = self.run_dir / filename
        torch.save(model.state_dict(), path)

    def save_metrics(self, metrics: dict) -> None:
        path = self.run_dir / "metrics.json"

        with open(path, "w", encoding="utf-8") as file:
            json.dump(metrics, file, indent=4)