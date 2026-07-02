from pathlib import Path
import json


class DataManager:
    def __init__(self, root: str = "storage_data"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def save_json(self, name: str, data: dict) -> Path:
        path = self.root / f"{name}.json"

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

        return path

    def load_json(self, name: str) -> dict | None:
        path = self.root / f"{name}.json"

        if not path.exists():
            return None

        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def exists(self, name: str) -> bool:
        return (self.root / f"{name}.json").exists()