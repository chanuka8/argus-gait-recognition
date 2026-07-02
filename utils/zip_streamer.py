from pathlib import Path
from zipfile import ZipFile


class ZipStreamer:
    def __init__(self, zip_path: str):
        self.zip_path = Path(zip_path)

    def exists(self) -> bool:
        return self.zip_path.exists()

    def list_files(self) -> list[str]:
        with ZipFile(self.zip_path, "r") as archive:
            return archive.namelist()

    def extract_all(self, destination: str) -> None:
        destination_path = Path(destination)
        destination_path.mkdir(parents=True, exist_ok=True)

        with ZipFile(self.zip_path, "r") as archive:
            archive.extractall(destination_path)

    def file_count(self) -> int:
        return len(self.list_files())