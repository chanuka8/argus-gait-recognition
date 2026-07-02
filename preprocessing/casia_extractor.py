from pathlib import Path

from core.logger import setup_logger
from utils.zip_streamer import ZipStreamer


class CasiaExtractor:
    def __init__(
        self,
        zip_path: str = "data/casia_b_raw.zip",
        output_dir: str = "data/casia_cache",
    ):
        self.logger = setup_logger("ARGUS.CASIA")

        self.zip_path = zip_path
        self.output_dir = output_dir

    def extract(self) -> None:

        streamer = ZipStreamer(self.zip_path)

        if not streamer.exists():
            raise FileNotFoundError(
                f"Dataset not found: {self.zip_path}"
            )

        self.logger.info(
            f"ZIP contains {streamer.file_count()} files"
        )

        output_path = Path(self.output_dir)

        if any(output_path.iterdir()):
            self.logger.info(
                "CASIA cache already populated. Skipping extraction."
            )
            return

        self.logger.info(
            "Extracting CASIA dataset..."
        )

        streamer.extract_all(self.output_dir)

        self.logger.info(
            "CASIA extraction completed."
        )