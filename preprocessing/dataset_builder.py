from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile

import cv2
import numpy as np
from tqdm import tqdm

from core.logger import setup_logger
from preprocessing.gei_builder import GEIBuilder


class CasiaGEIDatasetBuilder:
    def __init__(
        self,
        zip_path: str = "data/casia_b_raw.zip",
        output_dir: str = "data/casia_processed/gei",
        min_frames: int = 15,
        max_sequences: int | None = None,
    ) -> None:
        self.zip_path = Path(zip_path)
        self.output_dir = Path(output_dir)
        self.min_frames = min_frames
        self.max_sequences = max_sequences
        self.logger = setup_logger("ARGUS.CASIA.GEIBuilder")

    def _parse_key(self, file_name: str) -> tuple[str, str, str] | None:
        parts = Path(file_name).parts

        if len(parts) < 5:
            return None

        if parts[0] != "output":
            return None

        person_id = parts[1]
        condition = parts[2]
        angle = parts[3]

        return person_id, condition, angle

    def _is_image(self, file_name: str) -> bool:
        return file_name.lower().endswith((".png", ".jpg", ".jpeg"))

    def _read_image_from_zip(
        self,
        archive: ZipFile,
        file_name: str,
    ) -> np.ndarray | None:
        try:
            raw_bytes = archive.read(file_name)
            array = np.frombuffer(raw_bytes, dtype=np.uint8)
            image = cv2.imdecode(array, cv2.IMREAD_GRAYSCALE)
            return image
        except Exception as error:
            self.logger.warning(f"Failed to read {file_name}: {error}")
            return None

    def scan_sequences(self) -> dict[tuple[str, str, str], list[str]]:
        if not self.zip_path.exists():
            raise FileNotFoundError(f"ZIP file not found: {self.zip_path}")

        sequence_map: dict[tuple[str, str, str], list[str]] = defaultdict(list)

        self.logger.info("Scanning ZIP structure")

        with ZipFile(self.zip_path, "r") as archive:
            for file_name in archive.namelist():
                if not self._is_image(file_name):
                    continue

                key = self._parse_key(file_name)

                if key is None:
                    continue

                sequence_map[key].append(file_name)

        for key in sequence_map:
            sequence_map[key].sort()

        self.logger.info(f"Detected sequences: {len(sequence_map)}")

        return dict(sequence_map)

    def build(self) -> dict:
        self.output_dir.mkdir(parents=True, exist_ok=True)

        sequences = self.scan_sequences()
        items = list(sequences.items())

        if self.max_sequences is not None:
            items = items[: self.max_sequences]

        built_count = 0
        skipped_count = 0

        with ZipFile(self.zip_path, "r") as archive:
            for (person_id, condition, angle), frame_files in tqdm(
                items,
                desc="Building GEI",
            ):
                if len(frame_files) < self.min_frames:
                    skipped_count += 1
                    continue

                builder = GEIBuilder()

                for file_name in frame_files:
                    image = self._read_image_from_zip(archive, file_name)

                    if image is None:
                        continue

                    builder.add_frame(image)

                gei = builder.build()

                if gei is None:
                    skipped_count += 1
                    continue

                person_dir = self.output_dir / person_id
                person_dir.mkdir(parents=True, exist_ok=True)

                output_name = f"{person_id}_{condition}_{angle}.png"
                output_path = person_dir / output_name

                cv2.imwrite(str(output_path), gei)

                built_count += 1

        summary = {
            "zip_path": str(self.zip_path),
            "output_dir": str(self.output_dir),
            "detected_sequences": len(sequences),
            "built_gei": built_count,
            "skipped": skipped_count,
        }

        self.logger.info(f"GEI build summary: {summary}")

        return summary