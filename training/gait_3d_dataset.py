"""
PyTorch Dataset for 3D Pose Gait Model Training & Evaluation.
Loads per-sequence 2D/3D skeleton keypoint arrays from data/casia_processed/skeletons/.
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class Gait3DSkeletonDataset(Dataset):
    """
    Dataset for loading 3D Pose Gait sequence keypoint arrays (.npy).
    """

    def __init__(
        self,
        subjects: list[str],
        data_dir: str = "data/casia_processed/skeletons",
        sequence_length: int = 30,
        split_config_path: str = "configs/subject_split.json",
    ) -> None:
        self.data_dir = Path(data_dir)
        self.subjects = set(subjects)
        self.sequence_length = sequence_length

        self.samples: list[tuple[Path, str, str, str]] = []
        self.label_to_index: dict[str, int] = {}
        self.index_to_label: dict[int, str] = {}

        self._build_index()

    def _build_index(self) -> None:
        if not self.data_dir.exists():
            return

        unique_subs = sorted(self.subjects)
        for idx, sub in enumerate(unique_subs):
            self.label_to_index[sub] = idx
            self.index_to_label[idx] = sub

        for sub_dir in sorted(self.data_dir.glob("*")):
            if not sub_dir.is_dir() or sub_dir.name not in self.subjects:
                continue

            for npy_path in sorted(sub_dir.glob("*.npy")):
                stem = npy_path.stem
                parts = stem.split("_")
                if len(parts) >= 3:
                    sub_id = parts[0]
                    cond_raw = parts[1].upper()
                    view = parts[2]

                    if "NM" in cond_raw:
                        cond = "NM"
                    elif "BG" in cond_raw:
                        cond = "BG"
                    elif "CL" in cond_raw:
                        cond = "CL"
                    else:
                        cond = "NM"

                    self.samples.append((npy_path, sub_id, cond, view))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, str]:
        npy_path, sub_id, cond, _view = self.samples[idx]
        seq = np.load(npy_path)

        T_curr = len(seq)
        if T_curr == 0:
            seq_fixed = np.zeros((self.sequence_length, 17, 3), dtype=np.float32)
        elif T_curr < self.sequence_length:
            repeats = (self.sequence_length // T_curr) + 1
            seq_fixed = np.tile(seq, (repeats, 1, 1))[: self.sequence_length]
        elif T_curr > self.sequence_length:
            start_idx = np.random.randint(0, T_curr - self.sequence_length + 1)
            seq_fixed = seq[start_idx : start_idx + self.sequence_length]
        else:
            seq_fixed = seq

        tensor = torch.from_numpy(seq_fixed).float()
        label_idx = self.label_to_index[sub_id]

        return tensor, label_idx, cond
