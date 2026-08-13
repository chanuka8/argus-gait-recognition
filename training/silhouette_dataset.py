from pathlib import Path
import zipfile
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class SilhouetteSegmentationDataset(Dataset):
    """
    Fast In-Memory Dataset for UNet Person Silhouette Segmentation.
    Pre-loads silhouette frames from CASIA-B raw ZIP archive into memory and synthesizes realistic RGB crop pairs.
    """

    def __init__(
        self,
        zip_path: str = "data/casia_b_raw.zip",
        subject_range: tuple[int, int] = (1, 62),
        img_size: tuple[int, int] = (256, 256),
        max_samples: int = 300,
        seed: int = 42,
    ) -> None:
        self.zip_path = Path(zip_path)
        self.img_size = img_size
        self.rng = np.random.default_rng(seed)
        self.masks: list[np.ndarray] = []

        if not self.zip_path.exists():
            raise FileNotFoundError(f"CASIA-B ZIP archive not found at: {self.zip_path}")

        min_sub, max_sub = subject_range
        matching_files: list[str] = []

        with zipfile.ZipFile(self.zip_path, "r") as archive:
            for name in archive.namelist():
                if not name.endswith(".png"):
                    continue
                parts = Path(name).parts
                if len(parts) >= 2 and parts[0] == "output":
                    try:
                        sub_id = int(parts[1])
                        if min_sub <= sub_id <= max_sub:
                            matching_files.append(name)
                    except ValueError:
                        continue

            if max_samples and len(matching_files) > max_samples:
                self.rng.shuffle(matching_files)
                matching_files = matching_files[:max_samples]

            for name in matching_files:
                raw_bytes = archive.read(name)
                mask = cv2.imdecode(np.frombuffer(raw_bytes, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
                if mask is not None and mask.size > 0 and np.max(mask) > 0:
                    self.masks.append(mask)

    def __len__(self) -> int:
        return len(self.masks)

    def _render_synthetic_rgb_crop(self, mask: np.ndarray, sample_idx: int) -> np.ndarray:
        h, w = mask.shape
        fg_mask = (mask > 128).astype(np.uint8)
        rng = np.random.default_rng(sample_idx * 1337 + 7)

        bg_color1 = rng.integers(30, 200, size=3, dtype=np.uint8)
        bg_color2 = rng.integers(30, 200, size=3, dtype=np.uint8)
        grad = np.linspace(0, 1, h)[:, None, None]
        bg_rgb = (bg_color1[None, None, :] * (1 - grad) + bg_color2[None, None, :] * grad).astype(np.uint8)

        bg_noise = rng.integers(-20, 20, size=(h, w, 3), dtype=np.int16)
        bg_rgb = np.clip(bg_rgb.astype(np.int16) + bg_noise, 0, 255).astype(np.uint8)

        fg_top_color = rng.integers(10, 240, size=3, dtype=np.uint8)
        fg_bot_color = rng.integers(10, 240, size=3, dtype=np.uint8)
        fg_grad = np.linspace(0, 1, h)[:, None, None]
        fg_rgb = (fg_top_color[None, None, :] * (1 - fg_grad) + fg_bot_color[None, None, :] * fg_grad).astype(np.uint8)

        fg_noise = rng.integers(-15, 15, size=(h, w, 3), dtype=np.int16)
        fg_rgb = np.clip(fg_rgb.astype(np.int16) + fg_noise, 0, 255).astype(np.uint8)

        crop_rgb = bg_rgb.copy()
        crop_rgb[fg_mask == 1] = fg_rgb[fg_mask == 1]

        if rng.random() > 0.5:
            crop_rgb = cv2.GaussianBlur(crop_rgb, (3, 3), 0)

        return crop_rgb

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        mask = self.masks[idx]

        rgb_crop = self._render_synthetic_rgb_crop(mask, idx)
        rgb_resized = cv2.resize(rgb_crop, self.img_size, interpolation=cv2.INTER_LINEAR)
        mask_resized = cv2.resize(mask, self.img_size, interpolation=cv2.INTER_NEAREST)

        rgb_norm = rgb_resized.astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(np.transpose(rgb_norm, (2, 0, 1)))

        mask_binary = (mask_resized > 128).astype(np.float32)
        mask_tensor = torch.from_numpy(mask_binary[np.newaxis, ...])

        return img_tensor, mask_tensor
