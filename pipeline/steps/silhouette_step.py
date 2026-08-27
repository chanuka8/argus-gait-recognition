from pathlib import Path

import cv2
import numpy as np

try:
    import yaml
except ImportError:
    yaml = None


class LearnedSilhouetteSegmenter:
    """
    Learned human silhouette segmentation strategy using local ONNX model.
    Falls back gracefully if ONNX runtime or model file is unavailable.
    """

    def __init__(self, model_path: str = "models/weights/silhouette_segmenter.onnx", threshold: float = 0.5) -> None:
        self.model_path = Path(model_path)
        self.threshold = threshold
        self.session = None
        self._init_session()

    def _init_session(self) -> None:
        target_path = None
        if self.model_path.exists():
            target_path = self.model_path
        else:
            defaults = [
                Path("models/weights/silhouette_segmenter.onnx"),
                Path("models/engines/silhouette_segmenter.onnx"),
            ]
            if str(self.model_path) in {
                "models/weights/silhouette_segmenter.onnx",
                "models/engines/silhouette_segmenter.onnx",
            }:
                for p in defaults:
                    if p.exists():
                        target_path = p
                        break

        if target_path is None or not target_path.exists():
            return
        try:
            from automation.device_manager import DeviceManager
            from automation.dll_manager import setup_cuda_dll_paths

            setup_cuda_dll_paths()

            import onnxruntime as ort

            dm = DeviceManager.get_instance()
            providers = ort.get_available_providers()
            provider_list = []
            if dm.is_cuda and "CUDAExecutionProvider" in providers:
                provider_list.append("CUDAExecutionProvider")
            provider_list.append("CPUExecutionProvider")
            self.session = ort.InferenceSession(str(target_path), providers=provider_list)
            self.model_path = target_path
        except (OSError, ValueError, RuntimeError, TypeError, AttributeError):
            self.session = None

    def is_available(self) -> bool:
        return self.session is not None

    def validate_model(self) -> tuple[bool, str]:
        """
        Validate model file presence, ONNX session initialization, and dry-run inference.
        """
        if not self.model_path.exists():
            return False, f"Model file not found at {self.model_path}"
        if not self.is_available():
            return False, "ONNX InferenceSession failed to initialize"

        try:
            dummy_crop = np.zeros((256, 256, 3), dtype=np.uint8)
            mask = self.segment(dummy_crop)
            if mask is None:
                return False, "Dry-run segmentation returned None"
            if mask.shape != (256, 256):
                return False, f"Expected (256, 256) mask, got {mask.shape}"
            if not np.all(np.isfinite(mask)):
                return False, "Non-finite values detected in dry-run mask"
            return True, "Learned ONNX silhouette segmenter is valid"
        except (OSError, ValueError, RuntimeError, TypeError, AttributeError) as e:
            return False, f"Dry-run inference failed: {e}"

    def segment(self, crop: np.ndarray) -> np.ndarray | None:
        if not self.is_available() or crop is None or crop.size == 0:
            return None

        h, w = crop.shape[:2]
        try:
            resized = cv2.resize(crop, (256, 256))
            blob = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            tensor = np.transpose(blob, (2, 0, 1))[np.newaxis, ...]

            input_name = self.session.get_inputs()[0].name
            output_name = self.session.get_outputs()[0].name
            raw_out = self.session.run([output_name], {input_name: tensor})[0]

            prob_map = np.squeeze(raw_out)
            if not np.all(np.isfinite(prob_map)):
                return None

            if prob_map.ndim > 2:
                prob_map = prob_map[0]

            mask_256 = (prob_map > self.threshold).astype(np.uint8) * 255
            mask = cv2.resize(mask_256, (w, h), interpolation=cv2.INTER_NEAREST)
            return mask
        except (OSError, ValueError, RuntimeError, TypeError, AttributeError):
            return None


class OtsuSilhouetteExtractor:
    """Otsu thresholding silhouette extraction strategy (fallback)."""

    def extract_mask(self, crop: np.ndarray) -> np.ndarray | None:
        if crop is None or crop.size == 0:
            return None

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return mask


class SilhouetteStep:
    """
    Unified Silhouette Step.
    Tries learned segmentation strategy first; falls back to Otsu strategy if unavailable.
    Applies shared morphological cleaning, primary person contour filtering, height normalization,
    and 64x128 canvas alignment.
    """

    def __init__(
        self,
        target_size: tuple[int, int] = (64, 128),
        method: str = "auto",
        model_path: str = "models/weights/silhouette_segmenter.onnx",
        threshold: float = 0.5,
        config_path: str = "configs/inference.yaml",
    ) -> None:
        self.target_size = target_size
        self.config = self._load_config(Path(config_path))

        sil_cfg = self.config.get("silhouette", {})
        self.method = method if method != "auto" else sil_cfg.get("method", "learned")
        default_path = sil_cfg.get("model_path", "models/weights/silhouette_segmenter.onnx")
        self.model_path = model_path if model_path != "models/weights/silhouette_segmenter.onnx" else default_path
        self.threshold = threshold if threshold != 0.5 else float(sil_cfg.get("threshold", 0.5))

        self.learned_segmenter = LearnedSilhouetteSegmenter(model_path=self.model_path, threshold=self.threshold)
        self.otsu_extractor = OtsuSilhouetteExtractor()

    @staticmethod
    def _load_config(config_path: Path) -> dict:
        if not config_path.exists() or yaml is None:
            return {}
        try:
            with open(config_path, encoding="utf-8") as f:
                data = yaml.safe_load(f)
                return data if isinstance(data, dict) else {}
        except (OSError, ValueError, TypeError, AttributeError):
            return {}

    def extract_from_crop(self, crop: np.ndarray) -> np.ndarray | None:
        if crop is None or crop.size == 0:
            return None

        raw_mask = None
        if self.method in {"learned", "auto"} and self.learned_segmenter.is_available():
            raw_mask = self.learned_segmenter.segment(crop)

        if raw_mask is None:
            raw_mask = self.otsu_extractor.extract_mask(crop)

        if raw_mask is None or raw_mask.size == 0:
            return None

        return self._align_and_normalize(raw_mask)

    def _align_and_normalize(self, mask: np.ndarray) -> np.ndarray | None:
        cleaned_mask = self._clean_mask(mask)

        contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        largest_contour = max(contours, key=cv2.contourArea)
        crop_area = cleaned_mask.shape[0] * cleaned_mask.shape[1]
        contour_area = cv2.contourArea(largest_contour)

        if contour_area < 50 or contour_area > 0.95 * crop_area:
            return None

        x, y, w, h = cv2.boundingRect(largest_contour)
        if w < 5 or h < 15:
            return None

        aspect_ratio = h / w
        if aspect_ratio < 1.2 or aspect_ratio > 6.0:
            return None

        contour_mask = np.zeros_like(cleaned_mask)
        cv2.drawContours(contour_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
        cropped_silhouette = contour_mask[y : y + h, x : x + w]

        target_h = int(self.target_size[1] * 0.85)
        scale_factor = target_h / h
        new_w = int(w * scale_factor)
        new_w = max(1, min(new_w, self.target_size[0]))

        resized_silhouette = cv2.resize(
            cropped_silhouette,
            (new_w, target_h),
            interpolation=cv2.INTER_NEAREST,
        )

        canvas = np.zeros((self.target_size[1], self.target_size[0]), dtype=np.uint8)
        x_offset = (self.target_size[0] - new_w) // 2
        y_offset = (self.target_size[1] - target_h) // 2

        canvas[y_offset : y_offset + target_h, x_offset : x_offset + new_w] = resized_silhouette
        return canvas

    @staticmethod
    def _clean_mask(mask: np.ndarray) -> np.ndarray:
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        return mask
