import numpy as np
import torch

from app.core.paths import resolve_app_path
from ml_platform.models.architectures.bygait_light import ByGaitLight
from ml_platform.models.inference.backend import BaseInferenceBackend


class PyTorchBackend(BaseInferenceBackend):
    def __init__(
        self,
        config: dict | None = None,
        model_path: str | None = None,
    ) -> None:
        super().__init__(config=config)
        self.backend_name = "pytorch"
        self.model_path = resolve_app_path(model_path or self.config.get("model_path") or "runs/exp_001/best_model.pth")

        self.device = self._resolve_device(self.device_str)
        self.execution_provider = f"PyTorch-{self.device.type.upper()}"
        self.model = self._load_model()
        self.warmup()

    def _resolve_device(self, device_str: str) -> torch.device:
        from ops.automation.device_manager import DeviceManager

        resolved = DeviceManager.get_instance().resolve_component_device(device_str)
        return torch.device(resolved)

    def _load_model(self) -> ByGaitLight:
        part_bins = 4
        filtered = {}
        if self.model_path.exists():
            import io

            from app.security_layer.model_confidentiality import (
                ArtifactConfidentiality,
                load_verified_model_bytes,
            )
            from app.security_layer.model_integrity import ROLE_GAIT_EMBEDDING

            try:
                logical_name = "best_model.pth" if self.model_path.name.endswith(".enc") else self.model_path.name
                model_bytes = load_verified_model_bytes(
                    self.model_path,
                    expected_role=ROLE_GAIT_EMBEDDING,
                    logical_filename=logical_name,
                    confidentiality=ArtifactConfidentiality.PROTECTED,
                )
                buffer = io.BytesIO(model_bytes)
                checkpoint = torch.load(buffer, map_location="cpu", weights_only=True)
                for key, value in checkpoint.items():
                    if key.startswith("backbone."):
                        filtered[key.replace("backbone.", "")] = value
                    else:
                        filtered[key] = value

                if "embedding.weight" in filtered:
                    in_features = filtered["embedding.weight"].shape[1]
                    part_bins = max(1, in_features // 128)
            except (RuntimeError, ValueError, OSError, EOFError) as e:
                self.logger.warning(f"Could not load checkpoint from {self.model_path}: {e}")

        model = ByGaitLight(part_bins=part_bins)
        if filtered:
            valid_keys = {k: v for k, v in filtered.items() if k in model.state_dict()}
            model.load_state_dict(valid_keys, strict=False)

        model.to(self.device)
        model.eval()

        if self.precision == "fp16" and self.device.type == "cuda":
            model.half()

        return model

    def predict(self, x: np.ndarray | torch.Tensor) -> np.ndarray:
        if isinstance(x, np.ndarray):
            tensor = torch.from_numpy(x).float()
        else:
            tensor = x.float()

        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0).unsqueeze(0)
        elif tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)

        tensor = tensor.to(self.device, non_blocking=True)
        if self.precision == "fp16" and self.device.type == "cuda":
            tensor = tensor.half()

        with torch.inference_mode():
            output = self.model(tensor)
            if self.precision == "fp16":
                output = output.float()
            embeddings = output.cpu().numpy()

        return embeddings.astype(np.float32)
