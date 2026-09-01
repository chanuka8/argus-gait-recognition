from pathlib import Path

import numpy as np
import torch

from models.architectures.bygait_light import ByGaitLight
from models.inference.backend import BaseInferenceBackend


class PyTorchBackend(BaseInferenceBackend):
    def __init__(
        self,
        config: dict | None = None,
        model_path: str | None = None,
    ) -> None:
        super().__init__(config=config)
        self.backend_name = "pytorch"
        self.model_path = Path(model_path or self.config.get("model_path") or "runs/exp_001/best_model.pth")
        self.device = self._resolve_device(self.device_str)
        self.execution_provider = f"PyTorch-{self.device.type.upper()}"
        self.model = self._load_model()
        self.warmup()

    def _resolve_device(self, device_str: str) -> torch.device:
        from automation.device_manager import DeviceManager

        resolved = DeviceManager.get_instance().resolve_component_device(device_str)
        return torch.device(resolved)

    def _load_model(self) -> ByGaitLight:
        part_bins = 4
        filtered = {}
        if self.model_path.exists():
            try:
                checkpoint = torch.load(self.model_path, map_location="cpu", weights_only=True)
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

        tensor = tensor.to(self.device)
        if self.precision == "fp16" and self.device.type == "cuda":
            tensor = tensor.half()

        with torch.no_grad():
            output = self.model(tensor)
            if self.precision == "fp16":
                output = output.float()

            embeddings = output.cpu().numpy()

        return self.normalize_embedding(embeddings)
