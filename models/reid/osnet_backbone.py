
"""
OSNet-x0.25 Deep Person Re-Identification Backbone.

Lightweight omni-scale network for person re-identification.
Thread-safe singleton with lazy model loading.
GPU if available, CPU fallback.

Architecture matches torchreid/deep-person-reid for
pretrained weight compatibility.
"""

import threading
from pathlib import Path
from typing import Self

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

# --- OSNet Architecture Components ---


class _ConvLayer(nn.Module):
    """Convolution layer (conv + bn + relu)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        groups: int = 1,
    ) -> None:
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
            groups=groups,
        )

        self.bn = nn.BatchNorm2d(
            out_channels,
        )

        self.relu = nn.ReLU(
            inplace=True,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.relu(
            self.bn(
                self.conv(x),
            ),
        )


class _Conv1x1(nn.Module):
    """1x1 convolution + bn + relu."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        groups: int = 1,
    ) -> None:
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            1,
            stride=stride,
            padding=0,
            bias=False,
            groups=groups,
        )

        self.bn = nn.BatchNorm2d(
            out_channels,
        )

        self.relu = nn.ReLU(
            inplace=True,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.relu(
            self.bn(
                self.conv(x),
            ),
        )


class _Conv1x1Linear(nn.Module):
    """1x1 convolution + bn (no relu, for residual)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
    ) -> None:
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            1,
            stride=stride,
            padding=0,
            bias=False,
        )

        self.bn = nn.BatchNorm2d(
            out_channels,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.bn(
            self.conv(x),
        )


class _LightConv3x3(nn.Module):
    """Lightweight 3x3 depthwise separable convolution (Pointwise 1x1 -> Depthwise 3x3 -> BN -> ReLU)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            1,
            stride=1,
            padding=0,
            bias=False,
        )

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            3,
            stride=1,
            padding=1,
            bias=False,
            groups=out_channels,
        )

        self.bn = nn.BatchNorm2d(
            out_channels,
        )

        self.relu = nn.ReLU(
            inplace=True,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.relu(
            self.bn(
                self.conv2(
                    self.conv1(x),
                ),
            ),
        )


class _ChannelGate(nn.Module):
    """Channel attention gate (squeeze-excitation)."""

    def __init__(
        self,
        in_channels: int,
        num_gates: int | None = None,
        return_gates: bool = False,
        reduction: int = 16,
    ) -> None:
        super().__init__()

        if num_gates is None:
            num_gates = in_channels

        self.return_gates = return_gates
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)

        hidden = max(
            in_channels // reduction,
            1,
        )

        self.fc1 = nn.Conv2d(
            in_channels,
            hidden,
            1,
            bias=True,
        )
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(
            hidden,
            num_gates,
            1,
            bias=True,
        )
        self.gate_activation = nn.Sigmoid()

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        g = self.global_avgpool(x)
        g = self.fc2(self.relu(self.fc1(g)))
        gate = self.gate_activation(g)

        if self.return_gates:
            return gate

        return x * gate


class _OSBlock(nn.Module):
    """Omni-scale feature learning block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bottleneck_reduction: int = 4,
    ) -> None:
        super().__init__()

        mid_channels = out_channels // bottleneck_reduction

        self.conv1 = _Conv1x1(
            in_channels,
            mid_channels,
        )

        self.conv2a = _LightConv3x3(
            mid_channels,
            mid_channels,
        )

        self.conv2b = nn.Sequential(
            _LightConv3x3(mid_channels, mid_channels),
            _LightConv3x3(mid_channels, mid_channels),
        )

        self.conv2c = nn.Sequential(
            _LightConv3x3(mid_channels, mid_channels),
            _LightConv3x3(mid_channels, mid_channels),
            _LightConv3x3(mid_channels, mid_channels),
        )

        self.conv2d = nn.Sequential(
            _LightConv3x3(mid_channels, mid_channels),
            _LightConv3x3(mid_channels, mid_channels),
            _LightConv3x3(mid_channels, mid_channels),
            _LightConv3x3(mid_channels, mid_channels),
        )

        self.gate = _ChannelGate(
            mid_channels,
        )

        self.conv3 = _Conv1x1Linear(
            mid_channels,
            out_channels,
        )

        self.downsample = None

        if in_channels != out_channels:
            self.downsample = _Conv1x1Linear(
                in_channels,
                out_channels,
            )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        identity = x

        x1 = self.conv1(x)
        x2a = self.conv2a(x1)
        x2b = self.conv2b(x1)
        x2c = self.conv2c(x1)
        x2d = self.conv2d(x1)

        x2 = self.gate(x2a) + self.gate(x2b) + self.gate(x2c) + self.gate(x2d)

        x3 = self.conv3(x2)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out = x3 + identity

        return F.relu(out)


class _OSNet(nn.Module):
    """Omni-Scale Network architecture."""

    def __init__(
        self,
        blocks: list[type],
        layers: list[int],
        channels: list[int],
        feature_dim: int = 512,
    ) -> None:
        super().__init__()

        self.feature_dim = feature_dim

        # Stem
        self.conv1 = _ConvLayer(
            3,
            channels[0],
            7,
            stride=2,
            padding=3,
        )

        self.maxpool = nn.MaxPool2d(
            3,
            stride=2,
            padding=1,
        )

        # Body
        self.conv2 = self._make_layer(
            blocks[0],
            layers[0],
            channels[0],
            channels[1],
            reduce_spatial_size=True,
        )

        self.conv3 = self._make_layer(
            blocks[1],
            layers[1],
            channels[1],
            channels[2],
            reduce_spatial_size=True,
        )

        self.conv4 = self._make_layer(
            blocks[2],
            layers[2],
            channels[2],
            channels[3],
            reduce_spatial_size=False,
        )

        # Head
        self.conv5 = _Conv1x1(
            channels[3],
            channels[3],
        )

        self.global_avgpool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(channels[3], feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
        )

    def _make_layer(
        self,
        block: type,
        num_blocks: int,
        in_channels: int,
        out_channels: int,
        reduce_spatial_size: bool = True,
    ) -> nn.Sequential:
        layers = [
            block(in_channels, out_channels),
        ]

        for _ in range(1, num_blocks):
            layers.append(
                block(out_channels, out_channels),
            )

        if reduce_spatial_size:
            layers.append(
                nn.Sequential(
                    _Conv1x1(out_channels, out_channels),
                    nn.AvgPool2d(2, stride=2),
                )
            )

        return nn.Sequential(*layers)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        v = self.global_avgpool(x)
        v = v.view(v.size(0), -1)

        if self.fc is not None:
            v = self.fc(v)

        return v


def _build_osnet_x0_25() -> _OSNet:
    """Build OSNet-x0.25 (~530K params)."""

    return _OSNet(
        blocks=[_OSBlock, _OSBlock, _OSBlock],
        layers=[2, 2, 2],
        channels=[16, 64, 96, 128],
        feature_dim=512,
    )


# --- ImageNet normalization ---

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


# --- Public API ---


class OSNetBackbone:
    """
    Thread-safe singleton OSNet-x0.25 backbone
    for person re-identification.

    Lazy-loads model on first extraction call.
    GPU if available, CPU fallback.
    Single model instance across all consumers.
    """

    _instance: "OSNetBackbone | None" = None
    _lock = threading.Lock()

    def __new__(
        cls,
        *args,
        **kwargs,
    ) -> Self:
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(
        self,
        model_path: str = "models/weights/osnet_x0_25.pth",
        device: str = "auto",
    ) -> None:
        if self._initialized:
            return

        self._initialized = True
        self.model_path = Path(model_path)
        self.device = self._resolve_device(device)
        self._model: _OSNet | None = None
        self._model_lock = threading.Lock()

        self._mean = torch.tensor(
            _IMAGENET_MEAN,
        ).view(1, 3, 1, 1)

        self._std = torch.tensor(
            _IMAGENET_STD,
        ).view(1, 3, 1, 1)

    @staticmethod
    def _resolve_device(
        device: str,
    ) -> torch.device:
        if device == "auto":
            return torch.device(
                "cuda" if torch.cuda.is_available() else "cpu",
            )

        return torch.device(device)

    def _ensure_model(self) -> _OSNet:
        """Lazy-load model with double-checked locking."""

        if self._model is not None:
            return self._model

        with self._model_lock:
            if self._model is not None:
                return self._model

            model = _build_osnet_x0_25()

            if self.model_path.exists():
                try:
                    checkpoint = torch.load(
                        self.model_path,
                        map_location="cpu",
                        weights_only=True,
                    )
                except (RuntimeError, ValueError, TypeError, OSError, EOFError, AttributeError):
                    checkpoint = torch.load(
                        self.model_path,
                        map_location="cpu",
                        weights_only=False,
                    )

                # Handle different checkpoint formats
                if isinstance(checkpoint, dict):
                    if "state_dict" in checkpoint:
                        state_dict = checkpoint["state_dict"]
                    elif "model" in checkpoint:
                        state_dict = checkpoint["model"]
                    else:
                        state_dict = checkpoint
                else:
                    state_dict = checkpoint

                # Clean keys: strip DataParallel prefix,
                # skip classifier weights
                cleaned = {}

                for key, value in state_dict.items():
                    clean_key = key

                    clean_key = clean_key.removeprefix("module.")

                    if "classifier" in clean_key:
                        continue

                    cleaned[clean_key] = value

                model.load_state_dict(
                    cleaned,
                    strict=False,
                )
                print(f"[REID] OSNet-x0.25 loaded weights from {self.model_path} on {self.device}")
            else:
                print(f"[REID] OSNet-x0.25 initialized with architecture defaults on {self.device}")

            model.eval()
            model.to(self.device)

            self._mean = self._mean.to(self.device)
            self._std = self._std.to(self.device)

            self._model = model

            return self._model

    def _preprocess(
        self,
        image: np.ndarray,
    ) -> torch.Tensor:
        """Preprocess single BGR crop to tensor."""
        if image is None or getattr(image, "size", 0) == 0:
            raise ValueError("Input image or crop is empty")

        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.ndim == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

        rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

        resized = cv2.resize(
            rgb,
            (128, 256),
        )

        tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
        tensor = tensor.unsqueeze(0).to(self.device)
        tensor = (tensor - self._mean) / self._std

        return tensor

    def _preprocess_batch(
        self,
        images: list[np.ndarray],
    ) -> torch.Tensor:
        """Preprocess batch of BGR crops."""

        tensors = []

        for image in images:
            if image is None or getattr(image, "size", 0) == 0:
                continue
            if image.ndim == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif image.ndim == 3 and image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

            rgb = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB,
            )

            resized = cv2.resize(
                rgb,
                (128, 256),
            )

            tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0

            tensors.append(tensor)

        if not tensors:
            raise ValueError("No valid images provided in batch")

        batch = torch.stack(tensors).to(self.device)
        batch = (batch - self._mean) / self._std

        return batch

    def extract(
        self,
        image: np.ndarray | str | Path,
    ) -> np.ndarray:
        """
        Extract normalized 512-dim embedding
        from a single BGR person crop or image path.
        """
        if isinstance(image, (str, Path)):
            loaded = cv2.imread(str(image))
            if loaded is None:
                raise ValueError(f"Unable to read image: {image}")
            image = loaded

        model = self._ensure_model()

        tensor = self._preprocess(image)

        with torch.no_grad():
            embedding = model(tensor).cpu().numpy().flatten()

        norm = float(np.linalg.norm(embedding))

        embedding = embedding / (norm + 1e-8)

        return embedding.astype(np.float32)

    def extract_batch(
        self,
        images: list[np.ndarray],
    ) -> list[np.ndarray]:
        """
        Extract normalized embeddings from
        a batch of BGR person crops.
        """

        if not images:
            return []

        model = self._ensure_model()

        batch = self._preprocess_batch(images)

        with torch.no_grad():
            embeddings = model(batch).cpu().numpy()

        results = []

        for embedding in embeddings:
            norm = np.linalg.norm(embedding)

            normalized = embedding / (norm + 1e-8)

            results.append(
                normalized.astype(np.float32),
            )

        return results
