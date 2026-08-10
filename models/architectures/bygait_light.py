import torch
import torch.nn.functional as F
from torch import nn


class ByGaitLight(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 256,
        part_bins: int = 4,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.part_bins = max(1, int(part_bins))

        self.features = nn.Sequential(
            nn.Conv2d(
                1,
                32,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        if self.part_bins > 1:
            self.pool = nn.AdaptiveAvgPool2d((self.part_bins, 1))
            in_features = 128 * self.part_bins
        else:
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            in_features = 128

        self.embedding = nn.Linear(
            in_features,
            embedding_dim,
        )

    def load_state_dict(self, state_dict: dict, strict: bool = True):
        if "embedding.weight" in state_dict:
            ckpt_weight = state_dict["embedding.weight"]
            expected_in = self.embedding.weight.shape[1]
            actual_in = ckpt_weight.shape[1]
            if expected_in != actual_in:
                raise ValueError(
                    f"Incompatible checkpoint: embedding.weight expects in_features={expected_in} "
                    f"(part_bins={self.part_bins}), but checkpoint contains in_features={actual_in}. "
                    f"Model architecture has been upgraded for HPP part-aware pooling. Retraining required."
                )
        return super().load_state_dict(state_dict, strict=strict)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.embedding(x)
        return F.normalize(
            x,
            p=2,
            dim=1,
        )

