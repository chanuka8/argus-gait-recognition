import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class BatchHardTripletLoss(nn.Module):
    def __init__(
        self,
        margin: float = 0.3,
    ) -> None:
        super().__init__()
        self.margin = margin

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        if embeddings.size(0) < 2:
            return embeddings.sum() * 0.0

        embeddings = F.normalize(
            embeddings,
            p=2,
            dim=1,
        )

        distance_matrix = torch.cdist(
            embeddings,
            embeddings,
            p=2,
        )

        labels = labels.view(
            -1,
            1,
        )

        positive_mask = labels.eq(
            labels.t(),
        )

        negative_mask = ~positive_mask

        positive_mask.fill_diagonal_(
            False,
        )

        hardest_positive = torch.where(
            positive_mask,
            distance_matrix,
            torch.zeros_like(distance_matrix),
        ).max(dim=1)[0]

        hardest_negative = torch.where(
            negative_mask,
            distance_matrix,
            torch.full_like(distance_matrix, 1e6),
        ).min(dim=1)[0]

        valid_positive = positive_mask.any(
            dim=1,
        )

        valid_negative = negative_mask.any(
            dim=1,
        )

        valid = valid_positive & valid_negative

        if not valid.any():
            return embeddings.sum() * 0.0

        loss = F.relu(
            hardest_positive[valid]
            - hardest_negative[valid]
            + self.margin
        )

        return loss.mean()


class JointGaitLoss(nn.Module):
    def __init__(
        self,
        triplet_margin: float = 0.3,
        triplet_weight: float = 0.5,
    ) -> None:
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.triplet_loss = BatchHardTripletLoss(
            margin=triplet_margin,
        )
        self.triplet_weight = triplet_weight

    def forward(
        self,
        logits: torch.Tensor,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ce = self.ce_loss(
            logits,
            labels,
        )

        triplet = self.triplet_loss(
            embeddings,
            labels,
        )

        total = ce + self.triplet_weight * triplet

        return total, ce, triplet


class ArcMarginProduct(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        s: float = 30.0,
        m: float = 0.50,
        easy_margin: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(
            torch.FloatTensor(out_features, in_features)
        )
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(
        self,
        input: torch.Tensor,
        label: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        cosine = F.linear(
            F.normalize(input),
            F.normalize(self.weight),
        )
        cosine = cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7)

        if label is None:
            return cosine * self.s

        sine = torch.sqrt(1.0 - torch.pow(cosine, 2)).clamp(0, 1)
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros(cosine.size(), device=input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output, cosine * self.s