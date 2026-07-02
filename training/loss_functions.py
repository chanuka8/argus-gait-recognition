import torch.nn as nn

from models.architectures.losses import TripletLoss


def build_classification_loss() -> nn.Module:
    return nn.CrossEntropyLoss()


def build_triplet_loss(margin: float = 0.3) -> TripletLoss:
    return TripletLoss(margin=margin)