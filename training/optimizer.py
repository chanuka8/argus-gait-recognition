import torch


def build_optimizer(model, learning_rate: float = 0.0001):
    return torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-4,
    )