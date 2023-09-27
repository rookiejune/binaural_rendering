import torch
from torch import Tensor


def l1(x: Tensor, y: Tensor) -> Tensor:
    return torch.mean(torch.abs(x - y))

def l2(input: Tensor, target: Tensor) -> Tensor:
    return torch.mean((input - target) ** 2)
