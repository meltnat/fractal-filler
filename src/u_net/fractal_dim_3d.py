"""This file contains the implementation of the FractalDim2d class, which is a."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class FractalDim3d(nn.Module):
    """This class is used to calculate the fractal dimension of a 2D image."""

    def __init__(self, n_counts: int, bounds: Tensor) -> None:
        """Initialize the FractalDim2d class."""
        super().__init__()
        self.n = torch.tensor(n_counts)
        self.range = torch.arange(self.n)
        self.q = (2 ** self.range.clone()).log()
        self.ql = self.q.numel()
        self.qs = self.q.sum()
        self.qs2 = self.qs**2
        self.q2s = (self.q**2).sum()
        self.bounds = bounds
        self.pool = nn.AvgPool3d(kernel_size=2, stride=2, divisor_override=1)

    def to(self, device: str) -> FractalDim3d:
        """Move the FractalDim2d class to the specified device."""
        super().to(device)
        self.n = self.n.to(device)
        self.range = self.range.to(device)
        self.q = self.q.to(device)
        self.bounds = self.bounds.to(device)
        return self

    def dn(self, n: Tensor, x: list[Tensor]) -> Tensor:
        """Calculate the fractal dimension of the input image."""
        maps = []
        for i in x:
            tensor = i.clone().flatten()
            maps.append(tensor[tensor != 0])
        ni = n.item()
        a = 1.0
        b = maps[0].sum()
        if ni != 0.0:
            maps = [i / b for i in maps]
        if ni == 1.0:
            maps = [i * i.log() for i in maps]
            p = torch.stack([i.sum() for i in maps])
        else:
            a /= ni - 1
            maps = [i**n for i in maps]
            p = torch.stack([i.sum() for i in maps]).log()
        a00 = min(p.numel(), self.ql)
        return a * (a00 * (p * self.q).sum() - p.sum() * self.qs) / (a00 * self.q2s - self.qs2)

    def forward(self, image: Tensor) -> Tensor:
        """Calculate the fractal dimension of the input image."""
        maps = [image.clone()]
        for _ in torch.arange(1, self.n):
            maps.append(self.pool(maps[-1]))
        return torch.stack(
            [
                torch.stack(
                    [
                        torch.stack(
                            [self.dn(k, [_[i, j] for _ in maps]) for k in self.bounds],
                        )
                        for j in range(image.shape[1])
                    ],
                )
                for i in range(image.shape[0])
            ],
        )
