"""This file contains the implementation of the FractalDim2d class, which is a."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class FractalDim2d(nn.Module):
    """This class is used to calculate the fractal dimension of a 2D image."""

    def __init__(self, n_counts: int, min_n: float = 0, max_n: float = 9, bounds: Tensor | None = None) -> None:
        """Initialize the FractalDim2d class."""
        super().__init__()
        self.n = torch.tensor(n_counts)
        self.range = torch.arange(self.n)
        self.q = (2 ** self.range.clone()).log()
        self.ql = self.q.numel()
        self.qs = self.q.sum()
        self.bounds = torch.arange(min_n, max_n, (max_n - min_n) / 100.0) if bounds is None else bounds
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2, divisor_override=1)

    def to(self, device: str) -> FractalDim2d:
        """Move the FractalDim2d class to the specified device."""
        super().to(device)
        self.n = self.n.to(device)
        self.range = self.range.to(device)
        self.q = self.q.to(device)
        self.bounds = self.bounds.to(device)
        return self

    def dn(self, n: Tensor, x: list[Tensor]) -> Tensor:
        """Calculate the fractal dimension of the input image."""
        maps = [i.clone().flatten() for i in x]
        maps = [i[i != 0] for i in maps]
        ni = n.item()
        b = maps[0].sum()
        if ni == 1.0:
            a = 1.0
            p = torch.stack([(i * i.log()).sum() for i in [j / b for j in maps]])
        else:
            a = 1.0 / (ni - 1)
            p = torch.stack([(i**n).sum() for i in maps])
        a00 = min(p.numel(), self.ql)
        a01 = p.sum()
        return a * (a00 * (p * self.q).sum() - a01 * self.qs) / (a00 * (p**2).sum() - a01**2)

    def forward(self, image: Tensor) -> Tensor:
        """Calculate the fractal dimension of the input image."""
        maps = [image.clone()]
        for _ in torch.arange(1, self.n):
            maps.append(self.pool(maps[-1]))
        return torch.stack([self.dn(i, maps) for i in self.bounds])
