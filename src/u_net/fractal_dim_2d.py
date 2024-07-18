"""This file contains the implementation of the FractalDim2d class, which is a."""

import torch
from torch import Tensor, nn


class FractalDim2d(nn.Module):
    """This class is used to calculate the fractal dimension of a 2D image."""

    fractal_step: nn.Module
    n: int

    def __init__(self, n_counts: int, fractal_step: nn.Module) -> None:
        """Initialize the FractalDim2d class."""
        super().__init__()
        self.fractal_step = fractal_step
        self.n = n_counts

    @staticmethod
    def lsm(x: Tensor, y: Tensor) -> Tensor:
        """Least squares method."""
        a00 = min(len(x), len(y))
        a01 = x.sum()
        a02 = y.sum()
        a11 = (x**2).sum()
        a12 = (x * y).sum()
        return (a00 * a12 - a01 * a02) / (a00 * a11 - a01**2)

    def forward(self, image: Tensor) -> Tensor:
        """Calculate the fractal dimension of the input image."""
        data = [image.clone().unsqueeze_(0)]
        for _ in range(self.n - 1):
            data.append(self.fractal_step(data[-1]))
            counts = [d.sum().unsqueeze_(0) for d in data]
            boxes = torch.tensor([2**_ for _ in range(len(data))], device=image.device)
        return -FractalDim2d.lsm(boxes.log().squeeze_(0), torch.cat(counts).log().squeeze_(0))
