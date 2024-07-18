"""This file contains the implementation of the box count algorithm."""

from __future__ import annotations

import numpy as np
import torch
from log import log
from torch import Tensor, nn

logger = log(__name__)


def lsm(x: Tensor, y: Tensor) -> Tensor:
    """Least squares method."""
    a00 = min(len(x), len(y))
    a01 = x.sum()
    a02 = y.sum()
    a11 = (x**2).sum()
    a12 = (x * y).sum()
    return (a00 * a12 - a01 * a02) / (a00 * a11 - a01**2)


def count_pool(image: Tensor) -> Tensor:
    """Count the fractal dimension of an image."""
    pool = nn.MaxPool2d(2, 2)
    data = [image.clone().unsqueeze_(0)]
    for _ in range(6):
        data.append(pool(data[-1]))
    counts = [data[i].sum().unsqueeze_(0) for i in range(len(data))]
    boxes = torch.tensor([1, 2, 4, 8, 16, 32, 64], device=image.device)
    return -lsm(boxes.log().squeeze_(0), torch.cat(counts).log().squeeze_(0))


def count_resize(image: Tensor) -> Tensor:
    """Count the fractal dimension of an image."""
    data = image.clone().squeeze_(0)
    x, y = data.shape
    data2 = torch.zeros(x // 2, y // 2, device="cuda")
    data4 = torch.zeros(x // 4, y // 4, device="cuda")
    data8 = torch.zeros(x // 8, y // 8, device="cuda")
    data16 = torch.zeros(x // 16, y // 16, device="cuda")
    data32 = torch.zeros(x // 32, y // 32, device="cuda")
    data64 = torch.zeros(x // 64, y // 64, device="cuda")
    for i in range(x):
        for j in range(y):
            if data[i, j]:
                data2[i // 2, j // 2] = 1
                data4[i // 4, j // 4] = 1
                data8[i // 8, j // 8] = 1
                data16[i // 16, j // 16] = 1
                data32[i // 32, j // 32] = 1
                data64[i // 64, j // 64] = 1
    counts = torch.cat(
        data.sum().reshape(1),
        data2.sum().reshape(1),
        data4.sum().reshape(1),
        data8.sum().reshape(1),
        data16.sum().reshape(1),
        data32.sum().reshape(1),
        data64.sum().reshape(1),
    )
    boxes = torch.tensor([1, 2, 4, 8, 16, 32, 64], device="cuda")
    return -lsm(boxes.log().squeeze_(0), counts.log().squeeze_(0))


def count_cpu(image: np.ndarray) -> float:
    """Count the fractal dimension of an image."""
    x, y = image.shape
    data = image.copy()
    data2 = np.zeros((x // 2, y // 2))
    data4 = np.zeros((x // 4, y // 4))
    data8 = np.zeros((x // 8, y // 8))
    data16 = np.zeros((x // 16, y // 16))
    data32 = np.zeros((x // 32, y // 32))
    data64 = np.zeros((x // 64, y // 64))
    for i in range(x):
        for j in range(y):
            if data[i, j]:
                data2[i // 2, j // 2] = 1
                data4[i // 4, j // 4] = 1
                data8[i // 8, j // 8] = 1
                data16[i // 16, j // 16] = 1
                data32[i // 32, j // 32] = 1
                data64[i // 64, j // 64] = 1
    counts = np.array(
        [data.sum(), data2.sum(), data4.sum(), data8.sum(), data16.sum(), data32.sum(), data64.sum()],
    )
    boxes = np.array([1, 2, 4, 8, 16, 32, 64])
    return -lsm(np.log(boxes), np.log(counts))
