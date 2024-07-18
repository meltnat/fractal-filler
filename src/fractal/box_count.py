"""This file contains the implementation of the box count algorithm."""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor
from utils import log

logger = log(__name__)


def lsm(x: Tensor | np.ndarray, y: Tensor | np.ndarray) -> Tensor | float:
    """Least squares method."""
    a00 = min(len(x), len(y))
    a01 = x.sum()
    a02 = y.sum()
    a11 = (x**2).sum()
    a12 = (x * y).sum()
    return (a00 * a12 - a01 * a02) / (a00 * a11 - a01**2)


def resize(image: Tensor) -> Tensor:
    """Resize an image."""
    image = image.squeeze(0).squeeze_(0)
    x, y = image.shape
    data = torch.zeros(x // 2, y // 2, device=image.device)
    for i in range(x):
        for j in range(y):
            if image[i, j]:
                data[i // 2, j // 2] = 1
    return data.unsqueeze_(0).unsqueeze_(0)


def resize_cpu(image: np.ndarray) -> np.ndarray:
    """Resize an image."""
    x, y = image.shape
    data = np.zeros((x // 2, y // 2))
    for i in range(x):
        for j in range(y):
            if image[i, j]:
                data[i // 2, j // 2] = 1
    return data


def count(image: Tensor | np.ndarray, fractal_step: any, size: int = 7) -> float:
    """Count the fractal dimension of an image."""
    dq = -1.0
    if isinstance(image, Tensor):
        data = [image.clone().unsqueeze_(0)]
        for _ in range(size - 1):
            data.append(fractal_step(data[-1]))
            counts = [d.sum().unsqueeze_(0) for d in data]
            boxes = torch.tensor([2**_ for _ in range(len(data))], device=image.device)
        dq = -lsm(boxes.log().squeeze_(0), torch.cat(counts).log().squeeze_(0))
    elif isinstance(image, np.ndarray):
        data = [image.copy()]
        for _ in range(size - 1):
            data.append(fractal_step(data[-1]))
            counts = np.array([d.sum() for d in data])
            boxes = np.array([2**_ for _ in range(len(data))])
        dq = -lsm(np.log(boxes), np.log(counts))
    return dq
