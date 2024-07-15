"""This file contains the implementation of the box count algorithm."""

import cupy as cp
import numpy as np
import torch
from torch import Tensor


def fractal_dimension(image: np.ndarray) -> float:
    """Count the fractal dimension of an image."""
    # Convert the image to a boolean array on gpu.
    data = cp.array(image, dtype=bool)
    x, y = data.shape
    limit = data.shape[0] // 10
    box_sizes = [1]
    counts = [cp.sum(data).get()]

    while box_sizes[-1] < limit:
        box_sizes.append(box_sizes[-1] * 2)
        old = data.copy()
        old_x, old_y = old.shape
        new_x, new_y = -(-old_x // 2), -(-old_y // 2)
        data = cp.zeros((new_x, new_y))
        for i in range(old_x):
            for j in range(old_y):
                if old[i, j]:
                    data[i // 2, j // 2] = 1
        counts.append(cp.sum(data).get())

    log_sizes = np.log(box_sizes)
    log_counts = np.log(counts)

    return abs(np.polyfit(log_sizes, log_counts, 1)[0])


def lsm(x: Tensor, y: Tensor) -> Tensor:
    """Least squares method."""
    a01, a02, a11, a12 = 0, 0, 0, 0
    a00 = min(len(x), len(y))
    for i in range(a00):
        a01 += x[i]
        a02 += y[i]
        a11 += x[i] ** 2
        a12 += x[i] * y[i]
    return (a00 * a12 - a01 * a02) / (a00 * a11 - a01**2)


def count(image: Tensor) -> Tensor:
    """Count the fractal dimension of an image."""
    data = image.clone().squeeze_(0)
    x, y = data.shape
    limit = min(x, y) // 2
    box_sizes = [torch.tensor([1], device="cuda")]
    counts = [data.sum().reshape(1)]

    while box_sizes[-1] < limit:
        box_sizes.append(box_sizes[-1] * 2)
        old = data.clone()
        old_x, old_y = old.shape
        new_x, new_y = -(-old_x // 2), -(-old_y // 2)
        data = torch.zeros(new_x, new_y, device="cuda")
        for i in range(old_x):
            for j in range(old_y):
                if old[i, j]:
                    data[i // 2, j // 2] = 1
        counts.append(data.sum().reshape(1))
    return lsm(torch.cat(box_sizes).log().squeeze_(0), torch.cat(counts).log().squeeze_(0)) * -1


def dq_from_tensor(images: Tensor) -> Tensor:
    """Count the fractal dimension of an image."""
    return torch.cat([count(images[i]).unsqueeze_(0) for i in range(images.shape[0])])
