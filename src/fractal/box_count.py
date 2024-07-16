"""This file contains the implementation of the box count algorithm."""

from __future__ import annotations

import multiprocessing
import os
import time
from concurrent import futures

import log
import numpy as np
import torch
from torch import Tensor

logger = log.log(__name__)


def fractal_dimension(image: np.ndarray) -> float:
    """Count the fractal dimension of an image."""
    # Convert the image to a boolean array on gpu.
    data = np.array(image, dtype=bool)
    x, y = data.shape
    limit = data.shape[0] // 10
    box_sizes = [1]
    counts = [np.sum(data).get()]

    while box_sizes[-1] < limit:
        box_sizes.append(box_sizes[-1] * 2)
        old = data.copy()
        old_x, old_y = old.shape
        new_x, new_y = -(-old_x // 2), -(-old_y // 2)
        data = np.zeros((new_x, new_y))
        for i in range(old_x):
            for j in range(old_y):
                if old[i, j]:
                    data[i // 2, j // 2] = 1
        counts.append(np.sum(data))

    log_sizes = np.log(box_sizes)
    log_counts = np.log(counts)

    return abs(np.polyfit(log_sizes, log_counts, 1)[0])


def lsm(x: Tensor, y: Tensor) -> Tensor:
    """Least squares method."""
    a00 = min(len(x), len(y))
    a01 = x.sum()
    a02 = y.sum()
    a11 = (x**2).sum()
    a12 = (x * y).sum()
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


def count_multi(index: int, images: Tensor) -> tuple[int, Tensor]:
    """Count the fractal dimension of an image."""
    return index, count(images)


def dq_multi(images: Tensor) -> Tensor:
    """Count the fractal dimension of an image."""
    results: list[tuple[int, Tensor]] = []
    with futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures_list = []
        logger.info("Counting fractal dimensions...")
        for i in images.shape[0]:
            future = executor.submit(count_multi, i, images[i])
            futures_list.append(future)
            future.add_done_callback(lambda future: results.append(future.result()))
        size = len(futures_list)
        while futures_list:
            active_processes = len(multiprocessing.active_children())
            logger.info(
                f"Active processes: {active_processes} / {100 - int(len(futures_list) * 100 / size)}% Done ({len(futures_list)})",  # noqa: G004
            )
            time.sleep(1)
            futures_list = [f for f in futures_list if not f.done()]
    results.sort(key=lambda x: x[0])
    results = [x[1].unsqueeze_(0) for x in results]
    return torch.cat(results)
