"""This file contains the FractalDataSet class, which is a subclass of the PyTorch Dataset class."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from torch.utils.data import Dataset

from .image_dataset import ImageDataset

if TYPE_CHECKING:
    from torch import Tensor
    from torchvision.transforms import Transform


class FractalDataset(Dataset):
    """FractalDataSet class."""

    origin: ImageDataset
    edited: ImageDataset
    dims: list[float]
    weights: list[float]
    histogram: list[int]
    w: float = 1e-2

    def __init__(self, edited_images: list[Path], original_images: list[Path], original_dims: list[float], transform: Transform = None) -> None:
        """Initialize the FractalDataSet class."""
        if transform:
            self.origin = ImageDataset(images=original_images, transform=transform)
            self.edited = ImageDataset(images=edited_images, transform=transform)
        else:
            self.origin = ImageDataset(images=original_images)
            self.edited = ImageDataset(images=edited_images)
        self.dims = original_dims
        self.histogram = [0] * (int(2 / self.w))
        for dim in self.dims:
            self.histogram[int(dim / self.w)] += 1

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.dims)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, float, float]:
        """Return the item at the given index."""
        return self.origin[idx], self.edited[idx], self.dims[idx], 1 - (self.histogram[int(self.dims[idx] / self.w)] / len(self.dims))
