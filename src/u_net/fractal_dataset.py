"""This file contains the FractalDataSet class, which is a subclass of the PyTorch Dataset class."""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch.utils.data import Dataset

from .image_dataset import ImageDataset

if TYPE_CHECKING:
    from pathlib import Path

    from torch import Tensor
    from torchvision.transforms import Transform


class FractalDataset(Dataset):
    """FractalDataSet class."""

    original_images: ImageDataset
    edited_images: ImageDataset
    original_dims: list[float]

    def __init__(self, edited_images: list[Path], original_images: list[Path], original_dims: list[float], transform: Transform = None) -> None:
        """Initialize the FractalDataSet class."""
        if transform:
            self.original_images = ImageDataset(images=original_images, transform=transform)
            self.edited_images = ImageDataset(images=edited_images, transform=transform)
        else:
            self.original_images = ImageDataset(images=original_images)
            self.edited_images = ImageDataset(images=edited_images)
        self.original_dims = original_dims

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.original_dims)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, float]:
        """Return the item at the given index."""
        return self.original_images[idx], self.edited_images[idx], self.original_dims[idx]
