"""This file contains the dataset class for training the U-Net model."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

if TYPE_CHECKING:
    from pathlib import Path

    from torch import Tensor
    from torchvision.transforms import Transform


class ImageDataset(Dataset):
    """Dataset class for training the U-Net model."""

    images: list[Path]
    input_path: Path
    target_path: Path
    to_tensor: transforms.ToTensor
    transform: Transform

    def __init__(self: ImageDataset, images: list[Path], transform: Transform = transforms.ToTensor) -> None:
        """Initialize the dataset class."""
        self.images = images
        self.transform = transform()

    def __len__(self: ImageDataset) -> int:
        """Return the length of the dataset."""
        return len(self.images)

    def __getitem__(self: ImageDataset, index: int) -> Tensor:
        """Return the item at the given index."""
        """Convert the image at the given path to a tensor."""
        image = Image.open(self.images[index]).convert("L").resize((512, 512))
        tensor = self.transform(image)
        tensor[tensor > 0] = 1
        return tensor
