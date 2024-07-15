"""This file contains the dataset class for training the U-Net model."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

if TYPE_CHECKING:
    from pathlib import Path

    from torch import Tensor


class ImageDataset(Dataset):
    """Dataset class for training the U-Net model."""

    images: list[str]
    input_path: Path
    target_path: Path

    def __init__(self: ImageDataset, csv: Path, input_path: Path, target_path: Path) -> None:
        """Initialize the dataset class."""
        self.images = csv.read_text().split("\n")
        self.input_path = input_path
        self.target_path = target_path

    def __len__(self: ImageDataset) -> int:
        """Return the length of the dataset."""
        return len(self.images)

    def to_tensor(self, path: Path) -> Tensor:
        """Convert the image at the given path to a tensor."""
        image = Image.open(path).convert("L").resize((512, 512))
        tensor = transforms.ToTensor()(image)
        tensor[tensor > 0] = 1
        return tensor

    def __getitem__(self: ImageDataset, index: int) -> Tensor:
        """Return the item at the given index."""
        return (
            self.to_tensor(self.input_path / self.images[index]),
            self.to_tensor(self.target_path / self.images[index]),
        )
