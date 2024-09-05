"""This file contains the Loss class used to calculate the loss of the model."""

from torch import Tensor, nn


class Loss(nn.Module):
    """Loss class used to calculate the loss of the model."""

    def __init__(self) -> None:
        """Initialize the Loss class."""
        super().__init__()

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """Forward function used to calculate the loss of the model."""
        return (target - pred) ** 2
