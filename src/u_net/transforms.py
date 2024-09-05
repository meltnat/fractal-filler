"""Description: Transforms for the U-Net model."""

from torch import Tensor


class Binarization:
    """Binarize the input image based on a threshold."""

    threshold: float

    def __init__(self, threshold: float) -> None:
        """Initialize Binarization."""
        self.threshold = threshold

    def __call__(self, x: Tensor, fill: float = 1) -> Tensor:
        """Binarize the input tensor."""
        result = x.clone()
        result[result <= self.threshold] = 0
        result[result > self.threshold] = fill
        return result


class Overwrite:
    """Overwrite the input image with a new value."""

    def __call__(self, x: Tensor, y: Tensor) -> Tensor:
        """Overwrite the x tensor with the y tensor."""
        result = x.clone()
        result[y > 0] = y[y > 0]
        return result
