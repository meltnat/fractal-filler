"""DqModel."""

from __future__ import annotations

from torch import Tensor, nn

SIZE = 512
STRIDE = 1
KERNEL_SIZE = 5
POOL_SIZE = 2
OUT_CHANNELS = 16
OUT_CHANNELS_2 = 32
OUT_FEATURES = 128
SIZE_2 = (SIZE - KERNEL_SIZE + 1 // STRIDE) // POOL_SIZE
SIZE_3 = (SIZE_2 - KERNEL_SIZE + 1 // STRIDE) // POOL_SIZE


class DqModel(nn.Module):
    """DqModel."""

    def __init__(self: DqModel) -> None:
        """Initialize DqModel."""
        super().__init__()
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=POOL_SIZE)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=OUT_CHANNELS, kernel_size=KERNEL_SIZE)
        self.conv2 = nn.Conv2d(in_channels=OUT_CHANNELS, out_channels=OUT_CHANNELS_2, kernel_size=KERNEL_SIZE)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=OUT_CHANNELS_2 * SIZE_3 * SIZE_3, out_features=OUT_FEATURES)
        self.fc2 = nn.Linear(in_features=OUT_FEATURES, out_features=1)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        x0 = self.conv1(x)
        x1 = self.relu(x0)
        x2 = self.pool(x1)
        x3 = self.conv2(x2)
        x4 = self.relu(x3)
        x5 = self.pool(x4)
        x6 = self.flatten(x5)
        x7 = self.fc1(x6)
        x8 = self.relu(x7)
        x9 = self.fc2(x8)
        return x9 * 2
