"""This file is used to train the model to predict the dq value of the image."""

import datetime
from logging import Logger
from pathlib import Path

import torch
from torch import Tensor, nn, optim
from torch.utils.data import DataLoader

from .loss import Loss


def learn(model: nn.Module, dataloader: DataLoader, epochs: int, out: Path, device: str, logger: Logger) -> None:
    """Learn."""
    torch.backends.cudnn.benchmark = True
    criterion = Loss()
    optimizer = optim.Adam(params=model.parameters(), lr=1e-5)
    size = len(dataloader.dataset)
    total_size = size * epochs
    start = datetime.datetime.now(tz=datetime.timezone.utc)
    batch_size = dataloader.batch_size
    model.train()
    for epoch in range(epochs):
        for batch, (x, _, dq, weight) in enumerate(dataloader):
            x_g, dq_g = x.to(device), dq.to(device)
            y: Tensor = model(x_g)
            optimizer.zero_grad()
            loss = (criterion(dq_g, y) * weight.to(device)).mean()
            loss.backward()
            optimizer.step()
            if (batch + 1) % (size // 100) == 0:
                delta = datetime.datetime.now(tz=datetime.timezone.utc) - start
                dones = batch * batch_size + epoch * size
                logger.info(
                    f"Loss: {loss.item()} Epoch:{epoch} {(batch * batch_size * 100+1) // size}%[{batch * batch_size+1}/{size}] "
                    f"Total:{(dones+1) * 100 // total_size}%[{dones+1}/{total_size}] "
                    f"Spend:{delta} End:{(delta)*(total_size - dones) / (dones + 1)}",
                )
    torch.save(model.state_dict(), out)
