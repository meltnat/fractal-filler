#!/usr/bin/env python

"""this is the main file of the Fractal Dimension project."""

from __future__ import annotations

import datetime
import multiprocessing
import os
import random
import time
from concurrent import futures
from pathlib import Path

import log
import numpy as np
import torch
from fractal import dq_from_tensor, dq_multi
from PIL import Image
from torch import Tensor, nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from u_net import ImageDataset, Loss, UNet

logger = log.log(__name__)

DATASETS = Path("data/datasets.csv")
ORIGIN = Path("data/origin")
EDITED = Path("data/edited")
MODEL = Path("data/model.pth")
MAX_WORKERS = os.cpu_count()
OUT = Path("data/out")
BATCH_SIZE = 8
EPOCHS = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def test() -> None:
    """Test."""
    labels = load_files()
    result = labeling_images(labels=labels)
    DATASETS.write_text("\n".join([f"{labels[i]},{result[i]}" for i in result.shape[0]]))
    return
    main()


def main() -> None:
    """Main function."""
    logger.info("Starting...")
    logger.info("Device: %", DEVICE)
    train_data = ImageDataset(csv=PATHS, input_path=EDITED, target_path=ORIGIN)
    train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=MAX_WORKERS)
    logger.info("Data size:%", len(train_data))
    model = UNet(in_channels=1).to(DEVICE)
    logger.info("Start Learning...")
    learn(model=model, dataloader=train_loader)
    model.load_state_dict(torch.load(MODEL))

    test_data = ImageDataset(csv=TEST, input_path=EDITED, target_path=ORIGIN)
    test_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, num_workers=MAX_WORKERS)

    count = 0
    for file in OUT.iterdir():
        file.unlink()
    for x, _ in test_loader:
        with torch.no_grad():
            output, _ = model(x.to(DEVICE))
        for o in output:
            Image.fromarray((o.cpu().squeeze_(1).squeeze_(1).numpy() * 255).astype(np.uint8)).save(OUT / f"{count}.png")
            count += 1


def learn(model: nn.Module, dataloader: DataLoader) -> None:
    """Learn."""
    bcl = Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    size = len(dataloader.dataset)
    start = datetime.datetime.now(tz=datetime.timezone.utc)
    for epoch in range(EPOCHS):
        model.train()
        result = []
        for batch, (images, target) in enumerate(dataloader):
            images_g = images.to(DEVICE)
            target_g = target.to(DEVICE)
            optimizer.zero_grad()
            result.append(model(images_g))
            loss = bcl(dq_from_tensor(target_g))
            loss.backward()
            optimizer.step()
            if batch % 10 == 0:
                now = datetime.datetime.now(tz=datetime.timezone.utc)
                logger.info(
                    f"[{batch * len(images):>5d}/{size:>5d}]"  # noqa: G004
                    f"Spend:{now - start} End:{(now - start)*(1-(epoch*size+batch*BATCH_SIZE)/EPOCHS*size)}",
                )

    torch.save(model.state_dict(), MODEL)


def edit_image(label: str) -> None:
    """Edit image."""
    img = Image.open(ORIGIN / label)
    w = img.size[0] // 5
    h = img.size[1] // 5
    x = random.randrange(0, img.size[0] - w)
    y = random.randrange(0, img.size[1] - h)
    data = list(img.getdata())
    for i in range(x, x + w):
        for j in range(y, y + h):
            data[j * img.size[0] + i] = (0, 0, 0)
    img.putdata(data)
    img.save(EDITED / label)


def edit_images() -> None:
    """Edit images."""
    print("Editing images...")
    labels = Path("Paths").read_text().split("\n")
    with futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures_list = []
        for label in labels:
            future = executor.submit(edit_image, label)
            future.add_done_callback(lambda f: f.result())
            futures_list.append(future)
        size = len(futures_list)
        while futures_list:
            active_processes = len(multiprocessing.active_children())
            futures_list = [f for f in futures_list if not f.done()]
            print(f"Active processes: {active_processes} / {100 - int(len(futures_list) * 100 / size)}% Done")
            time.sleep(1)


def labeling_images(labels: list[str]) -> Tensor:
    """Label images."""
    logger.info("Labeling images...")
    images: list[Tensor] = []
    for label in labels:
        image = Image.open(ORIGIN / label).convert("L")
        t = transforms.ToTensor()(image)
        t[t > 0] = 1
        t.to(DEVICE)
        images.append(t)
    return dq_multi(images)


def load_files() -> list[str]:
    """Load files."""
    logger.info("Loading files...")
    labels = []
    for dirs in [d for d in ORIGIN.iterdir() if not d.is_file()]:
        if not (EDITED / dirs.name).exists():
            (EDITED / dirs.name).mkdir()
        labels += [f"{dirs.name}/{f.name}" for f in dirs.iterdir() if f.is_file()]
    labels.sort()
    return labels


if __name__ == "__main__":
    test()
