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

import numpy as np
import torch
import utils
from PIL import Image
from torch import Tensor, nn, optim
from torch.utils.data import DataLoader
from u_net import FractalDim2d, ImageDataset, Loss, UNet

logger = utils.log(__name__)

DATASETS = Path("data/datasets.csv")
INDEX = Path("data/index.csv")
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
    labels = INDEX.read_text().split("\n")
    result = labeling_images(labels=labels, fractal_step=nn.MaxPool2d(2, 2))
    DATASETS.write_text("\n".join([f"{labels[i]},{result[i]}" for i in range(result.shape[0])]))
    return
    main()


def main() -> None:
    """Main function."""
    logger.info("Starting...")
    logger.info("Device: %", DEVICE)
    dqs = []
    images = []
    for line in DATASETS.read_text().split("\n"):
        image, dq = line.split(",")
        dqs.append(dq)
        images.append(image)
    train_y = ImageDataset(images=[ORIGIN / _ for _ in images])
    train_x = ImageDataset(images=[EDITED / _ for _ in images])
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
                    f"[{batch * len(images):>5d}/{size:>5d}]"
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


def labeling_images(labels: list[str], fractal_step: any) -> Tensor:
    """Labeling images."""
    logger.info("Labeling images...")
    datasets = ImageDataset([ORIGIN / _ for _ in labels])
    fd2d = FractalDim2d(n_counts=7, fractal_step=fractal_step)
    dqs = utils.loop_with_progress(lambda x: fd2d(x).unsqueeze_(0), datasets, logger)
    return torch.cat(dqs)


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
