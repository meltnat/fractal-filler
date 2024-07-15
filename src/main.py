#!/usr/bin/env python

"""this is the main file of the Fractal Dimension project."""

from __future__ import annotations

import multiprocessing
import os
import random
import time
from concurrent import futures
from pathlib import Path

import numpy as np
import torch
from fractal import dq_from_tensor, fractal_dimension
from PIL import Image
from torch import nn, optim
from torch.utils.data import DataLoader
from u_net import ImageDataset, Loss, UNet

DATASETS = Path("data/datasets.csv")
PATHS = Path("data/paths.csv")
ORIGIN = Path("data/origin")
EDITED = Path("data/edited")
MODEL = Path("data/model.pth")
MAX_WORKERS = os.cpu_count() // 2
OUT = Path("data/out")
TEST = Path("data/test.csv")
BATCH_SIZE = 8
EPOCHS = 1


def test() -> None:
    """Test."""
    main()


def main() -> None:
    """Main function."""
    print("Starting...")
    train_data = ImageDataset(csv=PATHS, input_path=EDITED, target_path=ORIGIN)
    train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=MAX_WORKERS)
    model = UNet(in_channels=1).to("cuda")
    learn(model=model, dataloader=train_loader)
    model.load_state_dict(torch.load(MODEL))

    test_data = ImageDataset(csv=TEST, input_path=EDITED, target_path=ORIGIN)
    test_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, num_workers=MAX_WORKERS)

    count = 0
    for file in OUT.iterdir():
        file.unlink()
    for x, _ in test_loader:
        with torch.no_grad():
            output = model(x.to("cuda"))
        for o in output:
            Image.fromarray((o.cpu().numpy() * 255).astype(np.uint8)).save(OUT / f"{count}.png")
            count += 1


def learn(model: nn.Module, dataloader: DataLoader) -> None:
    """Learn."""
    bcl = Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    size = len(dataloader.dataset)
    start = time.time()
    for epoch in range(EPOCHS):
        model.train()
        for batch, (images, target) in enumerate(dataloader):
            images_g = images.to("cuda")
            target_g = target.to("cuda")
            optimizer.zero_grad()
            _, dq = model(images_g)
            loss = bcl(dq, dq_from_tensor(target_g))
            loss.backward()
            optimizer.step()
            if batch % 10 == 0:
                now = time.time()
                print(
                    f"epoch:{epoch}/loss:{loss.item():>7f}/[{batch * len(images):>5d}/{size:>5d}]"
                    f"Spend:{now - start:.2f}s End:{(now - start) * (size / (batch + 1) - 1):.2f}s",
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
    labels = PATHS.read_text().split("\n")
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


def labeling(label: str) -> tuple[float, float, str]:
    """Labeling function."""
    threshold = 1
    np_o = np.array(Image.open(ORIGIN / label).convert("L").point(lambda p: p > threshold and 255))
    np_e = np.array(Image.open(EDITED / label).convert("L").point(lambda p: p > threshold and 255))
    return fractal_dimension(np_o), fractal_dimension(np_e), label


def labeling_images() -> None:
    """Label images."""
    print("Labeling images...")
    labels = PATHS.read_text().split("\n")
    labeled_list = []
    with futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures_list = []
        for label in labels:
            future = executor.submit(labeling, label)
            futures_list.append(future)
            future.add_done_callback(lambda f: labeled_list.append(f.result()))
        size = len(futures_list)
        while futures_list:
            active_processes = len(multiprocessing.active_children())
            print(
                f"Active processes: {active_processes} / {100 - int(len(futures_list) * 100 / size)}% Done ({len(futures_list)})",
            )
            time.sleep(1)
            futures_list = [f for f in futures_list if not f.done()]
    datasets = Path("data/datasets.csv")
    datasets.write_text("\n".join([f"{o},{e},{label}" for o, e, label in labeled_list]))


def load_files() -> None:
    """Load files."""
    print("Loading files...")
    labels = []
    for dirs in [d for d in ORIGIN.iterdir() if not d.is_file()]:
        if not (EDITED / dirs.name).exists():
            (EDITED / dirs.name).mkdir()
        labels += [f"{dirs.name}/{f.name}" for f in dirs.iterdir() if f.is_file()]
    labels.sort()
    PATHS.write_text("\n".join(labels[:990]))
    TEST.write_text("\n".join(labels[990:1000]))


if __name__ == "__main__":
    test()
