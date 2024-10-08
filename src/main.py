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
from PIL import Image
from torch import Tensor, nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

import utils
from u_net import DqModel, FractalDataset, FractalDim2d, ImageDataset, Loss, UNet, learn_dq

logger = utils.log(__name__)

DATASETS = Path("data/datasets.csv")
INDEX = Path("data/index.csv")
ORIGIN = Path("data/origin")
EDITED = Path("data/edited")
MODEL = Path("data/model.pth")
MAX_WORKERS = os.cpu_count()
OUT = Path("data/out")
BATCH_SIZE = 8
EPOCHS = 20
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DATA_SIZE = 10**4
TEST_DATA_SIZE = 10**2


def test() -> None:
    """Test."""
    path1 = "data/test/image.bmp"
    path2 = "data/test/image.png"
    image1 = Image.open(path1).convert("L")
    image2 = Image.open(path2).convert("L")
    tensor = torch.stack([transforms.ToTensor()(image1), transforms.ToTensor()(image2)])
    threshold = 0.5
    tensor[tensor > threshold] = 1
    tensor[tensor <= threshold] = 0
    tensor = tensor - 1
    tensor = tensor * -1
    result = FractalDim2d(10, bounds=torch.tensor([-1.2, 0, 1, 2]))(tensor)
    print(result)
    return
    for i, j in enumerate(FractalDim2d(7, bounds=torch.tensor([0, 1, 2]))(tensor)):
        logger.info(f"Index: {i} Value: {j}")
    return
    dqs: list[float] = []
    images: list[str] = []
    for line in DATASETS.read_text().split("\n"):
        image, dq = line.split(",")
        dqs.append(float(dq))
        images.append(image)
    dataset = FractalDataset(edited_images=[EDITED / _ for _ in images], original_images=[ORIGIN / _ for _ in images], original_dims=dqs)
    train_data, test_data, _ = random_split(
        dataset=dataset,
        lengths=[TRAIN_DATA_SIZE, TEST_DATA_SIZE, len(dataset) - TRAIN_DATA_SIZE - TEST_DATA_SIZE],
    )
    train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=MAX_WORKERS, pin_memory=True)
    model = DqModel().to(DEVICE)
    out_path = Path("data") / (model.__class__.__name__ + ".pth")
    learn_dq.learn(
        model=model,
        dataloader=train_loader,
        epochs=EPOCHS,
        out=out_path,
        device=DEVICE,
        logger=logger,
    )

    model.load_state_dict(torch.load(out_path))

    test_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, num_workers=MAX_WORKERS)

    count = 0
    for file in OUT.iterdir():
        file.unlink()
    dqs = []
    loss = Loss()
    for x, _, dq, _ in test_loader:
        with torch.no_grad():
            output = model(x.to(DEVICE))
        for i, o in enumerate(output):
            count += 1
            dqs.append((dq[i].item(), o.item(), loss(dq[i].to(DEVICE), o).item()))
    Path("data/out.csv").write_text("\n".join([f"{dq[0]},{dq[1]},{dq[2]}" for dq in dqs]))


def main() -> None:
    """Main function."""
    logger.info("Starting...")
    logger.info(f"Device: {DEVICE}")
    return test()
    dqs: list[float] = []
    images: list[str] = []
    for line in DATASETS.read_text().split("\n"):
        image, dq = line.split(",")
        dqs.append(float(dq))
        images.append(image)
    dataset = FractalDataset(edited_images=[EDITED / _ for _ in images], original_images=[ORIGIN / _ for _ in images], original_dims=dqs)
    size = len(dataset)
    train_data, test_data, _ = random_split(dataset=dataset, lengths=[TRAIN_DATA_SIZE, TEST_DATA_SIZE, size - TRAIN_DATA_SIZE - TEST_DATA_SIZE])
    train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=MAX_WORKERS)
    logger.info(f"Train Data size:{len(train_data)}")
    model = UNet(in_channels=1).to(DEVICE)
    logger.info("Start Learning...")
    learn(model=model, dataloader=train_loader)
    model.load_state_dict(torch.load(MODEL))

    test_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, num_workers=MAX_WORKERS)

    count = 0
    for file in OUT.iterdir():
        file.unlink()
    dqs = []
    for y, x, dq in test_loader:
        with torch.no_grad():
            output, pred_dq = model(x.to(DEVICE))
        for i, o in enumerate(output):
            Image.fromarray((o.cpu().squeeze_(0).squeeze_(0).numpy() * 255).astype(np.uint8)).save(OUT / f"{count}.png")
            Image.fromarray((x[i].squeeze_(0).squeeze_(0).numpy() * 255).astype(np.uint8)).save(OUT / f"{count}_y.png")
            count += 1
            dqs.append((dq[i].item(), pred_dq[i].item()))
    Path("data/out.csv").write_text("\n".join([f"{dq[0]},{dq[1]}" for dq in dqs]))


def learn(model: nn.Module, dataloader: DataLoader) -> None:
    """Learn."""
    bcl = Loss()
    dq = FractalDim2d(7, nn.MaxPool2d(kernel_size=2, stride=2))
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    size = len(dataloader.dataset)
    total_size = size * EPOCHS
    start = datetime.datetime.now(tz=datetime.timezone.utc)
    for epoch in range(EPOCHS):
        model.train()
        for batch, (y, edited, dim) in enumerate(dataloader):
            optimizer.zero_grad()
            pred = model(edited.to(DEVICE))
            loss = bcl(dq(y, False), dq(pred, False)).sum()
            loss.backward()
            optimizer.step()
            if (batch + 1) % (size // 100) == 0:
                delta = datetime.datetime.now(tz=datetime.timezone.utc) - start
                dones = batch * BATCH_SIZE + epoch * size
                logger.info(
                    f"Loss: {loss.item()} Epoch:{epoch} {(batch * BATCH_SIZE * 100+1) // size}%[{batch * BATCH_SIZE+1}/{size}] "
                    f"Total:{(dones+1) * 100 // total_size}%[{dones+1}/{total_size}] "
                    f"Spend:{delta} End:{(delta)*(total_size - dones) / (dones + 1)}",
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
    main()
