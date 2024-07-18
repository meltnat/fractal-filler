"""This file is created to provide some utility functions."""  # noqa: INP001

from __future__ import annotations

import json
from datetime import datetime, timezone
from logging import Logger, config, getLogger
from pathlib import Path

LOGGER = Path("logger.json")

before_loaded_config = True


def log(name: str) -> Logger:
    """Log the name."""
    global before_loaded_config  # noqa: PLW0603
    if before_loaded_config:
        config.dictConfig(json.load(LOGGER.open()))
        before_loaded_config = False
    return getLogger(name)


def loop_with_progress(step: any, iterable: list, logger: Logger) -> list:
    """Loop with progress."""
    size = len(iterable)
    start = datetime.now(timezone.utc)
    result = []
    logger.info(f"Start: {start} Size: {size} Step: {step.__name__}")
    for i, item in enumerate(iterable):
        result.append(step(item))
        j = i + 1
        if j % (size // 100) == 0:
            delta = datetime.now(timezone.utc) - start
            logger.info(f"{j * 100 // size}%[{j}/{size}] Spend:{delta} End:{delta * (size - i) /j}")
    return result
