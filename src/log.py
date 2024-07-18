"""Log module."""  # noqa: INP001

from __future__ import annotations

import json
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
