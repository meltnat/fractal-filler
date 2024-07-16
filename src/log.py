"""Log module."""  # noqa: INP001

from __future__ import annotations

import json
import logging
from pathlib import Path

from typing_extensions import Self

LOGGER = Path("logger.json")


class Logger:
    """Logger class."""

    _instance: Logger = None

    def __new__(cls: Logger, *args: tuple, **kargs: tuple[str, any]) -> Self:  # noqa: ARG003
        """Create a new instance of the class."""
        if not hasattr(cls, "_instance"):
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialize the class."""
        self.load_config()

    def logger(self, name: str) -> logging.Logger:
        """Log the name."""
        return logging.getLogger(name)

    def load_config() -> None:
        """Load the logger configuration."""
        logging.config.dictConfig(json.load(LOGGER.open()))


def log(name: str) -> logging.Logger:
    """Log the name."""
    return Logger().logger(name)
