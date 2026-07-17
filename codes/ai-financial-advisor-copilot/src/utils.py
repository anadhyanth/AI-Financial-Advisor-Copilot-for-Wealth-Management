"""Shared utility helpers: logging setup and pickle IO for the vector store."""

import logging
import pickle
import sys
from pathlib import Path
from typing import Any

import config


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(config.LOG_LEVEL)
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def save_pickle(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: Path) -> Any:
    if not Path(path).exists():
        raise FileNotFoundError(f"Nothing found at {path}. Run ingestion first.")
    with open(path, "rb") as f:
        return pickle.load(f)
