"""
Utils for RAG
"""

import os
from typing import Optional

import torch

from src.config.constants import EMBEDDINGS_DEVICE_ENV
from src.config.constants import LOGGER as logger
from src.config.constants import PathsStorage


def _choose_device(device: Optional[str] = None) -> str:
    """
    Method that chooses device

    Args:
        device (Optional[str]): Device that operates embeddings processing

    Returns:
        str: Chosen device that operates embeddings processing
    """
    if device is None:
        if not (env_device := os.getenv(EMBEDDINGS_DEVICE_ENV)):
            return "cuda" if torch.cuda.is_available() else "cpu"
        return env_device
    return device


def _load_md_files() -> tuple[list[str], list[str]]:
    """
    Method that loads all .md files from md_storage directory.

    Returns:
        tuple[list[str], list[str]]: Texts and their document IDs
    """
    md_dir = PathsStorage.RAW_MD_COLLECTION.value
    md_files = sorted(md_dir.glob("*.md"))

    if not md_files:
        raise FileNotFoundError(f"No .md files found in {md_dir}")

    texts, doc_ids = [], []
    for path in md_files:
        texts.append(path.read_text(encoding="utf-8"))
        doc_ids.append(path.stem)
        logger.info("Loaded file: %s", path.name)

    return texts, doc_ids
