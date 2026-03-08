"""
Utils for RAG
"""

import os
from typing import Optional

import torch

from src.config.constants import EMBEDDINGS_DEVICE_ENV


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
