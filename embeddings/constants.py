"""
Constants for processing embeddings
"""

from enum import Enum, StrEnum
from pathlib import Path

PROJECT_ROOT = PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data"


class PathsStorage(Enum):
    """
    Storage for paths
    """

    QDRANT_PATH = DATA_PATH / "qdrant_db"
    PARENT_CHUNKS_PATH = DATA_PATH / "chunks"
    CHILD_COLLECTION = "document_child_chunks"


class LLMsAndVectorizersStorage(StrEnum):
    """
    Storage for LLMs and vectorizers that are used
    """

    DENSE_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
    SPARSE_MODEL_NAME = "Qdrant/bm25"
