"""
Constants for processing embeddings
"""

from enum import Enum, StrEnum
from pathlib import Path

PROJECT_ROOT = PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data"

TEXT_SPLITTER_SEPARATORS = [
    "\n#",
    "\n##",
    "\n###",
    "\n####",
    "\n#####",
    "\n######",
    "\n\n",
    "\n",
    ".",
    "?",
    "!",
    " ",
    "",
]


class PathsStorage(Enum):
    """
    Storage for paths
    """

    QDRANT_PATH = DATA_PATH / "qdrant_db"
    PARENT_CHUNKS_PATH = DATA_PATH / "chunks"
    CHILD_COLLECTION = "document_child_chunks"
    PARENT_COLLECTION = PARENT_CHUNKS_PATH / "parent_chunks_storage.json"


class LLMsAndVectorizersStorage(StrEnum):
    """
    Storage for LLMs and vectorizers that are used
    """

    DENSE_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
    SPARSE_MODEL_NAME = "Qdrant/bm25"


PROMPT_TEMPLATE = """
You are a helpful AI assistant that answers questions based on the provided context.

Rules:
1. Only use information from the provided context to answer questions
2. If the context doesn't contain enough information, say so honestly
3. Be specific and cite relevant parts of the context
4. Keep your answers clear and concise
5. If you're unsure, admit it rather than guessing

Context:
{context}

Question: {question}

Answer based on the context above:
"""
