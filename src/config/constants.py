"""
Constants for processing workflow
"""

import logging
from enum import Enum, StrEnum
from pathlib import Path

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
LOGGER = logging.getLogger(name=__name__)


PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_PATH = PROJECT_ROOT / "data"

EMBEDDINGS_DEVICE_ENV = "EMBEDDINGS_DEVICE"

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
    RAW_PDF_COLLECTION = DATA_PATH / "raw_texts" / "pdf_storage"
    RAW_MD_COLLECTION = DATA_PATH / "raw_texts" / "md_storage"


class PromptsStorage(StrEnum):
    """
    Storage for prompts
    """

    NEEDS_PARENT_CHUNK_PROMPT = """
                    Based on the question and the available text chunk, 
                    determine if you need MORE CONTEXT to properly answer the question.

                    Question: {question}
                
                    Available text chunk: {child_chunk}
                
                    Do you need the full parent chunk (larger context) to answer this question?
                    Answer ONLY "yes" or "no".
                    """

    RESPONCE_PROMPT = """
                    You are a helpful AI assistant that answers questions based on the provided context.
                                    
                    Rules:
                        1. Only use information from the provided context to answer questions
                        2. If the context doesn't contain enough information, say so honestly
                        3. Be specific and cite relevant parts of the context
                        4. Keep your answers clear and concise
                        5. If you're unsure, admit it rather than guessing
                                    
                    Context: {context}
                                    
                    Question: {question}
                                    
                    Answer based on the context above:
                    """

    REFORMULATE_PROMPT = """
                    The following question did not return relevant results from the knowledge base.
                    Rephrase it to improve semantic search recall: use synonyms, change the structure,
                    or make it more general. Return ONLY the rephrased question, nothing else.

                    Original question: {question}

                    Rephrased question:
                    """


class LLMsAndVectorizersStorage(Enum):
    """
    Storage for LLMs and vectorizers that are used
    """

    DENSE_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
    SPARSE_MODEL_NAME = "Qdrant/bm25"
    GENERATION_LLM = "mistral"


class GraphLabelsStorage(StrEnum):
    """
    Storage of enum values for LLM to identify entities
    """

    PERSON = "person"
    COMPANY = "company"
    PRODUCT = "product"


class GraphAllowedConstants(Enum):
    """
    Storage for constants required for extracting entities
    """

    ALLOWED_NODES = ["Person", "Organization", "Location", "Award", "ResearchField"]
    ALLOWED_RELATIONSHIPS = [
        ("Person", "SPOUSE", "Person"),
        ("Person", "AWARD", "Award"),
        ("Person", "WORKS_AT", "Organization"),
        ("Organization", "IN_LOCATION", "Location"),
        ("Person", "FIELD_OF_RESEARCH", "ResearchField"),
    ]
    NODE_PROPERTIES = True
    RELATIONSHIPS_PROPERTIES = True
    STRICT_MODE = False


class GraphInitializerStorage(Enum):
    """
    Storage for graph db initialization
    """

    URL = "neo4j://localhost:7687"
    USERNAME = "neo4j"
    PASSWORD = "neo4jneo4j"
    REFRESH_SCHEMA = False


ENUM_VALUES = [item.value for item in GraphLabelsStorage]

EMBEDDINGS_SIZE = 1024
