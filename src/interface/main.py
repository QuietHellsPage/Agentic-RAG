"""
Main point of RAG
"""

from pathlib import Path
from typing import Iterable

import gradio as gr
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

from src.agent.agent import RAGAgent
from src.config.constants import LLMsAndVectorizersStorage
from src.config.constants import LOGGER as logger
from src.config.models import EmbedderConfig
from src.embeddings.embedder import Embedder, EmbedSparse
from src.vector_db.vector_db import VectorDatabase

_DATA_FILE = Path("data/raw_texts/md_storage/US_Code_Title_18.md")

_EMBEDDER_CONFIG = EmbedderConfig(
    parent_chunk_size=1024,
    parent_chunk_overlap=256,
    child_chunk_size=512,
    child_chunk_overlap=128,
)


def _build_agent(populate: bool = True) -> RAGAgent:
    """
    Creates embedder and agent once

    Args:
        populate (bool): Flag to recreate embedder

    Returns:
        RAGAgent: Instance of agent
    """
    embedder = Embedder(
        config=_EMBEDDER_CONFIG,
        embeddings_model=HuggingFaceEmbeddings,
        sparse_model=EmbedSparse,
        vector_db=VectorDatabase,
        recreate_collection=populate,
    )

    if populate:
        if not _DATA_FILE.exists():
            raise FileNotFoundError(f"Data file not found: {_DATA_FILE}")
        data = _DATA_FILE.read_text(encoding="utf-8")
        embedder.add_documents(texts=[data], document_ids=["pinker"])
        logger.info("Documents loaded into vector store")
    else:
        logger.info("Using existing vector store")

    llm = ChatOllama(model=LLMsAndVectorizersStorage.GRAPH_LLM.value, temperature=1.2)
    return RAGAgent(embedder, llm)


_agent = _build_agent(populate=True)


def chat(message: str, _) -> Iterable:
    """
    Main method that operates running chat

    Args:
        message (str): Inpute message

    Returns:
        Iterable: Answer through tokens
    """
    full_response = ""
    for token in _agent.stream(message):
        full_response += token
        yield full_response


if __name__ == "__main__":
    gr.ChatInterface(chat).launch()
