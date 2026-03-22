"""
Main point of RAG
"""
import os
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
from src.helpers.utils import _collection_is_ready, _load_md_files
from src.helpers.hashing_files import FileHashChecker
from src.vector_db.vector_db import VectorDatabase

_EMBEDDER_CONFIG = EmbedderConfig(
    parent_chunk_size=1024,
    parent_chunk_overlap=256,
    child_chunk_size=512,
    child_chunk_overlap=128,
)


def _build_agent() -> RAGAgent:
    """
    Method that creates embedder and agent.
    Skips populating if collection already exists.

    Returns:
        RAGAgent: Instance of agent
    """
    directory = Path('Agentic-RAG/data/raw_texts/md_storage')
    checker = FileHashChecker()
    populate = False
    for file in directory.iterdir():
        populate = not checker.check_file(str(file))
        if populate is True:
            break

    embedder = Embedder(
        config=_EMBEDDER_CONFIG,
        embeddings_model=HuggingFaceEmbeddings,
        sparse_model=EmbedSparse,
        vector_db=VectorDatabase,
        recreate_collection=populate,
    )

    if populate:
        texts, doc_ids = _load_md_files()
        embedder.add_documents(texts=texts, document_ids=doc_ids)
        logger.info("Documents loaded into vector store (%d files)", len(texts))
    else:
        logger.info("Reusing existing vector store — skipping populate")

    llm = ChatOllama(
        model=LLMsAndVectorizersStorage.GENERATION_LLM.value, temperature=1.2
    )
    return RAGAgent(embedder, llm)


_agent = _build_agent()


def chat(message: str, _) -> Iterable:
    """
    Main method that operates running chat

    Args:
        message (str): Input message

    Returns:
        Iterable: Answer through tokens
    """
    full_response = ""
    for token in _agent.stream(message):
        full_response += token
        yield full_response


if __name__ == "__main__":
    gr.ChatInterface(chat).launch()
