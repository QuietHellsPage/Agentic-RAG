"""
Agent class
"""

from typing import Iterable

from langchain_ollama import ChatOllama

from src.config.constants import LLMsAndVectorizersStorage
from src.config.constants import LOGGER as logger
from src.embeddings.embedder import Embedder
from src.helpers.agent_factories import build_rag_graph


class RAGAgent:
    """
    Wrapper for LangGraph pipeline
    """

    def __init__(self, embedder: Embedder, llm: ChatOllama) -> None:
        """
        Initialize an instance of RAGAgent

        Args:
            embedder (Embedder): Embedder for chunks
            llm (ChatOllama): agent llm
        """
        self.embedder = embedder
        self.llm = llm
        self._app = build_rag_graph(embedder, llm)

        logger.info(
            "RAGAgent (LangGraph) initialised with model: %s",
            LLMsAndVectorizersStorage.GRAPH_LLM.value,
        )

    def __repr__(self) -> str:
        """
        Method that returns string representation of the class

        Returns:
            str: String representation
        """
        return f"{self.__class__.__name__!r}(" f"{self.embedder=!r}, " f"{self.llm=!r})"

    def stream(self, question: str) -> Iterable:
        """
        Function that shows streaming process of answering

        Args:
            question (str): Input question

        Returns:
            Iterable: stream state
        """
        initial_state = {
            "question": question,
            "original_question": question,
            "child_chunks": "",
            "parent_chunks": "",
            "answer": "",
            "reformulated": False,
        }
        for chunk, *_ in self._app.stream(initial_state, stream_mode="messages"):
            if hasattr(chunk, "content") and chunk.content:
                yield chunk.content
