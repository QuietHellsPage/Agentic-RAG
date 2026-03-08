"""
Factories and helpers for agent working pipeline
"""

from collections.abc import Callable
from typing import Literal

from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from tqdm import tqdm

from src.config.constants import LOGGER as logger
from src.config.constants import PromptsStorage
from src.config.models import RAGState
from src.embeddings.embedder import Embedder


def make_retrieve_children(embedder: Embedder) -> Callable:
    """
    Child chunks retrieving factory

    Args:
        embedder (Embedder): embedder for child chunks

    Returns:
        Callable: Function that retrieves child chunks
    """

    search_tool = embedder.get_tools()[0]

    def retrieve_children(state: RAGState) -> dict[str, list]:
        """
        Function that retrieves child chunks

        Args:
            state (RAGState): State of the agent

        Returns:
            dict[str, list]: Found chunks
        """
        question = state["question"]
        logger.info("retrieve_children: query=%r", question)

        result = search_tool.invoke({"query": question, "limit": 4})

        if not result.found:
            logger.warning("retrieve_children: no relevant chunks found")
            return {"child_chunks": []}

        logger.info("retrieve_children: found %d chunks", len(result.chunks))
        return {"child_chunks": result.chunks}

    return retrieve_children


def make_reformulate_query(llm: ChatOllama) -> Callable:
    """
    Question reformulating factory

    Args:
        llm (ChatOllama): llm for judging and reformulating

    Returns:
        Callable: Function that reformulates question
    """

    def reformulate_query(state: RAGState) -> dict[str, str | bool | list]:
        """
        Function that reformulates question

        Args:
            state (RAGState): State of the agent

        Returns:
            dict[str, str | bool]: Info about question
        """
        original = state["original_question"]
        logger.info("reformulate_query: reformulating %r", original)

        prompt = PromptsStorage.REFORMULATE_PROMPT.value.format(question=original)
        response = llm.invoke(prompt)
        new_question = str(response.content).strip()

        logger.info("reformulate_query: new question=%r", new_question)
        return {
            "question": new_question,
            "reformulated": True,
            "child_chunks": [],
            "parent_chunks": "",
        }

    return reformulate_query


def make_retrieve_parents(embedder: Embedder) -> Callable:
    """
    Parent chunks retreiving factory

    Args:
        embedder (Embedder): embedder for searching parent chunks

    Returns:
        Callable: Function that retrieves parent chunks
    """

    parent_tool = embedder.get_tools()[1]

    def retrieve_parents(state: RAGState) -> dict[str, str]:
        """
        Function that retrieves parent chunks

        Args:
            state (RAGState): State of the agent

        Returns:
            dict[str, str]: Context
        """
        child_chunks = state["child_chunks"]

        if not child_chunks:
            return {"parent_chunks": ""}

        seen = set()
        unique = []
        for chunk in child_chunks:
            key = (chunk.parent_id, chunk.document_id)
            if key not in seen:
                seen.add(key)
                unique.append(chunk)

        parent_texts = []
        for chunk in tqdm(unique, desc="Retrieving parent chunks"):
            result = parent_tool.invoke(
                {"parent_id": chunk.parent_id, "document_id": chunk.document_id}
            )
            if result.found:
                parent_texts.append(result.content)

        combined = "\n\n---\n\n".join(parent_texts)
        logger.info("retrieve_parents: retrieved %d chunks", len(parent_texts))
        return {"parent_chunks": combined}

    return retrieve_parents


def make_generate(llm: ChatOllama) -> Callable:
    """
    Answer generation factory

    Args:
        llm (ChatOllama): llm for generating answer

    Returns:
        Callable: Function that generates answer
    """

    def generate(state: RAGState) -> dict[str, str]:
        """
        Function that generates answer

        Args:
            state (RAGState): State of the agent

        Returns:
            dict[str, str]: Answer
        """
        question = state["original_question"]
        parent_chunks = state.get("parent_chunks", "")
        child_chunks = state.get("child_chunks", [])

        if parent_chunks:
            context = parent_chunks
        elif child_chunks:
            context = "\n\n".join(c.content for c in child_chunks)
        else:
            logger.warning("generate: no context available")
            context = "No relevant information was found in the knowledge base."

        prompt = PromptsStorage.RESPONCE_PROMPT.value.format(
            context=context, question=question
        )
        response = llm.invoke(prompt)
        answer = str(response.content).strip()
        logger.info("generate: answer produced")
        return {"answer": answer}

    return generate


def route_after_retrieve_children(
    state: RAGState,
) -> Literal["retrieve_parents", "reformulate_query", "generate"]:
    """
    Function that operates agent actions

    Args:
        state (RAGState): State of the agent

    Returns:
        Literal["retrieve_parents", "reformulate_query", "generate"]: Info about state
    """
    if state["child_chunks"]:
        return "retrieve_parents"
    if not state.get("reformulated", False):
        return "reformulate_query"
    return "generate"


def build_rag_graph(embedder: Embedder, llm: ChatOllama) -> CompiledStateGraph:
    """
    Function that creates agent graph

    Args:
        embedder (RAGSEmbeddertate): Embedder for chunks
        llm (ChatOllama): llm for reformulationg query

    Returns:
        CompiledStateGraph: Info about state
    """
    graph = StateGraph(RAGState)

    graph.add_node("retrieve_children", make_retrieve_children(embedder))
    graph.add_node("reformulate_query", make_reformulate_query(llm))
    graph.add_node("retrieve_parents", make_retrieve_parents(embedder))
    graph.add_node("generate", make_generate(llm))

    graph.add_edge(START, "retrieve_children")
    graph.add_conditional_edges("retrieve_children", route_after_retrieve_children)
    graph.add_edge("reformulate_query", "retrieve_children")
    graph.add_edge("retrieve_parents", "generate")
    graph.add_edge("generate", END)

    return graph.compile()
