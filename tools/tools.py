"""
Tools for agent
"""

import json
from typing import TYPE_CHECKING

from langchain_core.tools import BaseTool, tool

from embeddings.constants import PathsStorage

if TYPE_CHECKING:
    from embeddings.embedder import Embedder


class AgentTools:
    """
    Class that operates creating tools
    """

    def __init__(self, embedder: "Embedder") -> None:
        """
        Initialize an instance of class

        Args:
            embedder (Embedder): Embedder whose children chunk storage is used
        """
        self.embedder = embedder
        self.parent_store_path = embedder._parent_store_path

    def create_tools(self) -> tuple[BaseTool, ...]:
        """
        Method that creates all tools for agent
        """

        @tool
        def search_child_chunks(query: str, limit: int) -> str:
            """
            Method that searches for relevant child chunks

            Args:
                query (str): Input query
                limit (int): Limit of chunks that can be returned

            Returns:
                str: Massive of chunks found
            """
            results = self.embedder.similarity_search_with_score_and_threshold(
                query, limit
            )
            if not results:
                return "NO RELEVANT CHUNKS FOUND"

            return "\n\n".join(
                [
                    f"Parent ID: {doc.metadata.get("parent_id", "NO PARENT ID")}\n"
                    f"Document ID: {doc.metadata.get("document_id", "NO DOCUMENT ID")}\n"
                    f"Chunk ID: {doc.metadata.get("chunk_id", "NO CHUNK ID")}\n"
                    f"Content: {doc.page_content.strip()}"
                    for doc, _ in results
                ]
            )

        @tool
        def retrieve_parent_chunks(parent_id: str, document_id: str) -> str:
            """
            Method that enables agent to retrieve parent chunks

            Args:
                parent_id (str): ID of parent chunk
                document_id (str): ID of document from where chunk was processed

            Returns:
                str: Massive of chunks found
            """
            parent_collection = PathsStorage.PARENT_COLLECTION.value
            if not parent_collection.exists():
                return "NO PARENT COLLECTION"
            with open(parent_collection, "r", encoding="utf-8") as file:
                data = json.load(file)

            for item in data:
                if (
                    str(item.get("document_id")) == document_id
                    and str(item.get("parent_id")) == parent_id
                ):
                    return (
                        f"Document ID: {item.get("document_id", "")}\n"
                        f"Parent ID: {item.get("parent_id", "")}\n"
                        f"Content: {item.get("parent_text", "").strip()}"
                    )

            return "PARENT_CHUNK_NOT_FOUND"

        return (search_child_chunks, retrieve_parent_chunks)
