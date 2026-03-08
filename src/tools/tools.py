"""
Tools for agent
"""

import json
from typing import TYPE_CHECKING

from langchain_core.tools import BaseTool, tool

from src.config.constants import PathsStorage
from src.config.models import ChildChunkItem, ParentChunkResult, SearchResult

if TYPE_CHECKING:
    from src.embeddings.embedder import Embedder


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

    def __repr__(self) -> str:
        """
        Method that returns string representation of the class

        Returns:
            str: String representation
        """
        return f"{self.__class__.__name__!r}({self.embedder=!r}, {self.parent_store_path=!r})"

    def create_tools(self) -> tuple[BaseTool, ...]:
        """
        Method that creates all tools for agent

        Returns:
            tuple[BaseTool]: Tools for agent
        """

        @tool
        def search_child_chunks(query: str, limit: int) -> SearchResult:
            """
            Method that searches for relevant child chunks

            Args:
                query (str): Natural language query
                limit (int): Maximum number of chunks to return

            Returns:
                SearchResult: Structured search results
            """
            raw = self.embedder.similarity_search_with_score_and_threshold(query, limit)
            chunks = [
                ChildChunkItem(
                    parent_id=doc.metadata.get("parent_id", -1),
                    document_id=doc.metadata.get("document_id", ""),
                    chunk_id=doc.metadata.get("chunk_id", -1),
                    content=doc.page_content.strip(),
                    score=score,
                )
                for doc, score in raw
            ]
            return SearchResult(chunks=chunks)

        @tool
        def retrieve_parent_chunks(
            parent_id: int, document_id: str
        ) -> ParentChunkResult:
            """
            Method that enables agent to retrieve parent chunks

            Args:
                parent_id (int): ID of the parent chunk
                document_id (str): ID of the source document

            Returns:
                ParentChunkResult: Structured parent chunk or not-found sentinel
            """
            parent_collection = PathsStorage.PARENT_COLLECTION.value
            if not parent_collection.exists():
                return ParentChunkResult.not_found()

            with open(parent_collection, "r", encoding="utf-8") as f:
                data = json.load(f)

            for item in data:
                if (
                    str(item.get("document_id")) == str(document_id)
                    and int(item.get("parent_id", -1)) == parent_id
                ):
                    return ParentChunkResult(
                        document_id=item["document_id"],
                        parent_id=item["parent_id"],
                        content=item.get("parent_text", "").strip(),
                        found=True,
                    )

            return ParentChunkResult.not_found()

        return (search_child_chunks, retrieve_parent_chunks)
