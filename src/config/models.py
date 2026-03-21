"""
Models for processing embeddings
"""

from typing import Annotated, TypedDict

from pydantic import BaseModel, Field, model_validator


class EmbedderConfig(BaseModel):
    """
    Storage for embeddings processing
    """

    parent_chunk_size: Annotated[int, Field(default=4096, gt=0)]
    parent_chunk_overlap: Annotated[int, Field(default=400, gt=0)]
    child_chunk_size: Annotated[int, Field(default=1024, gt=0)]
    child_chunk_overlap: Annotated[int, Field(default=200, gt=0)]

    @model_validator(mode="after")
    def validate_fields(self) -> "EmbedderConfig":
        """
        Validate fields of model

        Returns:
            EmbedderConfig: Self
        """
        if self.child_chunk_overlap >= self.child_chunk_size:
            raise ValueError(
                "Child chunk overlap can not be bigger or the same as child chunks"
            )

        if self.parent_chunk_overlap >= self.parent_chunk_size:
            raise ValueError(
                "Parent chunk overlap can not be bigger or the same as parent chunks"
            )
        return self


class ParentChunk(BaseModel):
    """
    Parent chunk model
    """

    document_id: Annotated[str, Field(min_length=1)]
    parent_id: Annotated[int, Field(ge=0)]
    parent_text: Annotated[str, Field(min_length=1)]


class ChildChunkItem(BaseModel):
    """
    Single retrieved child chunk with metadata
    """

    parent_id: int
    document_id: str
    chunk_id: int
    content: str
    score: float


class SearchResult(BaseModel):
    """
    Result of child chunk search
    """

    chunks: Annotated[list[ChildChunkItem], Field(default_factory=list)]

    @property
    def found(self) -> bool:
        """
        Property that returns result of search

        Returns:
            bool: Result of search
        """
        return len(self.chunks) > 0


class ParentChunkResult(BaseModel):
    """
    Result of parent chunk retrieval
    """

    document_id: Annotated[str, Field(default="")] = ""
    parent_id: Annotated[int, Field(default=-1)] = -1
    content: Annotated[str, Field(default="")] = ""
    found: Annotated[bool, Field(default=False)] = False

    @classmethod
    def not_found(cls) -> "ParentChunkResult":
        """
        Returns result of search

        Returns:
            ParentChunkResult: Result of search
        """
        return cls(found=False)


class RAGState(TypedDict):
    """
    Shared state passed between all graph nodes
    """

    question: str
    original_question: str
    child_chunks: list[ChildChunkItem]
    parent_chunks: str
    answer: str
    reformulated: bool


class HashEntry(BaseModel):
    """
    Single retrieved hash entry
    """

    file_path: str
    algorithm: str
    hash: str = Field(alias="hash")
