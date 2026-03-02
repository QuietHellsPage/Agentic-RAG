"""
Models for processing embeddings
"""

from typing import Annotated

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


class AgentConfig:
    """
    Configuration for RAG Agent
    """
    def __init__(
        self,
        llm_model_name: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        max_tokens: int = 500,
        retrieval_k: int = 4,
        similarity_threshold: float = 0.6,
        use_tools: bool = True,
        max_iterations: int = 5,
        use_full_context: bool = False,
    ):
        """
               Initialize AgentConfig

               Args:
                   llm_model_name (str): Name of the LLM model
                   temperature (float): Temperature for generation
                   max_tokens (int): Maximum tokens in response
                   retrieval_k (int): Number of documents to retrieve
                   similarity_threshold (float): Threshold for similarity search
                   use_tools (bool): Whether to use tools
                   verbose (bool): Whether to print agent steps
                   max_iterations (int): Maximum number of agent iterations
                   use_full_context (bool): Whether to retrieve full parent chunks
               """
        self.llm_model_name = llm_model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.retrieval_k = retrieval_k
        self.similarity_threshold = similarity_threshold
        self.use_tools = use_tools
        self.max_iterations = max_iterations
        self.use_full_context = use_full_context

