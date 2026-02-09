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
    def validate_fields(self):
        """
        Validate fields of model
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
