from typing import Annotated, Optional

from langchain_community.graphs.graph_document import Node, Relationship
from pydantic import BaseModel, ConfigDict, Field

from src.config.constants import ENUM_VALUES


class Property(BaseModel):
    """
    Properties of graph
    """

    key: Annotated[str, Field(description=f"Available options are {ENUM_VALUES}")]
    value: str
    model_config = ConfigDict(str_min_length=1, extra=False)


class Node(Node):
    """
    Node instance to be stored in graph
    """

    id: Annotated[str, Field(description=f"Name or human-readable unique identifier")]
    label: Annotated[str, Field(description=f"Available options are {ENUM_VALUES}")]
    properties: Optional[list[Property]]
    model_config = ConfigDict(str_min_length=1, extra=False)


class Relationship(Relationship):
    """
    Relationship instance for binding nodes
    """

    source_node_id: str
    source_node_label: Annotated[
        str, Field(description=f"Available options are {ENUM_VALUES}")
    ]
    target_node_id: str
    target_node_label: Annotated[
        str, Field(description=f"Available options are {ENUM_VALUES}")
    ]
    type: Annotated[str, Field(description=f"Available options are {ENUM_VALUES}")]
    properties: Optional[list[Property]]
    model_config = ConfigDict(str_min_length=1, extra=False)
