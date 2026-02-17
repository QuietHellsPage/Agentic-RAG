"""
Graph class instance
"""

from langchain_neo4j import Neo4jGraph
from langchain_neo4j.graphs.graph_document import GraphDocument

from src.config.constants import GraphInitializerStorage


class Graph:
    """
    Wrapper for Neo4jGraph
    """

    def __init__(self, graph_instance: type[Neo4jGraph]) -> None:
        """
        Initialize an instance of class

        Args:
            graph_instance (type[Neo4jGraph]): Graph db
        """
        self._graph = graph_instance(
            url=GraphInitializerStorage.URL.value,
            username=GraphInitializerStorage.USERNAME.value,
            password=GraphInitializerStorage.PASSWORD.value,
            refresh_schema=GraphInitializerStorage.REFRESH_SCHEMA.value,
        )

    def __repr__(self) -> str:
        """
        Method that returns string representation of the class

        Returns:
            str: String representation
        """
        return f"{self.__class__.__name__!r}(self._graph={self._graph.__class__.__name__!r})"

    def clean_graph(self) -> None:
        """
        Method that cleans graph
        """
        query = """
        MATCH (n)
        DETACH DELETE n
        """
        self._graph.query(query)

    def add_data(self, data: list[GraphDocument]) -> None:
        """
        Method that adds data to graph

        Args:
            data (list[GraphDocument]): Data to be added to graph
        """
        self._graph.add_graph_documents(data, include_source=True)
