from langchain_neo4j import Neo4jGraph

from src.config.constants import GraphInitializerStorage


class Graph:
    """
    Wrapper for Neo4jGraph
    """

    def __init__(self) -> None:
        """
        Initialize an instance of class
        """
        self._graph = Neo4jGraph(
            url=GraphInitializerStorage.URL.value,
            username=GraphInitializerStorage.USERNAME.value,
            password=GraphInitializerStorage.PASSWORD.value,
            refresh_schema=GraphInitializerStorage.REFRESH_SCHEMA.value,
        )

    def clean_graph(self) -> None:
        """
        Method that cleans graph
        """
        query = """
        MATCH (n)
        DETACH DELETE n
        """
        self._graph.query(query)

    def add_data(self, data: str) -> None:
        """
        Method that adds data to graph

        Args:
            data (str): Data to be added to graph
        """
        self._graph.add_graph_documents(data, include_source=True)
