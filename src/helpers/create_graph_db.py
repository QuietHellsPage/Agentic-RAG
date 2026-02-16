"""
Create graph db and process text
"""

from langchain_core.documents import Document
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_neo4j import Neo4jGraph

# isort: off
from src.config.constants import GraphAllowedConstants, LLMsAndVectorizersStorage
from src.graph_db.graph import Graph

if __name__ == "__main__":
    TEXT = """
    Marie Curie, 7 November 1867 – 4 July 1934, was a Polish and naturalised-French physicist and chemist who conducted pioneering research on radioactivity.
    She was the first woman to win a Nobel Prize, the first person to win a Nobel Prize twice, and the only person to win a Nobel Prize in two scientific fields.
    Her husband, Pierre Curie, was a co-winner of her first Nobel Prize, making them the first-ever married couple to win the Nobel Prize and launching the Curie family legacy of five Nobel Prizes.
    She was, in 1906, the first woman to become a professor at the University of Paris.
    Also, Robin Williams!
    """

    documents = [Document(page_content=TEXT)]

    graph = Graph(Neo4jGraph)
    graph.clean_graph()

    graph_transformer = LLMGraphTransformer(
        llm=LLMsAndVectorizersStorage.GRAPH_LLM.value,
        allowed_nodes=GraphAllowedConstants.ALLOWED_NODES.value,
        allowed_relationships=GraphAllowedConstants.ALLOWED_RELATIONSHIPS.value,
        node_properties=GraphAllowedConstants.NODE_PROPERTIES.value,
        relationship_properties=GraphAllowedConstants.RELATIONSHIPS_PROPERTIES.value,
        strict_mode=GraphAllowedConstants.STRICT_MODE.value,
    )

    data = graph_transformer.convert_to_graph_documents(documents=documents)
    print(data)
    graph.add_data(data=data)  # type: ignore[arg-type]
