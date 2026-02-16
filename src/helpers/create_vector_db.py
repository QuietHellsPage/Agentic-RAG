"""
Module that contains Vector database
"""

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from langchain_qdrant.fastembed_sparse import FastEmbedSparse
from qdrant_client import QdrantClient

# isort: off
from qdrant_client.models import (
    Distance,
    SparseVectorParams,
    VectorParams,
)

from src.config.constants import PathsStorage, EMBEDDINGS_SIZE, LOGGER as logger


class VectorDatabase:
    """
    Class that operates Vector Database
    """

    _qdrant_path = PathsStorage.QDRANT_PATH.value
    _child_collection = PathsStorage.CHILD_COLLECTION.value

    def __init__(
        self,
        dense_embeddings: HuggingFaceEmbeddings,
        sparse_embeddings: FastEmbedSparse,
        client: type[QdrantClient],
        recreate_collection: bool = True,
    ) -> None:
        """
        Initialize an instance of class

        Args:
            dense_embeddings (HuggingFaceEmbeddings): Dense embeddings
            sparse_embeddings (FastEmbedSparse): Sparse embeddings
            client (type[QdrantClient]): Client to operate work with vector database
            recreate_collection (bool): Flag to recreate collection
        """
        self._dense_embeddings = dense_embeddings
        self._sparse_embeddings = sparse_embeddings
        self._recreate_collection = recreate_collection
        self._client = client(path=str(self._qdrant_path))
        self._init_child_storage()

    def __repr__(self) -> str:
        """
        Method that returns string representation of the class

        Returns:
            str: String representation
        """
        return f"{self.__class__.__name__!r}({self._dense_embeddings=!r}, {self.
        _sparse_embeddings=!r}, {self._recreate_collection=!r})"

    def close(self) -> None:
        """
        Method to close client
        """
        if hasattr(self, "_client") and self._client:
            self._client.close()

    def add_documents(self, documents: list[Document]) -> list[str]:
        """
        Method that adds documents to vector db

        Args:
            documents (list[Document]): Documents to be added

        Returns:
            list[str]: IDs of added documents
        """
        return self._vector_store.add_documents(documents)

    def similarity_search_with_score(
        self, query: str, k: int = 5
    ) -> list[tuple[Document, float]]:
        """
        Performs hybrid similarity search with scores.

        Args:
            query (str): Input query
            k (int): K nearest neighbours

        Returns:
            list[tuple[Document, float]]: Massive of found documents with scores
        """
        return self._vector_store.similarity_search_with_score(query, k=k)

    def similarity_search_with_score_and_threshold(
        self, query: str, k: int = 4, threshold: float = 0.6
    ) -> list[tuple[Document, float]]:
        """
        Performs hybrid similarity search with scores and threshold.

        Args:
            query (str): Input query
            k (int): K nearest neighbours
            threshold (float): Minimum similarity score

        Returns:
            list[tuple[Document, float]]: Filtered documents with scores
        """
        result = self.similarity_search_with_score(query, k)
        filtered_results = [(doc, score) for doc, score in result if score > threshold]
        return filtered_results

    def _init_child_storage(self) -> None:
        """
        Method that initializes Qdrant vector storage for hybrid similarity search.
        """

        if self._recreate_collection and self._client.collection_exists(
            self._child_collection
        ):
            self._client.delete_collection(self._child_collection)

        if not self._client.collection_exists(self._child_collection):
            self._client.create_collection(
                collection_name=self._child_collection,
                vectors_config=VectorParams(
                    size=EMBEDDINGS_SIZE, distance=Distance.COSINE
                ),
                sparse_vectors_config={"sparse": SparseVectorParams()},
            )
            logger.info("Created collection: %s", self._child_collection)
        else:
            logger.info("Collection %s already exists", self._child_collection)

        self._vector_store = self._init_vector_store()

    def _init_vector_store(self) -> QdrantVectorStore:
        """
        Method that initializes vector store

        Returns:
            QdrantVectorStore: Vector store
        """
        vector_store = QdrantVectorStore(
            client=self._client,
            collection_name=self._child_collection,
            embedding=self._dense_embeddings,
            sparse_embedding=self._sparse_embeddings,
            retrieval_mode=RetrievalMode.HYBRID,
            sparse_vector_name="sparse",
        )
        return vector_store
