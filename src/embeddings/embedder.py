"""
Module to operate processing of raw text and saving it to vector db
"""

import json
import os
from typing import Optional

import torch
from langchain_core.documents import Document
from langchain_core.tools import BaseTool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant.fastembed_sparse import FastEmbedSparse
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import SparseVector
from tqdm import tqdm

# isort: off
from src.config.constants import (
    TEXT_SPLITTER_SEPARATORS,
    LLMsAndVectorizersStorage,
    PathsStorage,
    EMBEDDINGS_DEVICE_ENV,
    LOGGER as logger,
)
from src.config.models import EmbedderConfig, ParentChunk
from src.helpers.create_vector_db import VectorDatabase
from src.tools.tools import AgentTools


class EmbedSparse(FastEmbedSparse):
    """
    Make sparse embeddings
    """

    _model_name = LLMsAndVectorizersStorage.SPARSE_MODEL_NAME.value

    def __init__(self, device: Optional[str] = None) -> None:
        """
        Initialize an instance of class

        Args:
            device (Optional[str]): Device that operates embeddings processing
        """
        if device is None:
            if not (env_device := os.getenv(EMBEDDINGS_DEVICE_ENV)):
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self._device = env_device
        else:
            self._device = device

        super().__init__(model_name=self._model_name, device=self._device)

    def __repr__(self) -> str:
        """
        Method that returns string representation of the class

        Returns:
            str: String representation
        """
        return f"{self.__class__.__name__!r}({self._device=!r})"

    def embed_documents(self, texts: list[str]) -> list[SparseVector]:
        """
        Method to embed documents

        Args:
            texts (list[str]): Texts that have to be processed in embeddings

        Returns:
            list[SparseVector]: Massive of sparse vectors
        """
        embeddings = super().embed_documents(texts)
        result = []
        for embedding in embeddings:
            indices = embedding.indices
            values = embedding.values
            result.append(SparseVector(indices=indices, values=values))
        return result

    def embed_query(self, text: str) -> SparseVector:
        """
        Method to embed input query

        Args:
            text (str): Input query

        Returns:
            SparseVector: Embedded query
        """
        embedding = super().embed_documents([text])[0]
        return SparseVector(indices=embedding.indices, values=embedding.values)


class Embedder:
    """
    Instance for all operations with embeddings via Qdrant
    """

    _parent_store_path = PathsStorage.PARENT_CHUNKS_PATH.value
    _dense_model_name = LLMsAndVectorizersStorage.DENSE_MODEL_NAME.value
    _sparse_model_name = LLMsAndVectorizersStorage.SPARSE_MODEL_NAME.value

    def __init__(
        self,
        config: EmbedderConfig,
        embeddings_model: type[HuggingFaceEmbeddings],
        sparse_model: type[EmbedSparse],
        vector_db: type[VectorDatabase],
        device: Optional[str] = None,
        recreate_collection: bool = True,
    ):
        """
        Embedding model wrapper for Qdrant.

        Args:
            config (EmbedderConfig): Configuration for embedder
            embeddings_model (type[HuggingFaceEmbeddings]): Model embedder
            sparse_model (type[EmbedSparse]): Sparse model
            vector_db (type[VectorDatabase]): Vector database
            device (Optional[str]): Device that operates embeddings processing
            recreate_collection (bool): Flag to recreate collection
        """
        if device is None:
            if not (env_device := os.getenv(EMBEDDINGS_DEVICE_ENV)):
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self._device = env_device
        else:
            self._device = device

        self._config = config
        self._parent_chunk_size = self._config.parent_chunk_size
        self._child_chunk_size = self._config.child_chunk_size
        self._parent_chunk_overlap = self._config.parent_chunk_overlap
        self._child_chunk_overlap = self._config.child_chunk_overlap
        self._recreate_collection = recreate_collection

        self._parent_store_path.mkdir(parents=True, exist_ok=True)

        self._dense_embeddings = embeddings_model(
            model_name=self._dense_model_name,
            model_kwargs={"device": self._device},
        )

        self._sparse_embeddings = sparse_model(device=self._device)

        self._vector_db = vector_db(
            dense_embeddings=self._dense_embeddings,
            sparse_embeddings=self._sparse_embeddings,
            client=QdrantClient,
            recreate_collection=self._recreate_collection,
        )

    def __repr__(self) -> str:
        """
        Method that returns string representation of the class

        Returns:
            str: String representation
        """
        return f"{self.__class__.__name__!r}({self._config=!r}, {self.
        _device=!r}, {self._recreate_collection=!r})"

    def add_documents(
        self, texts: list[str], document_ids: Optional[list[str]] = None
    ) -> None:
        """
        Tokenizes, embeds texts and adds to Qdrant.

        Args:
            texts (list[str]): Texts to be split into chunks
            document_ids (Optional[list[str]]): IDs of documents
        """
        if document_ids is None:
            document_ids = [str(i) for i in range(len(texts))]

        parent_splitter, child_splitter = self._init_text_splitters()

        docs = []
        chunk_storage = []

        for doc_idx, (document_id, document_chunks) in tqdm(
            enumerate(
                zip(document_ids, [parent_splitter.split_text(text) for text in texts])
            )
        ):
            for parent_id, parent_chunk in enumerate(document_chunks):
                logger.info("Processing parent chunk №%s", parent_id)
                chunk_storage.append(
                    ParentChunk(
                        document_id=document_id,
                        parent_id=parent_id,
                        parent_text=parent_chunk,
                    )
                )

                for child_id, child_chunk in enumerate(
                    child_splitter.split_text(parent_chunk)
                ):
                    logger.info("Processing child chunk №%s", child_id)
                    docs.append(
                        Document(
                            page_content=child_chunk,
                            metadata={
                                "document_id": document_id,
                                "parent_id": parent_id,
                                "chunk_id": child_id,
                                "document_idx": doc_idx,
                            },
                        )
                    )

        self._vector_db.add_documents(docs)

        logger.info("Saving parent chunks")
        self._save_parent_chunks(chunk_storage)

    def similarity_search_with_score(  # For future maybe
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
        return self._vector_db.similarity_search_with_score(query, k)

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
        return self._vector_db.similarity_search_with_score_and_threshold(
            query, k, threshold
        )

    def get_tools(self) -> tuple[BaseTool, ...]:
        """
        Method that gets tools for agent by using self

        Returns:
            tuple[BaseTool]: Tools
        """
        tools = AgentTools(self)
        return tools.create_tools()

    def close(self) -> None:
        """
        Method to close client
        """
        self._vector_db.close()

    def _init_text_splitters(self) -> tuple[RecursiveCharacterTextSplitter, ...]:
        """
        Method that initializes recursive text splitters

        Returns:
            tuple[RecursiveCharacterTextSplitter]: Text splitters
        """
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._child_chunk_size,
            chunk_overlap=self._child_chunk_overlap,
            separators=TEXT_SPLITTER_SEPARATORS,
        )

        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._parent_chunk_size,
            chunk_overlap=self._parent_chunk_overlap,
            separators=TEXT_SPLITTER_SEPARATORS,
        )

        return parent_splitter, child_splitter

    def _save_parent_chunks(self, chunks_storage: list[ParentChunk]) -> None:
        """
        Method that saves parent chunks to file

        Args:
            chunks_storage (list[ParentChunk]): Chunks
        """
        if (parent_store_file := PathsStorage.PARENT_COLLECTION.value).exists():
            parent_store_file.unlink()

        data = [item.model_dump() for item in chunks_storage]
        with open(parent_store_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
            f.write("\n")


if __name__ == "__main__":
    embedder_config = EmbedderConfig(
        parent_chunk_size=1024,
        parent_chunk_overlap=256,
        child_chunk_size=512,
        child_chunk_overlap=128,
    )

    embedder = Embedder(
        config=embedder_config,
        embeddings_model=HuggingFaceEmbeddings,
        sparse_model=EmbedSparse,
        vector_db=VectorDatabase,
        recreate_collection=True,
    )
    print(embedder)

    embedder.add_documents(
        texts=[
            "# This is a sample text.",
            "### Subsection.",
            "The quick brown fox jumps over the lazy dog.",
            "I believe I can fly",
            "I love animals",
            "My father loves my mother very much",
            "I know that my friend John is very lazy",
            "very bright yellow leafs and red blood",
            "I love to eat yellow snow",
            "He scores his first goal in professional league",
            "He is one of the best football players of all time. He is real GOAT!",
        ],
    )

    results = embedder.similarity_search_with_score_and_threshold(
        "autumn", k=2, threshold=0.2
    )
    for doc, score in results:
        print(f"Content: {doc.page_content}")
        print(f"Score: {score}")
        print("-" * 50)
    embedder.close()
