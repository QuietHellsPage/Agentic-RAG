"""
Module to operate processing of raw text and saving it to vector db
"""

import json
from typing import Optional

import torch
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from langchain_qdrant.fastembed_sparse import FastEmbedSparse
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client import models as qmodels

# isort: off
from embeddings.constants import (
    TEXT_SPLITTER_SEPARATORS,
    LLMsAndVectorizersStorage,
    PathsStorage,
)
from embeddings.models import EmbedderConfig
from tools.tools import AgentTools

class EmbedSparse(FastEmbedSparse):
    """
    Make sparse embeddings
    """

    _model_name = LLMsAndVectorizersStorage.SPARSE_MODEL_NAME.value

    def __init__(self, device: str) -> None:
        """
        Initialize an instance of class

        Args:
            device (str): Device that operates embeddings processing
        """
        super().__init__()
        self._device: str
        if device is None:

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._device = device

        self.model = FastEmbedSparse(model_name=self._model_name, device=self._device)

    def embed_documents(self, texts: list[str]) -> list[qmodels.SparseVector]:
        """
        Method to embed documents

        Args:
            texts (list[str]): Texts that have to be processed in embeddings

        Returns:
            list[qmodels.SparseVector]: Massive of sparse vectors
        """
        embeddings = list(self.model.embed_documents(texts))
        result = []
        for embedding in embeddings:
            indices = embedding.indices
            values = embedding.values
            result.append(qmodels.SparseVector(indices=indices, values=values))
        return result

    def embed_query(self, text: str) -> list[qmodels.SparseVector]:
        """
        Method to embed input query

        Args:
            text (str): Input query

        Returns:
            list[qmodels.SparseVector]: Embedded query
        """
        embedding = self.model.embed_documents([text])[0]
        return qmodels.SparseVector(indices=embedding.indices, values=embedding.values)


class Embedder:  # pylint: disable=R0902
    """
    Instance for all operations with embeddings via Qdrant
    """

    _qdrant_path = PathsStorage.QDRANT_PATH.value
    _parent_store_path = PathsStorage.PARENT_CHUNKS_PATH.value
    _child_collection = PathsStorage.CHILD_COLLECTION.value
    _dense_model_name = LLMsAndVectorizersStorage.DENSE_MODEL_NAME.value
    _sparse_model_name = LLMsAndVectorizersStorage.SPARSE_MODEL_NAME.value

    def __init__(
        self,
        config: EmbedderConfig,
        device: Optional[str] = None,
        recreate_collection: bool = True,
    ) -> None:
        """
        Embedding model wrapper for Qdrant.

        Args:
            config (EmbedderConfig): Configuration for embedder
            device (Optional[str]): Device that operates embeddings processing
            recreate_collection (bool): Flag to recreate collection
        """
        if device is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._device = device

        self._parent_chunk_size = config.parent_chunk_size
        self._child_chunk_size = config.child_chunk_size
        self._parent_chunk_overlap = config.parent_chunk_overlap
        self._child_chunk_overlap = config.child_chunk_overlap

        self._qdrant_path.parent.mkdir(parents=True, exist_ok=True)
        self._parent_store_path.mkdir(parents=True, exist_ok=True)

        self._dense_embeddings = HuggingFaceEmbeddings(
            model_name=self._dense_model_name,
            model_kwargs={"device": self._device},
        )

        self._sparse_embeddings = EmbedSparse(device=self._device)

        self._client = QdrantClient(path=str(self._qdrant_path))
        self._child_vector_store = self._init_child_storage(recreate_collection)

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

        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._parent_chunk_size,
            chunk_overlap=self._parent_chunk_overlap,
            separators=TEXT_SPLITTER_SEPARATORS,
        )
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._child_chunk_size,
            chunk_overlap=self._child_chunk_overlap,
            separators=TEXT_SPLITTER_SEPARATORS,
        )

        docs = []
        chunk_storage = []

        for doc_idx, (document_id, document_chunks) in enumerate(
            zip(document_ids, [parent_splitter.split_text(text) for text in texts])
        ):
            for parent_id, parent_chunk in enumerate(document_chunks):

                chunk_storage.append(
                    {
                        "document_id": document_id,
                        "parent_id": parent_id,
                        "parent_text": parent_chunk,
                    }
                )

                for child_id, child_chunk in enumerate(
                    child_splitter.split_text(parent_chunk)
                ):
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

        self._child_vector_store.add_documents(docs)

        if (parent_store_file := PathsStorage.PARENT_COLLECTION.value).exists():
            parent_store_file.unlink()

        with open(parent_store_file, "w", encoding="utf-8") as file:
            json.dump(chunk_storage, file, indent=4, ensure_ascii=False)

    def _init_child_storage(
        self, recreate_collection: bool = True
    ) -> QdrantVectorStore:
        """
        Initializes Qdrant vector storage for hybrid similarity search.

        Args:
            recreate_collection (bool): Flag to recreate collection

        Returns:
            QdrantVectorStore: Storage of embeddings
        """
        embeddings_dimension = len(self._dense_embeddings.embed_query("test"))

        if recreate_collection and self._client.collection_exists(
            self._child_collection
        ):
            self._client.delete_collection(self._child_collection)

        if not self._client.collection_exists(self._child_collection):
            self._client.create_collection(
                collection_name=self._child_collection,
                vectors_config=qmodels.VectorParams(
                    size=embeddings_dimension, distance=qmodels.Distance.COSINE
                ),
                sparse_vectors_config={"sparse": qmodels.SparseVectorParams()},
            )
            print(f"Created collection: {self._child_collection}")
        else:
            print(f"Collection {self._child_collection} already exists")

        vector_store = QdrantVectorStore(
            client=self._client,
            collection_name=self._child_collection,
            embedding=self._dense_embeddings,
            sparse_embedding=self._sparse_embeddings,
            retrieval_mode=RetrievalMode.HYBRID,
            sparse_vector_name="sparse",
        )
        return vector_store

    def similarity_search(self, query: str, k: int = 5) -> list[Document]:
        """
        Performs hybrid similarity search on a given query.

        Args:
            query (str): Input query
            k (int): K nearest neighbours

        Returns:
            list[Document]: Massive of found documents
        """
        return self._child_vector_store.similarity_search(query=query, k=k)

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
        return self._child_vector_store.similarity_search_with_score(query=query, k=k)
    
    def get_tools(self) -> None:
        """
        Method that gets tools for agent by using self
        """
        tools = AgentTools(self)
        return tools.create_tools()


if __name__ == "__main__":
    embedder_config = EmbedderConfig(
        parent_chunk_size=1024,
        parent_chunk_overlap=256,
        child_chunk_size=512,
        child_chunk_overlap=128,
    )

    embedder = Embedder(config=embedder_config, recreate_collection=True)

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
        document_ids=[
            "doc1",
            "doc2",
            "doc3",
            "doc4",
            "doc5",
            "doc6",
            "doc7",
            "doc8",
            "doc9",
            "doc10",
            "doc11",
        ],  # Optional
    )

    results = embedder.similarity_search("football", k=2)
    for doc in results:
        print(f"Content: {doc.page_content}")
        print(f"Metadata: {doc.metadata}")
        print("-" * 50)
