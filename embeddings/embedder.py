"""
Module to operate processing of raw text and saving it to vector db
"""

import json
from pathlib import Path
from typing import Optional

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from langchain_qdrant.fastembed_sparse import FastEmbedSparse
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client import models as qmodels

from embeddings.models import EmbedderConfig


class EmbedSparse:
    """
    Make sparse embeddings
    """

    def __init__(self, model_name: str = "Qdrant/bm25", device: str = None):
        """
        Initialize an instance of class
        """
        if device is None:
            import torch

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._device = device

        self.model = FastEmbedSparse(model_name=model_name, device=self._device)

    def embed_documents(self, texts):
        """
        Method to embed documents
        """
        embeddings = list(self.model.embed_documents(texts))
        result = []
        for embedding in embeddings:
            indices = embedding.indices
            values = embedding.values
            result.append(qmodels.SparseVector(indices=indices, values=values))
        return result

    def embed_query(self, text):
        """
        Method to embed input query
        """
        embedding = self.model.embed_documents([text])[0]
        return qmodels.SparseVector(indices=embedding.indices, values=embedding.values)


class EmbedSparse:
    """
    Make sparse embeddings
    """

    def __init__(self, model_name: str = "Qdrant/bm25", device: str = None):
        """
        Initialize an instance of class
        """
        if device is None:
            import torch

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._device = device

        self.model = FastEmbedSparse(model_name=model_name, device=self._device)

    def embed_documents(self, texts):
        """
        Method to embed documents
        """
        embeddings = list(self.model.embed_documents(texts))
        result = []
        for embedding in embeddings:
            indices = embedding.indices
            values = embedding.values
            result.append(qmodels.SparseVector(indices=indices, values=values))
        return result

    def embed_query(self, text):
        """
        Method to embed input query
        """
        embedding = self.model.embed_documents([text])[0]
        return qmodels.SparseVector(indices=embedding.indices, values=embedding.values)


class Embedder:
    """
    Instance for all operations with embeddings via Qdrant
    """

    def __init__(
        self,
        config: EmbedderConfig,
        device: Optional[str] = None,
        qdrant_path: str = None,
        parent_store_path: str = "data/chunks",
        child_collection: str = "document_child_chunks",
        dense_model_name: str = "Qwen/Qwen3-Embedding-0.6B",
        sparse_model_name: str = "Qdrant/bm25",
        recreate_collection: bool = False,
    ):
        """
        Embedding model wrapper for Qdrant.
        """
        if device is None:
            import torch

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._device = device

        self._parent_chunk_size = config.parent_chunk_size
        self._child_chunk_size = config.child_chunk_size
        self._parent_chunk_overlap = config.parent_chunk_overlap
        self._child_chunk_overlap = config.child_chunk_overlap
        self._qdrant_path = Path(qdrant_path)
        self._parent_store_path = Path(parent_store_path)
        self._child_collection = child_collection

        self._qdrant_path.parent.mkdir(parents=True, exist_ok=True)
        self._parent_store_path.mkdir(parents=True, exist_ok=True)

        self._dense_embeddings = HuggingFaceEmbeddings(
            model_name=dense_model_name,
            model_kwargs={"device": self._device},
        )

        self._sparse_embeddings = EmbedSparse(
            model_name=sparse_model_name, device=self._device
        )

        self._client = QdrantClient(path=str(self._qdrant_path))
        self._child_vector_store = self._init_child_storage(recreate_collection)

    def add_documents(self, texts: list[str], document_ids: Optional[list[str]] = None):
        """
        Tokenizes, embeds texts and adds to Qdrant.
        """
        if document_ids is None:
            document_ids = [str(i) for i in range(len(texts))]

        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._parent_chunk_size,
            chunk_overlap=self._parent_chunk_overlap,
            separators=[
                "\n#",
                "\n##",
                "\n###",
                "\n####",
                "\n#####",
                "\n######",
                "\n\n",
                "\n",
                ".",
                "?",
                "!",
                " ",
                "",
            ],
        )
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._child_chunk_size,
            chunk_overlap=self._child_chunk_overlap,
            separators=[
                "\n#",
                "\n##",
                "\n###",
                "\n####",
                "\n#####",
                "\n######",
                "\n\n",
                "\n",
                ".",
                "?",
                "!",
                " ",
                "",
            ],
        )

        documents_chunks = [parent_splitter.split_text(text) for text in texts]
        docs = []
        chunk_storage = []

        for doc_idx, (document_id, document_chunks) in enumerate(
            zip(document_ids, documents_chunks)
        ):
            for parent_id, parent_chunk in enumerate(document_chunks):
                children_chunks = child_splitter.split_text(parent_chunk)

                chunk_storage.append(
                    {
                        "document_id": document_id,
                        "parent_id": parent_id,
                        "parent_text": parent_chunk,
                    }
                )

                for child_id, child_chunk in enumerate(children_chunks):
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

        parent_store_file = self._parent_store_path / "parent_chunks_storage.json"
        if parent_store_file.exists():
            with open(parent_store_file, "r", encoding="utf-8") as file:
                existing_data = json.load(file)
            existing_data.extend(chunk_storage)
            chunk_storage = existing_data

        with open(parent_store_file, "w", encoding="utf-8") as file:
            json.dump(chunk_storage, file, indent=4, ensure_ascii=False)

    def embed(self, chunk: str):
        """
        Embed single chunk (dense only)
        """
        return self._dense_embeddings.embed_query(chunk)

    def _init_child_storage(self, recreate_collection: bool = False):
        """
        Initializes Qdrant vector storage for hybrid similarity search.
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

    def similarity_search(
        self, query: str, k: int = 5, filter_dict: Optional[dict] = None
    ):
        """
        Performs hybrid similarity search on a given query.
        """
        return self._child_vector_store.similarity_search(
            query=query, k=k, filter=filter_dict
        )

    def similarity_search_with_score(
        self, query: str, k: int = 5, filter_dict: Optional[dict] = None
    ):
        """
        Performs hybrid similarity search with scores.
        """
        return self._child_vector_store.similarity_search_with_score(
            query=query, k=k, filter=filter_dict
        )

    def get_parent_chunks(self, document_id: Optional[str] = None):
        """
        Retrieve parent chunks from storage.
        """
        parent_store_file = self._parent_store_path / "parent_chunks_storage.json"
        if not parent_store_file.exists():
            return []

        with open(parent_store_file, "r", encoding="utf-8") as file:
            data = json.load(file)

        if document_id:
            return [item for item in data if item["document_id"] == document_id]
        return data


if __name__ == "__main__":
    embedder_config = EmbedderConfig(
        parent_chunk_size=1024,
        parent_chunk_overlap=256,
        child_chunk_size=512,
        child_chunk_overlap=128,
    )

    embedder = Embedder(
        config=embedder_config,
        qdrant_path="data/qdrant_db",
        parent_store_path="data/chunks",
        recreate_collection=True,
    )

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
        ],  # Optional
    )

    results = embedder.similarity_search("autumn", k=2)
    for doc in results:
        print(f"Content: {doc.page_content}")
        print(f"Metadata: {doc.metadata}")
        print("-" * 50)
