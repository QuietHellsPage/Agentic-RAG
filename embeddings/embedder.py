"""
Module to operate processing of raw text and saving it to vector db
"""

import json
from pathlib import Path

import faiss
import torch
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from embeddings.models import EmbedderConfig


class Embedder:
    """
    Instance for all operations with embeddings
    """

    def __init__(self, config: EmbedderConfig, device: str | None = None):
        """
        Embedding model wrapper.
        :param parent_chunk_size: Number of tokens in a parent chunk.
        :param config: Configuration for embedder
        :param device: Device used for computation.
        """

        if device is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._device = device
        self._parent_chunk_size = config.parent_chunk_size
        self._child_chunk_size = config.child_chunk_size
        self._parent_chunk_overlap = config.parent_chunk_overlap
        self._child_chunk_overlap = config.child_chunk_overlap
        self._model = HuggingFaceEmbeddings(
            model_name="Qwen/Qwen3-Embedding-0.6B",  # TODO: larger model # pylint: disable=fixme
            model_kwargs={"device": self._device},
        )
        self._storage = self._init_storage()

    def add_documents(self, texts: list[str]):
        """
        Tokenizes, embeds texts and adds to FAISS.
        :param texts: Texts to process.
        :return:
        """
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

        for document_id, document_chunks in enumerate(documents_chunks):
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
                            child_chunk,
                            metadata={
                                "document_id": document_id,
                                "parent_id": parent_id,
                                "chunk_id": child_id,
                            },
                        )
                    )

        self._storage.add_documents(docs)
        with open(
            Path(__file__).parent.parent / "storage" / "parent_chunk_storage.json",
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(chunk_storage, file, indent=4)

    def embed(self, chunk: str):
        """
        Embed single chunk
        """
        return self._model.encode(chunk, device=self._device)  # type: ignore[attr-defined]

    def _init_storage(
        self,
        # TODO: 1024 is for 0.6B model, change according to the model we use later # pylint: disable=fixme
        model_dimension: int = 1024,
    ):
        """
        Initializes FAISS vector storage for similarity search.
        :param model_dimension: Embedding vector size.
        :return: Vector storage.
        """
        index = faiss.IndexFlatL2(model_dimension)
        vector_store = FAISS(
            embedding_function=self._model,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )
        return vector_store

    def similarity_search(self, query: str, k=5):
        """
        Performs similarity search on a given query.
        :param query:
        :param k: How many results should be returned.
        :return: List of most similar documents.
        """
        return self._storage.similarity_search(query, k)


# usage example for later:
if __name__ == "__main__":
    embedder_config = EmbedderConfig(
        parent_chunk_size=5,
        parent_chunk_overlap=3,
        child_chunk_size=3,
        child_chunk_overlap=1,
    )
    embedder = Embedder(embedder_config)
    embedder.add_documents(
        ["# this is a sample text.", "### subsection.", "the. quick? brown! fox jumps"]
    )

    print(embedder.similarity_search("fox"))
