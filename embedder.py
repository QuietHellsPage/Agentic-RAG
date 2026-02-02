import faiss
import torch
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class Embedder:
    def __init__(
            self,
            parent_chunk_size: int = 4096,
            parent_chunk_overlap: int = 400,
            child_chunk_size: int = 1024,
            child_chunk_overlap: int = 200,
            device: str | None = None
    ):
        """
        Embedding model wrapper.
        :param parent_chunk_size: Number of tokens in a **parent** chunk.
        :param parent_chunk_overlap: Number of shared tokens by neighbouring **parent** chunks.
                Best practice is 10-20% of parent chunk size.
        :param child_chunk_size: Number of tokens in a **child** chunk.
        :param child_chunk_overlap: Number of shared tokens by neighbouring **child** chunks.
                Best practice is 10-20% of child chunk size.
        :param device: Device used for computation.
        """
        if child_chunk_overlap >= child_chunk_size:
            raise ValueError(
                "Child chunk overlap can not be bigger or the same as child chunks")
        if parent_chunk_overlap >= parent_chunk_size:
            raise ValueError(
                "Parent chunk overlap can not be bigger or the same as parent chunks")

        if device is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._device = device

        self._parent_chunk_size = parent_chunk_size
        self._child_chunk_size = child_chunk_size
        self._parent_chunk_overlap = parent_chunk_overlap
        self._child_chunk_overlap = child_chunk_overlap
        self._model = HuggingFaceEmbeddings(
            model_name="Qwen/Qwen3-Embedding-0.6B",  # TODO: larger model
            model_kwargs={"device": self._device}
        )
        self._storage = self._init_storage()

    def add_documents(
            self,
            texts: list[str]
    ):
        """
        Tokenizes, embeds texts and adds to FAISS.
        :param texts: Texts to process.
        :return:
        """
        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._parent_chunk_size,
            chunk_overlap=self._parent_chunk_overlap,
            separators=["\n#", "\n##", "\n###", "\n####", "\n#####", "\n######",
                        "\n\n", "\n", ".", "?", "!", " ", ""]
        )
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._child_chunk_size,
            chunk_overlap=self._child_chunk_overlap,
            separators=["\n#", "\n##", "\n###", "\n####", "\n#####", "\n######",
                        "\n\n", "\n", ".", "?", "!", " ", ""]
        )

        documents_chunks = [parent_splitter.split_text(text) for text in texts]
        docs = []

        for document_id, document_chunks in enumerate(documents_chunks):
            for parent_id, parent_chunk in enumerate(document_chunks):
                children_chunks = child_splitter.split_text(parent_chunk)
                for child_id, child_chunk in enumerate(children_chunks):
                    metadata = {
                        "document_id": document_id,
                        "parent_id": parent_id,
                        "chunk_id": child_id
                    }
                    child_doc = Document(child_chunk, metadata=metadata)
                    docs.append(child_doc)

        self._storage.add_documents(docs)

    def embed(self, chunk: str):
        return self._model.encode(chunk, device=self._device)

    def _init_storage(
            self,
            # TODO: 1024 is for 0.6B model, change according to the model we use later
            model_dimension: int = 1024
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

    def similarity_search(
            self,
            query: str,
            k=5
    ):
        """
        Performs similarity search on a given query.
        :param query:
        :param k: How many results should be returned.
        :return: List of most similar documents.
        """
        return self._storage.similarity_search(query, k)


# usage example for later:
"""em = Embedder(parent_chunk_size=5, parent_chunk_overlap=3, child_chunk_size=3,
              child_chunk_overlap=1)
em.add_documents(["# this is a sample text."
                  "### subsection.",
                  "the. quick? brown! fox jumps"])

print(em.similarity_search("fox"))"""
