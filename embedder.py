import faiss
import torch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer


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
        self._model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")  # TODO: larger model

    def tokenize(
            self,
            texts: list[str]
    ):
        """
        Splits texts into chunks.
        :param texts:
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

        # a list[list[str]], where each list[str] is a document, and str is a chunk
        documents_chunks = [parent_splitter.split_text(text) for text in texts]
        child_chunks = []

        for document_id, document_chunks in enumerate(documents_chunks):
            for parent_id, parent_chunk in enumerate(document_chunks):
                children_chunks = child_splitter.split_text(parent_chunk)
                for child_id, child_chunk in enumerate(children_chunks):
                    child_embedding = self._embed_document(child_chunk)
                    metadata = {
                        "document_id": document_id,
                        "parent_id": parent_id,
                        "chunk_id": child_id,
                        "embedding": child_embedding
                    }

    def _embed_document(self, chunk: str):
        return self._model.encode_document(chunk, device=self._device)

    def embed_query(self, query: str):
        return self._model.encode_query(query, device=self._device)


em = Embedder(parent_chunk_size=5, parent_chunk_overlap=3, child_chunk_size=3,
              child_chunk_overlap=1)
print(em.tokenize(["# this is a sample text."
                   "### subsection.",
                   "the. quick? brown! fox jumps"]))
