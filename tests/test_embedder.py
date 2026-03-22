"""
Tests for EmbedSparse and Embedder of src/embeddings/embedder.py.
"""

from unittest.mock import MagicMock, mock_open, patch

import pytest
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client.models import SparseVector

from src.config.models import (  # pylint: disable=import-error
    EmbedderConfig,
    ParentChunk,
)
from src.embeddings.embedder import (  # pylint: disable=import-error
    Embedder,
    EmbedSparse,
)
from src.vector_db.vector_db import VectorDatabase  # pylint: disable=import-error


# pylint: disable=protected-access
class TestEmbedSparse:
    """
    Class for testing EmbedSparse class.
    """

    @pytest.fixture
    def sparse_embedder(self):
        """Create EmbedSparse instance"""
        with patch("src.helpers.utils._choose_device", return_value="cpu"):
            return EmbedSparse(device="cpu")

    def test_init_and_repr_with_different_devices(self):
        """
        Test EmbedSparse initializes correctly with different device values
        and string representation is correct.
        """
        for device in ["cuda", "cpu", "mps"]:
            with patch("src.helpers.utils._choose_device", return_value=device):
                embed_sparse = EmbedSparse(device)
                assert embed_sparse._device == device
                assert str(embed_sparse) == f"'EmbedSparse'(self._device='{device}')"

        embed_sparse = EmbedSparse(None)
        assert embed_sparse._device == "cpu"
        assert str(embed_sparse) == "'EmbedSparse'(self._device='cpu')"

    def test_embed_documents_returns_sparse_vectors(self, sparse_embedder):
        """
        Test embed_documents returns a list of SparseVector instances.
        """
        texts = ["text1", "text2", "text3"]

        mock_embedding1 = MagicMock()
        mock_embedding1.indices = [0, 1, 2]
        mock_embedding1.values = [0.1, 0.2, 0.3]

        mock_embedding2 = MagicMock()
        mock_embedding2.indices = [1, 3]
        mock_embedding2.values = [0.5, 0.7]

        with patch(
            "langchain_qdrant.fastembed_sparse.FastEmbedSparse.embed_documents",
            return_value=[mock_embedding1, mock_embedding2, mock_embedding1],
        ):
            result = sparse_embedder.embed_documents(texts)

        assert len(result) == 3
        assert all(isinstance(vec, SparseVector) for vec in result)

    def test_embed_query_returns_sparse_vector(self, sparse_embedder):
        """
        Test embed_query returns SparseVector instance.
        """
        query = "test query"

        mock_embedding = MagicMock()
        mock_embedding.indices = [0, 1]
        mock_embedding.values = [0.9, 0.8]

        with patch(
            "langchain_qdrant.fastembed_sparse.FastEmbedSparse.embed_documents",
            return_value=[mock_embedding],
        ):
            result = sparse_embedder.embed_query(query)

        assert isinstance(result, SparseVector)
        assert result.indices == [0, 1]
        assert result.values == [0.9, 0.8]


class TestEmbedder:
    """
    Class for testing Embedder class.
    """

    @pytest.fixture
    def embedder_config(self):
        """
        Create a mock EmbedderConfig class.
        """
        mock_config = MagicMock(spec=EmbedderConfig)
        mock_config.parent_chunk_size = 40
        mock_config.parent_chunk_overlap = 10
        mock_config.child_chunk_size = 10
        mock_config.child_chunk_overlap = 2
        return mock_config

    @pytest.fixture
    def mock_embeddings_model(self):
        """
        Create a mock HuggingFaceEmbeddings class.
        """
        return MagicMock(spec=HuggingFaceEmbeddings)

    @pytest.fixture
    def mock_sparse_model(self):
        """
        Create a mock EmbedSparse class.
        """
        return MagicMock(spec=EmbedSparse)

    @pytest.fixture
    def mock_vector_db(self):
        """
        Create a mock VectorDatabase class.
        """
        return MagicMock(spec=VectorDatabase)

    @pytest.fixture
    def embedder(
        self,
        embedder_config,
        mock_embeddings_model,
        mock_sparse_model,
        mock_vector_db,
    ):
        """
        Create Embedder instance with mocked dependencies.
        """
        with patch("src.helpers.utils._choose_device", return_value="cpu"):
            with patch("pathlib.Path.mkdir"):
                embedder = Embedder(
                    config=embedder_config,
                    embeddings_model=mock_embeddings_model,
                    sparse_model=mock_sparse_model,
                    vector_db=mock_vector_db,
                    device="cpu",
                    recreate_collection=True,
                )

        return embedder

    def test_init_stores_config(
        self, embedder_config, mock_embeddings_model, mock_sparse_model, mock_vector_db
    ):
        """
        Test if configuration is stored.
        """
        with patch("src.helpers.utils._choose_device", return_value="cpu"):
            with patch("pathlib.Path.mkdir"):
                embedder = Embedder(
                    config=embedder_config,
                    embeddings_model=mock_embeddings_model,
                    sparse_model=mock_sparse_model,
                    vector_db=mock_vector_db,
                    device="cpu",
                )

        assert embedder._config == embedder_config
        assert embedder_config.parent_chunk_size == 40
        assert embedder_config.parent_chunk_overlap == 10
        assert embedder_config.child_chunk_size == 10
        assert embedder_config.child_chunk_overlap == 2

    def test_add_documents_generates_parent_and_child_chunks(self, embedder):
        """
        Test add_documents generates parent and child chunks.
        """
        texts = ["test text"]

        with patch.object(embedder, "_init_text_splitters") as mock_splitters:
            mock_parent_splitter = MagicMock()
            mock_child_splitter = MagicMock()
            mock_parent_splitter.split_text.return_value = ["parent_chunk"]
            mock_child_splitter.split_text.return_value = [
                "child_chunk1",
                "child_chunk2",
            ]
            mock_splitters.return_value = (mock_parent_splitter, mock_child_splitter)

            with patch.object(embedder, "_generate_parent_chunks") as mock_parent_gen:
                with patch.object(embedder, "_generate_child_chunks") as mock_child_gen:
                    with patch.object(embedder, "_save_parent_chunks"):
                        mock_parent_gen.return_value = [MagicMock(spec=ParentChunk)]
                        mock_child_gen.return_value = [
                            MagicMock(spec=Document),
                            MagicMock(spec=Document),
                        ]

                        embedder.add_documents(texts)

        assert mock_parent_gen.called
        assert mock_child_gen.called

    def test_add_documents_saves_to_vector_db(self, embedder):
        """
        Test add_documents adds documents to vector database.
        """
        texts = ["text1"]

        with patch.object(embedder, "_init_text_splitters") as mock_splitters:
            mock_parent_splitter = MagicMock()
            mock_child_splitter = MagicMock()
            mock_parent_splitter.split_text.return_value = ["chunk1"]
            mock_child_splitter.split_text.return_value = ["child1"]
            mock_splitters.return_value = (mock_parent_splitter, mock_child_splitter)

            with patch.object(embedder, "_generate_parent_chunks") as mock_parent_gen:
                with patch.object(embedder, "_generate_child_chunks") as mock_child_gen:
                    with patch.object(embedder, "_save_parent_chunks"):
                        mock_parent_gen.return_value = [MagicMock(spec=ParentChunk)]
                        mock_doc = MagicMock(spec=Document)
                        mock_child_gen.return_value = [mock_doc]

                        embedder.add_documents(texts)

        embedder._vector_db.add_documents.assert_called_once()
        call_args = embedder._vector_db.add_documents.call_args[0][0]
        assert mock_doc in call_args

    def test_save_parent_chunks_converts_to_dict(self, embedder):
        """
        Test save_parent_chunks parent chunks are converted to dictionaries.
        """
        mock_chunk1 = MagicMock(spec=ParentChunk)
        mock_chunk2 = MagicMock(spec=ParentChunk)
        mock_chunk1.model_dump.return_value = {"id": "1", "text": "text1"}
        mock_chunk2.model_dump.return_value = {"id": "2", "text": "text2"}

        chunks = [mock_chunk1, mock_chunk2]

        with patch("pathlib.Path.exists", return_value=False):
            with patch("builtins.open", mock_open()):
                with patch("json.dump"):
                    embedder._save_parent_chunks(chunks)

        mock_chunk1.model_dump.assert_called_once()
        mock_chunk2.model_dump.assert_called_once()
