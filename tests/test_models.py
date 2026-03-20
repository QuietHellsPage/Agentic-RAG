"""
Tests for Models
"""

import pytest
from pydantic import ValidationError

from src.config.models import (  # pylint: disable=import-error
    ChildChunkItem,
    EmbedderConfig,
    ParentChunk,
    ParentChunkResult,
    SearchResult,
)


class TestEmbedderConfig:
    """
    Class with tests of EmbedderConfig
    """

    def test_default_values(self) -> None:
        """
        Test default values
        """

        config = EmbedderConfig()

        assert config.parent_chunk_size == 4096
        assert config.parent_chunk_overlap == 400
        assert config.child_chunk_size == 1024
        assert config.child_chunk_overlap == 200

    def test_custom_values(self) -> None:
        """
        Test custom values
        """

        config = EmbedderConfig(
            parent_chunk_size=2048,
            parent_chunk_overlap=200,
            child_chunk_size=512,
            child_chunk_overlap=100,
        )

        assert config.parent_chunk_size == 2048
        assert config.parent_chunk_overlap == 200
        assert config.child_chunk_size == 512
        assert config.child_chunk_overlap == 100

    def test_child_overlap_equal_to_size_raises(self) -> None:
        """
        Test overlap #1
        """

        with pytest.raises(ValidationError, match="Child chunk overlap"):
            EmbedderConfig(child_chunk_size=512, child_chunk_overlap=512)

    def test_child_overlap_greater_than_size_raises(self) -> None:
        """
        Test overlap #2
        """

        with pytest.raises(ValidationError, match="Child chunk overlap"):
            EmbedderConfig(child_chunk_size=512, child_chunk_overlap=1024)

    def test_parent_overlap_equal_to_size_raises(self) -> None:
        """
        Test overlap #3
        """

        with pytest.raises(ValidationError, match="Parent chunk overlap"):
            EmbedderConfig(parent_chunk_size=1024, parent_chunk_overlap=1024)

    def test_parent_overlap_greater_than_size_raises(self) -> None:
        """
        Test overlap #4
        """

        with pytest.raises(ValidationError, match="Parent chunk overlap"):
            EmbedderConfig(parent_chunk_size=1024, parent_chunk_overlap=2048)

    def test_zero_chunk_size_raises(self) -> None:
        """
        Test 0 size
        """

        with pytest.raises(ValidationError):
            EmbedderConfig(child_chunk_size=0)

    def test_negative_chunk_size_raises(self) -> None:
        """
        Test negative size
        """

        with pytest.raises(ValidationError):
            EmbedderConfig(parent_chunk_size=-1)

    def test_valid_boundary_overlap(self) -> None:
        """
        Test boundary overlap
        """

        cfg = EmbedderConfig(child_chunk_size=100, child_chunk_overlap=99)
        assert cfg.child_chunk_overlap == 99


class TestParentChunk:
    """
    Class with tests for ParentChunk
    """

    def test_valid_creation(self) -> None:
        """
        Test valid creation
        """

        config = ParentChunk(
            document_id="first", parent_id=1, parent_text="Hello Wolrd!"
        )

        assert config.document_id == "first"
        assert config.parent_id == 1
        assert config.parent_text == "Hello Wolrd!"

    def test_empty_doc_id_raises(self) -> None:
        """
        Test empty doc raises
        """

        with pytest.raises(ValidationError):
            ParentChunk(document_id="", parent_id=1, parent_text="Hello Wolrd!")

    def test_neg_parent_id_raises(self) -> None:
        """
        Test negative parent id
        """

        with pytest.raises(ValidationError):
            ParentChunk(document_id="first", parent_id=-1, parent_text="Hello Wolrd!")

    def test_empty_parent_text_raises(self) -> None:
        """
        Test empty parent text
        """

        with pytest.raises(ValidationError):
            ParentChunk(document_id="first", parent_id=1, parent_text="")


class TestChildChunkItem:
    """
    Tests for ChildChunkItem
    """

    def _make_item(self, **kwargs) -> ChildChunkItem:
        """
        Make item

        Returns:
            ChildChunkItem: Item
        """

        defaults = {
            "parent_id": 42,
            "document_id": "first",
            "chunk_id": 1,
            "content": "Hello World!",
            "score": 0.5,
        }
        defaults.update(kwargs)
        return ChildChunkItem(
            parent_id=kwargs.get("parent_id", 42),
            document_id=kwargs.get("document_id", "first"),
            chunk_id=kwargs.get("chunk_id", 1),
            content=kwargs.get("content", "Hello World!"),
            score=kwargs.get("score", 0.5),
        )

    def test_valid_creation(self) -> None:
        """
        Test valid creation
        """
        item = self._make_item()
        assert item.score == pytest.approx(0.5)
        assert item.content == "Hello World!"

    def test_fields_stored_correctly(self) -> None:
        """
        Test fields
        """
        item = self._make_item(parent_id=5, chunk_id=10, score=0.42)
        assert item.parent_id == 5
        assert item.chunk_id == 10
        assert item.score == pytest.approx(0.42)


class TestSearchResult:
    """
    Tests for SearchResult
    """

    def test_found_false_when_empty(self) -> None:
        """
        Test empty found equals to false
        """

        result = SearchResult()
        assert result.found is False

    def test_found_true_when_has_chunks(self) -> None:
        """
        Test returns true when haas chunks
        """

        chunk = ChildChunkItem(
            parent_id=0,
            document_id="first",
            chunk_id=0,
            content="Hello Wold!",
            score=1.0,
        )
        result = SearchResult(chunks=[chunk])
        assert result.found is True

    def test_default_chunks_is_empty_list(self) -> None:
        """
        Test default chunks
        """

        result = SearchResult()
        assert result.chunks == []


class TestParentChunkResult:
    """
    Tests for ParentChunkResult
    """

    def test_default_values(self) -> None:
        """
        Test default values
        """

        result = ParentChunkResult()
        assert result.document_id == ""
        assert result.parent_id == -1
        assert result.found is False

    def test_not_found_factory(self) -> None:
        """
        Test not found is correct
        """

        result = ParentChunkResult.not_found()

        assert result.found is False
        assert result.content == ""
        assert result.parent_id == -1

    def test_found_result(self) -> None:
        """
        Test fields
        """

        result = ParentChunkResult(
            document_id="first", parent_id=2, content="Hello World!", found=True
        )
        assert result.document_id == "first"
        assert result.parent_id == 2
        assert result.content == "Hello World!"
        assert result.found is True
