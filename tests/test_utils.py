"""
Tests for utils
"""

import os
from typing import Any
from unittest.mock import patch

import pytest

from src.helpers.utils import (  # pylint: disable=import-error
    _choose_device,
    _load_md_files,
)


class TestChooseDevice:
    """
    Class for testing device choice
    """

    def test_returns_passed_device(self) -> None:
        """
        Test defined device
        """

        assert _choose_device("mps") == "mps"

    def test_env_variable_used_when_no_arg(self) -> None:
        """
        Test env device is chosen
        """

        with patch.dict(os.environ, {"EMBEDDINGS_DEVICE": "cpu"}):
            with patch(
                "src.config.constants.EMBEDDINGS_DEVICE_ENV", "EMBEDDINGS_DEVICE"
            ):
                device = _choose_device()
                assert device == "cpu"

    def test_cuda_selected_when_available_and_no_env(self) -> None:
        """
        Test cuda selected if available
        """

        with patch.dict(os.environ, {}, clear=True):
            with patch("src.config.constants.EMBEDDINGS_DEVICE_ENV", "NO_SUCH_ENV_VAR"):
                with patch("torch.cuda.is_available", return_value=True):
                    assert _choose_device() == "cuda"

    def test_cpu_selected_when_cuda_unavailable(self) -> None:
        """
        Test cpu selected if no other option
        """

        with patch.dict(os.environ, {}, clear=True):
            with patch("src.config.constants.EMBEDDINGS_DEVICE_ENV", "NO_SUCH_ENV_VAR"):
                with patch("torch.cuda.is_available", return_value=False):
                    assert _choose_device() == "cpu"


class TestLoadMdFiles:
    """
    Tests for .md files loader
    """

    def test_raises_when_no_md_files(self, tmp_path: Any) -> None:
        """
        Test raises error when no raw files present
        """

        with patch("src.helpers.utils.PathsStorage") as mock_paths:
            mock_paths.RAW_MD_COLLECTION.value = tmp_path
            with pytest.raises(FileNotFoundError, match="No .md files found"):
                _load_md_files()

    def test_loads_md_files_correctly(self, tmp_path: Any) -> None:
        """
        Test loads .md files correctly
        """

        (tmp_path / "doc_a.md").write_text("content A", encoding="utf-8")
        (tmp_path / "doc_b.md").write_text("content B", encoding="utf-8")

        with patch("src.helpers.utils.PathsStorage") as mock_paths:
            mock_paths.RAW_MD_COLLECTION.value = tmp_path
            texts, doc_ids = _load_md_files()

        assert len(texts) == 2
        assert len(doc_ids) == 2
        assert "doc_a" in doc_ids
        assert "doc_b" in doc_ids
        assert "content A" in texts or "content B" in texts

    def test_returns_sorted_files(self, tmp_path: Any) -> None:
        """
        Test returns files correctly
        """

        (tmp_path / "z_file.md").write_text("Z")
        (tmp_path / "a_file.md").write_text("A")

        with patch("src.helpers.utils.PathsStorage") as mock_paths:
            mock_paths.RAW_MD_COLLECTION.value = tmp_path
            _, doc_ids = _load_md_files()

        assert doc_ids[0] == "a_file"
        assert doc_ids[1] == "z_file"
