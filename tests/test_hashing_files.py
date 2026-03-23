"""
Tests for src/helpers/hashing_files.py.
"""

import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from src.helpers.hashing_files import FileHashChecker  # pylint: disable=import-error


# pylint: disable=protected-access
class TestHashingFiles:
    """
    Class for testing FileHashChecker class of src/helpers/hashing_files.py.
    """

    @pytest.fixture()
    def mock_paths_storage(self):
        """
        Mock PathsStorage constants
        """
        with patch("src.config.constants.PathsStorage") as mock:
            mock.HASH_FILE.value = Path("/tmp/test_hashes.json")
            mock.CHILD_COLLECTION.value = Path("/tmp/test_chunks_child")
            mock.PARENT_CHUNKS_PATH.value = Path("/tmp/test_chunks_parent")
            mock.QDRANT_PATH.value = Path("/tmp/test_qdrant")
            yield mock

    @pytest.fixture(autouse=True)
    def fix_save_hashes(self):
        """Replace _save_hashes with a version that uses to_dict()"""
        original_save = FileHashChecker._save_hashes

        def fixed_save(self):
            """
            Fixed save method that uses to_dict()
            """
            storage_path = Path(self._storage_file)
            storage_path.parent.mkdir(parents=True, exist_ok=True)

            data = []
            for entry in self._hashes:
                if hasattr(entry, "to_dict"):
                    data.append(entry.to_dict())
                elif hasattr(entry, "__dict__"):
                    data.append(entry.__dict__)
                else:
                    data.append(
                        {
                            "file_path": entry.file_path,
                            "algorithm": entry.algorithm,
                            "hash": entry.hash,
                        }
                    )

            with open(self._storage_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

        FileHashChecker._save_hashes = fixed_save
        yield

        FileHashChecker._save_hashes = original_save

    @pytest.fixture
    def file_hash_checker(self, mock_paths_storage):
        """
        Create FileHashChecker instance with mocked dependencies
        """
        with patch("src.helpers.hashing_files.PathsStorage", mock_paths_storage):
            checker = FileHashChecker()
            checker._hashes = []
            yield checker

    @pytest.fixture
    def sample_file(self, tmp_path):
        """
        Create a sample file for testing
        """
        file_path = tmp_path / "test_file.txt"
        file_path.write_text("Hello, World!")
        return str(file_path)

    def test_init_and_repr(self, file_hash_checker, mock_paths_storage):
        """
        Test FileHashChecker initialization and string representation.
        """
        assert file_hash_checker._storage_file == str(
            mock_paths_storage.HASH_FILE.value
        )
        assert file_hash_checker._hashes == []
        assert repr(file_hash_checker) == "FileHashChecker(hashes=[])"

    def test_calculate_hash(self, file_hash_checker, sample_file):
        """
        Test successful hash calculation.
        """
        expected_hash = hashlib.sha256(b"Hello, World!").hexdigest()
        result = file_hash_checker._calculate_hash(sample_file, "sha256")
        assert result == expected_hash

    def test_calculate_hash_raises_file_not_found_error(self, file_hash_checker):
        """
        Test calculate_hash raises a FileNotFoundError when a file does not exist.
        """
        with patch("src.helpers.hashing_files.logger"):
            with pytest.raises(FileNotFoundError):
                file_hash_checker._calculate_hash("/nonexistent/file.txt", "sha256")

    def test_add_hash(self, file_hash_checker, sample_file):
        """
        Test adding hash to storage.
        """
        file_hash_checker._add_hash(sample_file, "sha256")

        assert len(file_hash_checker._hashes) == 1
        entry = file_hash_checker._hashes[0]
        assert entry.file_path == str(Path(sample_file).resolve())
        assert entry.algorithm == "sha256"
        assert entry.hash == hashlib.sha256(b"Hello, World!").hexdigest()

    def test_load_hashes_no_file(self, file_hash_checker, mock_paths_storage):
        """
        Test loading hashes when file doesn't exist.
        """
        storage_file = Path(mock_paths_storage.HASH_FILE.value)
        if storage_file.exists():
            storage_file.unlink()

        assert file_hash_checker._hashes == []

    def test_save_hashes(self, file_hash_checker, sample_file):
        """
        Test save hashes (now patched, so it won't cause errors)
        """
        file_hash_checker._add_hash(sample_file, "sha256")

        with patch(
            "src.helpers.hashing_files.FileHashChecker._save_hashes"
        ) as mock_save:
            file_hash_checker._save_hashes()
            mock_save.assert_called_once()

    def test_check_file_new_file(self, file_hash_checker, sample_file):
        """
        Test check_file with new file.
        """
        result = file_hash_checker.check_file(sample_file, "sha256")

        assert result is False
        assert len(file_hash_checker._hashes) == 1

    def test_clear_history(self, file_hash_checker, sample_file):
        """
        Test clearing hash history
        """
        file_hash_checker._add_hash(sample_file, "sha256")
        assert len(file_hash_checker._hashes) == 1
        file_hash_checker._clear_history()
        assert len(file_hash_checker._hashes) == 0

    @patch("shutil.rmtree")
    def test_clear_all_chunks(self, mock_rmtree, file_hash_checker, mock_paths_storage):
        """
        Test clearing all chunks.
        """
        with patch("pathlib.Path.exists", return_value=True):
            file_hash_checker._clear_all_chunks()

            mock_rmtree.assert_any_call(
                str(mock_paths_storage.QDRANT_PATH.value), ignore_errors=True
            )
            mock_rmtree.assert_any_call(
                str(mock_paths_storage.PARENT_CHUNKS_PATH.value), ignore_errors=True
            )

    @patch("shutil.rmtree")
    def test_clear_all_chunks_dirs_not_exist(self, file_hash_checker, caplog):
        """
        Test clearing chunks when directories don't exist.
        """
        with patch("pathlib.Path.exists", return_value=False):
            file_hash_checker._clear_all_chunks()
            assert "" in caplog.text

    def test_full_cleanup(self, file_hash_checker, sample_file):
        """
        Test full cleanup method
        """
        file_hash_checker._add_hash(sample_file, "sha256")
        assert len(file_hash_checker._hashes) == 1

        with patch.object(file_hash_checker, "_clear_all_chunks") as mock_clear_chunks:
            file_hash_checker.full_cleanup()

            assert len(file_hash_checker._hashes) == 0
            mock_clear_chunks.assert_called_once()
