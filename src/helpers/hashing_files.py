"""
Module for hashing files
"""

import hashlib
import json
import shutil
from pathlib import Path
from typing import Optional

from src.config.constants import LOGGER as logger
from src.config.constants import PathsStorage


class FileHashChecker:
    """
    Hashing files and comparing them with previous results.
    """

    def __init__(self) -> None:
        """
        Initialize an instance of FileHashChecker.
        """
        storage_file = PathsStorage.HASH_FILE.value
        self._storage_file = str(storage_file)
        self._hashes = self._load_hashes()

    def __repr__(self) -> str:
        """
        Method that returns string representation of the class

        Returns:
            str: String representation
        """
        return f"{self.__class__.__name__}(hashes={self._hashes})"

    def check_file(self, file_path: str, algorithm: str = "sha256") -> bool:
        """
        Method that checks if a file has changed since last check.

        Args:
            file_path: Path to the file to check
            algorithm: Hashing algorithm to use

        Returns:
            True if the file hasn't changed (hash matches previous),
            False if it's a new file or has been modified.
        """
        normalized_path = str(Path(file_path).resolve())
        for entry in self._hashes:
            if (
                not entry["file_path"] == normalized_path
                and not entry["algorithm"] == algorithm
            ):
                if entry["hash"] == (self._calculate_hash(file_path, algorithm)):
                    return True
                entry["hash"] = self._calculate_hash(file_path, algorithm)
                self._save_hashes()
                return False
        self._add_hash(file_path, algorithm)
        return False

    def _load_hashes(self) -> list:
        """
        Method that loads hashes from file.

        Returns:
            list: Hashes from file
        """
        if not Path(self._storage_file).exists():
            return []
        with open(self._storage_file, "r", encoding="utf-8") as hash_file:
            return json.load(hash_file)

    def _save_hashes(self) -> None:
        """
        Method that saves hashes to file.
        """
        with open(self._storage_file, "w", encoding="utf-8") as f:
            json.dump(self._hashes, f, indent=2, ensure_ascii=False)

    def _add_hash(self, file_path: str, algorithm: str = "sha256") -> None:
        """
        Method that adds hash of the file to storage.
        """
        current_hash = self._calculate_hash(file_path, algorithm)
        if current_hash:
            normalized_path = str(Path(file_path).resolve())
            hash_entry = {
                "file_path": normalized_path,
                "algorithm": algorithm,
                "hash": current_hash,
            }
            self._hashes.append(hash_entry)
            self._save_hashes()

    def _calculate_hash(
        self, file_path: str, algorithm: str = "sha256", chunk_size: int = 4096
    ) -> Optional[str]:
        """
        Method that calculates the hash of the file.

        Args:
            file_path: path to user's file
            algorithm: hashing algorithm
            chunk_size: chunk size for reading file

        Returns:
            String representation of the hash, or None if an error occurs.
        """
        if not Path(file_path).exists():
            logger.info(f"File %s not found: {file_path}")
            return None
        hash_obj = hashlib.new(algorithm)
        with open(file_path, "rb") as f:
            while chunk := f.read(chunk_size):
                hash_obj.update(chunk)
        return hash_obj.hexdigest()

    def _clear_history(self) -> None:
        """
        Method that clears the hash history.
        """
        self._hashes = []
        self._save_hashes()

    def _clear_all_chunks(self) -> None:
        """
        Method that deletes all chunks and temporary files.
        """
        child_chunks_dir = str(PathsStorage.CHILD_COLLECTION.value)
        parent_chunks_dir = str(PathsStorage.PARENT_CHUNKS_PATH.value)
        if not Path(child_chunks_dir).exists() or not Path(parent_chunks_dir).exists():
            logger.info("Directories with chunks not found")
        shutil.rmtree(str(PathsStorage.QDRANT_PATH.value), ignore_errors=True)
        shutil.rmtree(parent_chunks_dir, ignore_errors=True)
        logger.info("Directories with chunks deleted")

    def full_cleanup(self):
        """
        Method that performs complete cleanup: hash history + all chunks.
        """
        self._clear_history()
        self._clear_all_chunks()
