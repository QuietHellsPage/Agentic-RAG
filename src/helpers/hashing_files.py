"""
Module for hashing files
"""

import hashlib
import json
import os
import shutil
from typing import Optional

from src.config.constants import PathsStorage


class FileHashChecker:
    """
    Hashing files and comparing them with previous results.
    """

    def __init__(self):
        """
        Initialize an instance of FileHashChecker.

        """
        storage_file = PathsStorage.HASH_FILE.value
        self.storage_file = str(storage_file)
        self.hashes = self._load_hashes()

    def _load_hashes(self) -> list:
        """
        Loads hashes from file.

        Returns:
            list: Hashes from file
        """
        if os.path.exists(self.storage_file):
            with open(self.storage_file, 'r', encoding='utf-8') as hash_file:
                return json.load(hash_file)
        return []

    def _save_hashes(self) -> None:
        """Saves hashes to file."""
        with open(self.storage_file, 'w', encoding='utf-8') as f:
            json.dump(self.hashes, f, indent=2, ensure_ascii=False)

    def _add_hash(self, file_path: str, algorithm: str = 'sha256') -> None:
        """Adds hash of the file to storage."""
        current_hash = self.calculate_hash(file_path, algorithm)
        if current_hash:
            normalized_path = os.path.abspath(file_path)

            # Добавляем хэш в список
            hash_entry = {
                "file_path": normalized_path,
                "algorithm": algorithm,
                "hash": current_hash
            }

            self.hashes.append(hash_entry)
            self._save_hashes()

    def calculate_hash(self, file_path: str, algorithm: str = 'sha256', chunk_size: int = 4096) -> \
    Optional[str]:
        """
        Calculates the hash of the file.

        Args:
            file_path: path to user's file
            algorithm: hashing algorithm
            chunk_size: chunk size for reading file

        Returns:
            String representation of the hash, or None if an error occurs.
        """
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return None

        hash_obj = hashlib.new(algorithm)

        with open(file_path, 'rb') as f:
            while chunk := f.read(chunk_size):
                hash_obj.update(chunk)
        return hash_obj.hexdigest()

    def check_file(self, file_path: str, algorithm: str = 'sha256') -> bool:
        """
        Check if a file has changed since last check.

        Args:
            file_path: Path to the file to check
            algorithm: Hashing algorithm to use

        Returns:
            True if the file hasn't changed (hash matches previous),
            False if it's a new file or has been modified.
        """
        current_hash = self.calculate_hash(file_path, algorithm)

        normalized_path = os.path.abspath(file_path)

        for entry in self.hashes:
            if entry['file_path'] == normalized_path and entry['algorithm'] == algorithm:
                if entry['hash'] == current_hash:
                    return True
                entry['hash'] = current_hash
                self._save_hashes()
                return False

        self._add_hash(file_path, algorithm)
        return False

    def clear_history(self) -> None:
        """
        Clears the hash history.
        """
        self.hashes = []
        self._save_hashes()

    def clear_all_chunks(self) -> None:
        """
        Deletes all chunks and temporary files.
        """
        child_chunks_dir = str(PathsStorage.CHILD_COLLECTION.value)
        parent_chunks_dir = str(PathsStorage.PARENT_CHUNKS_PATH.value)
        if os.path.exists(child_chunks_dir) or os.path.exists(parent_chunks_dir):
            shutil.rmtree(str(PathsStorage.QDRANT_PATH.value))
            shutil.rmtree(parent_chunks_dir)
            print("Directories with chunks deleted")

    def full_cleanup(self):
        """
        Performs complete cleanup: hash history + all chunks.
        """
        self.clear_history()
        self.clear_all_chunks()
