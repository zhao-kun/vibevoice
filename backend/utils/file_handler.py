"""
File and directory handling utilities
"""
import json
import shutil
from pathlib import Path
from typing import List, Dict, Any


class FileHandler:
    """Utility class for file and directory operations"""

    @staticmethod
    def ensure_directory(path: Path) -> None:
        """
        Ensure directory exists, create if not

        Args:
            path: Directory path
        """
        path.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def read_json(file_path: Path) -> Dict[str, Any]:
        """
        Read JSON file

        Args:
            file_path: Path to JSON file

        Returns:
            Dictionary containing JSON data

        Raises:
            FileNotFoundError: If file doesn't exist
            json.JSONDecodeError: If JSON is invalid
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    @staticmethod
    def write_json(file_path: Path, data: Dict[str, Any], indent: int = 2) -> None:
        """
        Write data to JSON file

        Args:
            file_path: Path to JSON file
            data: Data to write
            indent: JSON indentation (default: 2)
        """
        # Ensure parent directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)

    @staticmethod
    def write_json_atomic(file_path: Path, data: Dict[str, Any], indent: int = 2) -> None:
        """
        Atomically write data to JSON file using temp file + rename

        This ensures that if the write fails, the original file remains intact.

        Args:
            file_path: Path to JSON file
            data: Data to write
            indent: JSON indentation (default: 2)
        """
        import tempfile
        import os

        # Ensure parent directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Write to temporary file in the same directory
        temp_fd, temp_path = tempfile.mkstemp(
            dir=file_path.parent,
            prefix=f'.{file_path.name}.',
            suffix='.tmp'
        )

        try:
            # Write JSON to temp file
            with os.fdopen(temp_fd, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=indent, ensure_ascii=False)

            # Atomically replace the original file
            os.replace(temp_path, file_path)
        except Exception:
            # Clean up temp file on error
            try:
                os.unlink(temp_path)
            except OSError:
                pass
            raise

    @staticmethod
    def delete_directory(path: Path, ignore_errors: bool = False) -> None:
        """
        Delete directory and all its contents

        Args:
            path: Directory path to delete
            ignore_errors: If True, ignore errors during deletion
        """
        if path.exists() and path.is_dir():
            shutil.rmtree(path, ignore_errors=ignore_errors)

    @staticmethod
    def list_directories(parent_path: Path) -> List[str]:
        """
        List all subdirectories in a parent directory

        Args:
            parent_path: Parent directory path

        Returns:
            List of directory names
        """
        if not parent_path.exists():
            return []

        return [
            d.name for d in parent_path.iterdir()
            if d.is_dir() and not d.name.startswith('.')
        ]

    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """
        Sanitize filename by removing/replacing invalid characters

        Args:
            filename: Original filename

        Returns:
            Sanitized filename
        """
        # Replace invalid characters with underscore
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            filename = filename.replace(char, '_')

        # Remove leading/trailing spaces and dots
        filename = filename.strip('. ')

        # Ensure filename is not empty
        if not filename:
            filename = 'untitled'

        return filename
