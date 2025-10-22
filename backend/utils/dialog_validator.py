"""
Dialog text parser and validator utilities
"""
import re
from typing import List, Set, Tuple, Optional
from pathlib import Path


class DialogValidator:
    """Utility class for parsing and validating dialog text files"""

    # Pattern: "Speaker N: dialog text" where N is a positive integer
    DIALOG_LINE_PATTERN = re.compile(r'^Speaker\s+(\d+):\s*(.+)$')

    @staticmethod
    def parse_dialog_text(text: str) -> List[Tuple[str, str]]:
        """
        Parse dialog text into list of (speaker_id, dialog_text) tuples

        Args:
            text: Dialog text content

        Returns:
            List of (speaker_id, dialog_text) tuples

        Raises:
            ValueError: If text format is invalid
        """
        if not text or not text.strip():
            raise ValueError("Dialog text cannot be empty")

        lines = text.strip().split('\n')
        dialogs = []
        current_line_num = 0

        for line in lines:
            current_line_num += 1

            # Skip empty lines (they are separators)
            if not line.strip():
                continue

            # Match dialog line pattern
            match = DialogValidator.DIALOG_LINE_PATTERN.match(line)
            if not match:
                raise ValueError(
                    f"Invalid dialog format at line {current_line_num}: '{line}'. "
                    f"Expected format: 'Speaker N: dialog text'"
                )

            speaker_num = match.group(1)
            speaker_id = f"Speaker {speaker_num}"
            dialog_text = match.group(2).strip()

            if not dialog_text:
                raise ValueError(
                    f"Dialog text cannot be empty at line {current_line_num} for {speaker_id}"
                )

            dialogs.append((speaker_id, dialog_text))

        if not dialogs:
            raise ValueError("No valid dialog entries found in text")

        return dialogs

    @staticmethod
    def extract_speaker_ids(text: str) -> Set[str]:
        """
        Extract all unique speaker IDs from dialog text

        Args:
            text: Dialog text content

        Returns:
            Set of speaker IDs (e.g., {"Speaker 1", "Speaker 2"})

        Raises:
            ValueError: If text format is invalid
        """
        dialogs = DialogValidator.parse_dialog_text(text)
        return {speaker_id for speaker_id, _ in dialogs}

    @staticmethod
    def validate_speaker_ids(text: str, valid_speaker_ids: Set[str]) -> Tuple[bool, Optional[str]]:
        """
        Validate that all speaker IDs in text exist in the valid set

        Args:
            text: Dialog text content
            valid_speaker_ids: Set of valid speaker IDs from speaker management system

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            used_speaker_ids = DialogValidator.extract_speaker_ids(text)
        except ValueError as e:
            return False, str(e)

        # Check for invalid speaker IDs
        invalid_speakers = used_speaker_ids - valid_speaker_ids
        if invalid_speakers:
            return False, f"Invalid speaker IDs found: {', '.join(sorted(invalid_speakers))}"

        return True, None

    @staticmethod
    def format_dialog_text(dialogs: List[Tuple[str, str]]) -> str:
        """
        Format list of (speaker_id, dialog_text) tuples into dialog text

        Args:
            dialogs: List of (speaker_id, dialog_text) tuples

        Returns:
            Formatted dialog text string
        """
        lines = []
        for speaker_id, dialog_text in dialogs:
            lines.append(f"{speaker_id}: {dialog_text}")
            lines.append("")  # Empty line separator

        # Remove trailing empty line
        return '\n'.join(lines).rstrip() + '\n'

    @staticmethod
    def read_and_validate_file(file_path: Path, valid_speaker_ids: Set[str]) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Read and validate a dialog file

        Args:
            file_path: Path to dialog text file
            valid_speaker_ids: Set of valid speaker IDs

        Returns:
            Tuple of (is_valid, error_message, file_content)
        """
        try:
            if not file_path.exists():
                return False, "File does not exist", None

            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            is_valid, error_msg = DialogValidator.validate_speaker_ids(content, valid_speaker_ids)
            return is_valid, error_msg, content

        except Exception as e:
            return False, f"Failed to read file: {str(e)}", None
