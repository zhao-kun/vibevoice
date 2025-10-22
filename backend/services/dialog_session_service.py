"""
Dialog session management service - handles business logic for dialog sessions
"""
import uuid
from pathlib import Path
from typing import List, Optional, Dict, Any, Set

from backend.models.dialog_session import DialogSession
from backend.utils.file_handler import FileHandler
from backend.utils.dialog_validator import DialogValidator


class DialogSessionService:
    """Service for managing dialog sessions and text files"""

    SESSIONS_META_FILE = 'sessions.json'

    def __init__(self, project_scripts_dir: Path, speaker_service=None):
        """
        Initialize dialog session service for a specific project

        Args:
            project_scripts_dir: Path to project's scripts directory
            speaker_service: Optional SpeakerService instance for validation
        """
        self.scripts_dir = Path(project_scripts_dir)
        self.meta_file_path = self.scripts_dir / self.SESSIONS_META_FILE
        self.file_handler = FileHandler()
        self.speaker_service = speaker_service

        # Ensure scripts directory exists
        self.file_handler.ensure_directory(self.scripts_dir)

        # Initialize metadata file if it doesn't exist
        if not self.meta_file_path.exists():
            self._save_metadata([])

    def _load_metadata(self) -> List[Dict[str, Any]]:
        """
        Load dialog sessions metadata from JSON file

        Returns:
            List of dialog session dictionaries
        """
        try:
            data = self.file_handler.read_json(self.meta_file_path)
            # Ensure it's a list
            if isinstance(data, list):
                return data
            return []
        except FileNotFoundError:
            return []
        except Exception as e:
            raise RuntimeError(f"Failed to load sessions metadata: {str(e)}")

    def _save_metadata(self, sessions: List[Dict[str, Any]]) -> None:
        """
        Atomically save dialog sessions metadata to JSON file

        Args:
            sessions: List of dialog session dictionaries
        """
        try:
            self.file_handler.write_json_atomic(self.meta_file_path, sessions)
        except Exception as e:
            raise RuntimeError(f"Failed to save sessions metadata: {str(e)}")

    def _get_valid_speaker_ids(self) -> Set[str]:
        """
        Get valid speaker IDs from speaker service

        Returns:
            Set of valid speaker IDs
        """
        if not self.speaker_service:
            # If no speaker service provided, skip validation
            return set()

        speakers = self.speaker_service.list_speakers()
        return {speaker.speaker_id for speaker in speakers}

    def list_sessions(self) -> List[DialogSession]:
        """
        List all dialog sessions for the project

        Returns:
            List of DialogSession objects
        """
        metadata = self._load_metadata()
        return [DialogSession.from_dict(data) for data in metadata]

    def get_session(self, session_id: str) -> Optional[DialogSession]:
        """
        Get dialog session by ID

        Args:
            session_id: Session identifier

        Returns:
            DialogSession object or None if not found
        """
        sessions = self.list_sessions()
        for session in sessions:
            if session.session_id == session_id:
                return session
        return None

    def create_session(self, name: str, description: str, dialog_text: str) -> DialogSession:
        """
        Create a new dialog session with text file

        Args:
            name: Session name
            description: Session description
            dialog_text: Dialog text content

        Returns:
            Created DialogSession object

        Raises:
            ValueError: If validation fails
            RuntimeError: If operation fails
        """
        if not name or not name.strip():
            raise ValueError("Session name cannot be empty")

        if not dialog_text or not dialog_text.strip():
            raise ValueError("Dialog text cannot be empty")

        # Validate dialog text format
        try:
            DialogValidator.parse_dialog_text(dialog_text)
        except ValueError as e:
            raise ValueError(f"Invalid dialog text format: {str(e)}")

        # Validate speaker IDs if speaker service is available
        valid_speaker_ids = self._get_valid_speaker_ids()
        if valid_speaker_ids:
            is_valid, error_msg = DialogValidator.validate_speaker_ids(dialog_text, valid_speaker_ids)
            if not is_valid:
                raise ValueError(f"Speaker ID validation failed: {error_msg}")

        # Generate unique session ID and filename
        session_id = str(uuid.uuid4())
        text_filename = f"{session_id}.txt"
        text_file_path = self.scripts_dir / text_filename

        # Load current sessions
        sessions = self.list_sessions()

        try:
            # Save dialog text file
            with open(text_file_path, 'w', encoding='utf-8') as f:
                f.write(dialog_text)

            # Create dialog session
            session = DialogSession.create(session_id, name.strip(), description.strip(), text_filename)

            # Update metadata atomically
            sessions.append(session)
            metadata = [s.to_dict() for s in sessions]
            self._save_metadata(metadata)

            return session

        except Exception as e:
            # Cleanup text file if metadata save fails
            if text_file_path.exists():
                text_file_path.unlink()
            raise RuntimeError(f"Failed to create session: {str(e)}")

    def update_session(self, session_id: str, name: Optional[str] = None,
                       description: Optional[str] = None,
                       dialog_text: Optional[str] = None) -> Optional[DialogSession]:
        """
        Update dialog session metadata and/or text content

        Args:
            session_id: Session identifier
            name: New session name (optional)
            description: New description (optional)
            dialog_text: New dialog text content (optional)

        Returns:
            Updated DialogSession object or None if not found

        Raises:
            ValueError: If validation fails
        """
        sessions = self.list_sessions()

        # Find session
        session_found = False
        for session in sessions:
            if session.session_id == session_id:
                # Validate and update dialog text if provided
                if dialog_text is not None:
                    if not dialog_text.strip():
                        raise ValueError("Dialog text cannot be empty")

                    # Validate dialog text format
                    try:
                        DialogValidator.parse_dialog_text(dialog_text)
                    except ValueError as e:
                        raise ValueError(f"Invalid dialog text format: {str(e)}")

                    # Validate speaker IDs if speaker service is available
                    valid_speaker_ids = self._get_valid_speaker_ids()
                    if valid_speaker_ids:
                        is_valid, error_msg = DialogValidator.validate_speaker_ids(dialog_text, valid_speaker_ids)
                        if not is_valid:
                            raise ValueError(f"Speaker ID validation failed: {error_msg}")

                    # Update text file
                    text_file_path = self.scripts_dir / session.text_filename
                    with open(text_file_path, 'w', encoding='utf-8') as f:
                        f.write(dialog_text)

                # Update metadata
                session.update(name=name, description=description)
                session_found = True
                break

        if not session_found:
            return None

        # Save updated metadata atomically
        metadata = [s.to_dict() for s in sessions]
        self._save_metadata(metadata)

        # Return updated session
        for session in sessions:
            if session.session_id == session_id:
                return session
        return None

    def delete_session(self, session_id: str) -> bool:
        """
        Delete dialog session and its text file

        Args:
            session_id: Session identifier

        Returns:
            True if deleted successfully, False if not found
        """
        sessions = self.list_sessions()

        # Find session
        session_to_delete = None
        for session in sessions:
            if session.session_id == session_id:
                session_to_delete = session
                break

        if not session_to_delete:
            return False

        try:
            # Delete text file
            text_file_path = self.scripts_dir / session_to_delete.text_filename
            if text_file_path.exists():
                text_file_path.unlink()

            # Remove from list
            sessions = [s for s in sessions if s.session_id != session_id]

            # Save updated metadata atomically
            metadata = [s.to_dict() for s in sessions]
            self._save_metadata(metadata)

            return True

        except Exception as e:
            raise RuntimeError(f"Failed to delete session: {str(e)}")

    def get_session_text(self, session_id: str) -> Optional[str]:
        """
        Get dialog text content for a session

        Args:
            session_id: Session identifier

        Returns:
            Dialog text content or None if session doesn't exist
        """
        session = self.get_session(session_id)
        if not session:
            return None

        text_file_path = self.scripts_dir / session.text_filename
        if not text_file_path.exists():
            return None

        try:
            with open(text_file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception:
            return None

    def get_text_file_path(self, session_id: str) -> Optional[Path]:
        """
        Get absolute path to session's text file

        Args:
            session_id: Session identifier

        Returns:
            Path to text file or None if session doesn't exist
        """
        session = self.get_session(session_id)
        if session:
            return self.scripts_dir / session.text_filename
        return None
