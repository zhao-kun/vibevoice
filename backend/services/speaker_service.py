"""
Speaker role management service - handles business logic for speaker roles
"""
import uuid
from pathlib import Path
from typing import List, Optional, Dict, Any
from werkzeug.datastructures import FileStorage

from backend.models.speaker import SpeakerRole
from backend.utils.file_handler import FileHandler


class SpeakerService:
    """Service for managing speaker roles and voice files"""

    SPEAKERS_META_FILE = 'speakers.json'
    ALLOWED_EXTENSIONS = {'.wav', '.mp3', '.m4a', '.flac'}

    def __init__(self, project_voices_dir: Path):
        """
        Initialize speaker service for a specific project

        Args:
            project_voices_dir: Path to project's voices directory
        """
        self.voices_dir = Path(project_voices_dir)
        self.meta_file_path = self.voices_dir / self.SPEAKERS_META_FILE
        self.file_handler = FileHandler()

        # Ensure voices directory exists
        self.file_handler.ensure_directory(self.voices_dir)

        # Initialize metadata file if it doesn't exist
        if not self.meta_file_path.exists():
            self._save_metadata([])

    def _load_metadata(self) -> List[Dict[str, Any]]:
        """
        Load speakers metadata from JSON file

        Returns:
            List of speaker role dictionaries (ordered by speaker_id)
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
            raise RuntimeError(f"Failed to load speakers metadata: {str(e)}")

    def _save_metadata(self, speakers: List[Dict[str, Any]]) -> None:
        """
        Atomically save speakers metadata to JSON file

        Args:
            speakers: List of speaker role dictionaries
        """
        try:
            self.file_handler.write_json_atomic(self.meta_file_path, speakers)
        except Exception as e:
            raise RuntimeError(f"Failed to save speakers metadata: {str(e)}")

    def _generate_speaker_id(self, index: int) -> str:
        """
        Generate speaker ID from index

        Args:
            index: Zero-based index

        Returns:
            Speaker ID in format "Speaker N" (N starts from 1)
        """
        return f"Speaker {index + 1}"

    def _reindex_speakers(self, speakers: List[SpeakerRole]) -> List[SpeakerRole]:
        """
        Reindex all speaker IDs to ensure continuity

        Args:
            speakers: List of speaker roles

        Returns:
            List of speaker roles with reindexed IDs
        """
        for i, speaker in enumerate(speakers):
            speaker.speaker_id = self._generate_speaker_id(i)
        return speakers

    def _validate_audio_file(self, filename: str) -> bool:
        """
        Validate audio file extension

        Args:
            filename: Filename to validate

        Returns:
            True if valid, False otherwise
        """
        ext = Path(filename).suffix.lower()
        return ext in self.ALLOWED_EXTENSIONS

    def list_speakers(self) -> List[SpeakerRole]:
        """
        List all speaker roles for the project

        Returns:
            List of SpeakerRole objects
        """
        metadata = self._load_metadata()
        return [SpeakerRole.from_dict(data) for data in metadata]

    def get_speaker(self, speaker_id: str) -> Optional[SpeakerRole]:
        """
        Get speaker role by ID

        Args:
            speaker_id: Speaker identifier (e.g., "Speaker 1")

        Returns:
            SpeakerRole object or None if not found
        """
        speakers = self.list_speakers()
        for speaker in speakers:
            if speaker.speaker_id == speaker_id:
                return speaker
        return None

    def add_speaker(self, name: str, description: str, voice_file: FileStorage) -> SpeakerRole:
        """
        Add a new speaker role with voice file

        Args:
            name: Speaker name
            description: Speaker description
            voice_file: Uploaded voice file

        Returns:
            Created SpeakerRole object

        Raises:
            ValueError: If validation fails
            RuntimeError: If operation fails
        """
        if not name or not name.strip():
            raise ValueError("Speaker name cannot be empty")

        if not voice_file or not voice_file.filename:
            raise ValueError("Voice file is required")

        # Validate file extension
        if not self._validate_audio_file(voice_file.filename):
            raise ValueError(f"Invalid audio file. Allowed extensions: {', '.join(self.ALLOWED_EXTENSIONS)}")

        # Generate unique filename
        ext = Path(voice_file.filename).suffix.lower()
        unique_filename = f"{uuid.uuid4().hex}{ext}"
        file_path = self.voices_dir / unique_filename

        # Load current speakers
        speakers = self.list_speakers()

        # Generate speaker ID (next in sequence)
        speaker_id = self._generate_speaker_id(len(speakers))

        try:
            # Save voice file
            voice_file.save(str(file_path))

            # Create speaker role
            speaker = SpeakerRole.create(speaker_id, name.strip(), description.strip(), unique_filename)

            # Update metadata atomically
            speakers.append(speaker)
            metadata = [s.to_dict() for s in speakers]
            self._save_metadata(metadata)

            return speaker

        except Exception as e:
            # Cleanup voice file if metadata save fails
            if file_path.exists():
                file_path.unlink()
            raise RuntimeError(f"Failed to add speaker: {str(e)}")

    def update_speaker(self, speaker_id: str, name: Optional[str] = None,
                       description: Optional[str] = None) -> Optional[SpeakerRole]:
        """
        Update speaker role metadata (not the voice file)

        Args:
            speaker_id: Speaker identifier
            name: New speaker name (optional)
            description: New description (optional)

        Returns:
            Updated SpeakerRole object or None if not found
        """
        speakers = self.list_speakers()

        # Find and update speaker
        speaker_found = False
        for speaker in speakers:
            if speaker.speaker_id == speaker_id:
                speaker.update(name=name, description=description)
                speaker_found = True
                break

        if not speaker_found:
            return None

        # Save updated metadata atomically
        metadata = [s.to_dict() for s in speakers]
        self._save_metadata(metadata)

        # Return updated speaker
        for speaker in speakers:
            if speaker.speaker_id == speaker_id:
                return speaker
        return None

    def delete_speaker(self, speaker_id: str) -> bool:
        """
        Delete speaker role and its voice file

        Args:
            speaker_id: Speaker identifier

        Returns:
            True if deleted successfully, False if not found
        """
        speakers = self.list_speakers()

        # Find speaker
        speaker_to_delete = None
        for speaker in speakers:
            if speaker.speaker_id == speaker_id:
                speaker_to_delete = speaker
                break

        if not speaker_to_delete:
            return False

        try:
            # Delete voice file
            voice_file_path = self.voices_dir / speaker_to_delete.voice_filename
            if voice_file_path.exists():
                voice_file_path.unlink()

            # Remove from list
            speakers = [s for s in speakers if s.speaker_id != speaker_id]

            # Reindex speaker IDs to maintain continuity
            speakers = self._reindex_speakers(speakers)

            # Save updated metadata atomically
            metadata = [s.to_dict() for s in speakers]
            self._save_metadata(metadata)

            return True

        except Exception as e:
            raise RuntimeError(f"Failed to delete speaker: {str(e)}")

    def get_voice_file_path(self, speaker_id: str) -> Optional[Path]:
        """
        Get absolute path to speaker's voice file

        Args:
            speaker_id: Speaker identifier

        Returns:
            Path to voice file or None if speaker doesn't exist
        """
        speaker = self.get_speaker(speaker_id)
        if speaker:
            return self.voices_dir / speaker.voice_filename
        return None

    def get_speakers_filepath(self, speaker_names: List[str]) -> List[str]:
        """
        Get absolute paths to multiple speakers' voice files

        Args:
            speaker_names: List of speaker names (e.g., "Speaker 1", "Speaker 2")

        Returns:
            List of paths to voice files
        """
        file_paths = []
        for name in speaker_names:
            speaker = self.get_speaker(name)
            if speaker:
                file_paths.append(str(self.voices_dir / speaker.voice_filename))
        return file_paths
