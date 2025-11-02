"""
Speaker role data models and schemas
"""
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional, Dict, Any


@dataclass
class SpeakerRole:
    """Speaker role metadata model"""
    speaker_id: str  # "Speaker 1", "Speaker 2", etc. (auto-generated, serves as the name)
    description: str  # Speaker description/details
    voice_filename: str  # Voice sample filename (stored in voices/ directory)
    created_at: str  # ISO format timestamp
    updated_at: str  # ISO format timestamp

    def to_dict(self) -> Dict[str, Any]:
        """Convert speaker role to dictionary"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SpeakerRole':
        """Create speaker role from dictionary"""
        # Handle legacy data that might have 'name' field
        if 'name' in data:
            data = {k: v for k, v in data.items() if k != 'name'}
        return cls(**data)

    @classmethod
    def create(cls, speaker_id: str, description: str, voice_filename: str) -> 'SpeakerRole':
        """Create a new speaker role with timestamps"""
        now = datetime.utcnow().isoformat()
        return cls(
            speaker_id=speaker_id,
            description=description,
            voice_filename=voice_filename,
            created_at=now,
            updated_at=now
        )

    def update(self, description: Optional[str] = None) -> None:
        """Update speaker role metadata"""
        if description is not None:
            self.description = description
        self.updated_at = datetime.utcnow().isoformat()
