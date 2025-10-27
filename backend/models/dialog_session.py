"""
Dialog session data models and schemas
"""
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional, Dict, Any


@dataclass
class DialogSession:
    """Dialog session metadata model"""
    session_id: str  # Unique identifier
    name: str  # Session name
    description: str  # Session description
    text_filename: str  # Text file name (stored in scripts/ directory)
    created_at: str  # ISO format timestamp
    updated_at: str  # ISO format timestamp

    def to_dict(self) -> Dict[str, Any]:
        """Convert dialog session to dictionary"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DialogSession':
        """Create dialog session from dictionary"""
        return cls(**data)

    @classmethod
    def create(cls, session_id: str, name: str, description: str, text_filename: str) -> 'DialogSession':
        """Create a new dialog session with timestamps"""
        now = datetime.utcnow().isoformat()
        return cls(
            session_id=session_id,
            name=name,
            description=description,
            text_filename=text_filename,
            created_at=now,
            updated_at=now
        )

    def update(self, name: Optional[str] = None, description: Optional[str] = None) -> None:
        """Update dialog session metadata"""
        if name is not None:
            self.name = name
        if description is not None:
            self.description = description
        self.updated_at = datetime.utcnow().isoformat()
