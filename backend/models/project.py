"""
Project data models and schemas
"""
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional, Dict, Any


@dataclass
class Project:
    """Project metadata model"""
    id: str  # Project directory name / unique identifier
    name: str  # Display name
    description: str  # Project description
    created_at: str  # ISO format timestamp
    updated_at: str  # ISO format timestamp

    def to_dict(self) -> Dict[str, Any]:
        """Convert project to dictionary"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Project':
        """Create project from dictionary"""
        return cls(**data)

    @classmethod
    def create(cls, project_id: str, name: str, description: str = "") -> 'Project':
        """Create a new project with timestamps"""
        now = datetime.utcnow().isoformat()
        return cls(
            id=project_id,
            name=name,
            description=description,
            created_at=now,
            updated_at=now
        )

    def update(self, name: Optional[str] = None, description: Optional[str] = None) -> None:
        """Update project metadata"""
        if name is not None:
            self.name = name
        if description is not None:
            self.description = description
        self.updated_at = datetime.utcnow().isoformat()
