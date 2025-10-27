"""
Generation data models and schemas
"""
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional, Dict, Any, Callable

from config.configuration_vibevoice import InferencePhase



# Callable type for status update callbacks
# Signature: (phase: InferencePhase, **kwargs) -> Any
UpdateStatusCallable = Callable[..., Any]


@dataclass
class Generation:
    """Generation metadata model"""
    request_id: str  # Unique identifier
    session_id: str  # Speaker role identifier
    status: InferencePhase  # Status of the generation request
    output_filename: Optional[str]  # Generated audio file name
    percentage: Optional[float]  # Completion percentage
    model_dtype: str  # Type of model used for generation, e.g. "bf16" and "float8_e4m3fn"
    cfg_scale: Optional[float]  # Classifier-free guidance scale
    attn_implementation: Optional[str]  # Attention implementation used
    seeds: int  # Random seed used for generation
    created_at: str  # ISO format timestamp
    updated_at: str  # ISO format timestamp
    project_id: Optional[str] = None  # Project identifier
    project_dir: Optional[str] = None  # Output audio directory
    details: Dict[str, Any] = None  # Additional details

    def to_dict(self) -> Dict[str, Any]:
        """Convert generation request to dictionary"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Generation':
        """Create generation request from dictionary"""
        return cls(**data)

    @classmethod
    def create(cls, request_id: str, session_id: str,
               seeds: int = 42,
               cfg_scale: float = 1.3,
               model_dtype: str = "float8_e4m3fn",
               attn_implementation: str = "sdpa",
               project_id: str = None,
               project_dir: str = "output/audio") -> 'Generation':
        """Create a new generation request with timestamps"""
        now = datetime.utcnow().isoformat()
        return cls(
            request_id=request_id,
            session_id=session_id,
            status=InferencePhase.PENDING,
            output_filename=None,
            percentage=None,
            model_dtype=model_dtype,
            attn_implementation=attn_implementation,
            cfg_scale=cfg_scale,
            created_at=now,
            seeds=seeds,
            updated_at=now,
            project_id=project_id,
            project_dir=project_dir,
            details={},
        )

    def update_status(self, phase: InferencePhase, *args, **kwargs) -> None:
        """Update generation request status"""
        self.status = phase
        self.details.update(kwargs)
        self.updated_at = datetime.utcnow().isoformat() 

