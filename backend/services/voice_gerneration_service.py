import threading

from typing import List, Dict, Any, Optional
from pathlib import Path
from utils.file_handler import FileHandler
from backend.services.speaker_service import SpeakerService
from backend.services.dialog_session_service import DialogSessionService
from backend.models.generation import Generation
from backend.gen_voice.task import Task, gm
from backend.inference.inference import InferenceBase
class VoiceGenerationService:

    GENERATION_META_FILE = 'generation.json'

    def __init__(self, project_generation_dir: Path, speaker_service: SpeakerService, dialog_service: DialogSessionService):
        """
        Initialize voice generation service for a specific project

        Args:
            project_generation_dir: Path to project's generation directory
        """
        self.output_dir = Path(project_generation_dir)
        self.meta_file_path = self.output_dir / self.GENERATION_META_FILE
        self.file_handler = FileHandler()

        # Ensure output directory exists
        self.file_handler.ensure_directory(self.output_dir)

        self.speaker_service = speaker_service
        self.dialog_service = dialog_service

        # Initialize metadata file if it doesn't exist
        if not self.meta_file_path.exists():
            self._save_metadata([])

    def _load_metadata(self) -> List[Dict[str, Any]]:
        """
        Load generation metadata from JSON file

        Returns:
            List of generation dictionaries
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
            raise RuntimeError(f"Failed to load generation metadata: {str(e)}")

    def _save_metadata(self, generations: List[Dict[str, Any]]) -> None:
        """
        Atomically save generation metadata to JSON file

        Args:
            generations: List of generation dictionaries
        """
        try:
            self.file_handler.write_json(self.meta_file_path, generations)
        except Exception as e:
            raise RuntimeError(f"Failed to save generation metadata: {str(e)}")
    
    def list_generations(self) -> List[Generation]:
        """
        List all generations for the project

        Returns:
            List of Generation objects
        """
        metadata = self._load_metadata()
        return [Generation.from_dict(data) for data in metadata]

    def generation(self, dialog_session_id: str, request_id: str, seeds: int = 42,
                   cfg_scale: float = 1.3, model_dtype: str = "float8_e4m3fn",
                   attn_implementation: str = "sdpa") -> Generation:
        """
        Generate voices for a specific dialog session

        Args:
            dialog_session_id: Dialog session identifier
            request_id: Request identifier
        """

        dialog_session = self.dialog_service.get_session(dialog_session_id)
        if not dialog_session:
            raise ValueError(f"Dialog session with ID '{dialog_session_id}' not found")

        # Placeholder for actual voice generation logic
        generation = Generation.create(request_id, dialog_session_id, 
                                       seeds=seeds,
                                       cfg_scale=cfg_scale,
                                       model_dtype=model_dtype,
                                       attn_implementation=attn_implementation)
        inference = InferenceBase.create(generation, self.speaker_service,
                                         self.dialog_service, self.meta_file_path)

        task = Task.from_inference(inference=inference,
                                   file_handler=self.file_handler,
                                   meta_file_path=str(self.meta_file_path))

        return generation if gm.add_inference_task(task) else None
