import threading

from time import sleep
from typing import Any, Dict, List
from transformers.utils import logging
from backend.inference.inference import InferenceBase
from backend.models.generation import InferencePhase
from utils.file_handler import FileHandler
logging.set_verbosity_info()
logger = logging.get_logger(__name__)

class Task:

    def __init__(self, inference: InferenceBase, file_handler: FileHandler, meta_file_path: str):
        self.inference = inference
        self.file_handler = file_handler
        self.meta_file_path = meta_file_path

    @classmethod
    def from_inference(cls, inference: InferenceBase, file_handler: FileHandler, meta_file_path: str) -> 'Task':
        return cls(inference, file_handler, meta_file_path)

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

    def _update_metadata(self, generation_dict: Dict[str, Any]) -> None:
        generations = self._load_metadata()
        for i, gen in enumerate(generations):
            if gen['request_id'] == generation_dict['request_id']:
                generations[i] = generation_dict
                break
        self._save_metadata(generations)

class Manager:
    def __init__(self):
        self.task: Task = None

    def generation_run_loop(self):
        while True:
            try:
                if self.task is not None:
                    logger.info(f"generation id{self.task.inference.generation.request_id} is created, "
                                "now running")
                    self.task.inference.run_inference(status_update=self.task.inference.generation.update_status)
                    self.task.inference.generation.status = InferencePhase.COMPLETED
            except Exception as e:
                logger.error(f"TaskManager task_run_loop error: {e}")
                if self.task is not None:
                    self.task.inference.generation.status = InferencePhase.FAILED
                    logger.error(f"generation id{self.task.inference.generation.request_id} running failed,")
            finally:
                self.task._update_metadata(self.task.inference.generation.to_dict())
                self.task = None

            sleep(0.5)

    def add_inference_task(self, task: Task) -> bool:
        if self.task is None:
            self.task = task
            logger.info(f"Added generation id{task.inference.generation.request_id} to current inference")
            generations = self.task._load_metadata()
            generations.append(task.inference.generation.to_dict())
            self.task._save_metadata(generations)
            return True
        else:
            logger.warning(f"Cannot add generation id{task.inference.generation.request_id}, another "
                           f"inference {self.task.inference.generation.request_id} is in progress")
            return False

    def get_current_generation(self) -> Dict[str, Any]:
        if self.task is not None and self.task.inference is not None:
            return self.task.inference.generation.to_dict()
        return None


gm = Manager()
threading.Thread(target=gm.generation_run_loop, daemon=True).start()
