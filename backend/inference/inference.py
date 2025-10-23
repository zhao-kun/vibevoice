import os
import re
from typing import List, Tuple
import time
import torch

from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalInference
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
from transformers.utils import logging
from config.configuration_vibevoice import VibeVoiceConfig, DEFAULT_CONFIG
from backend.models.generation import Generation
from backend.services.speaker_service import SpeakerService
from backend.services.dialog_session_service import DialogSessionService
from util.rand_init import get_generator

logging.set_verbosity_info()
logger = logging.get_logger(__name__)


class InferenceEngine:
    def __init__(self, generation: Generation, speaker_service: SpeakerService,
                 dialog_service: DialogSessionService):
        self.generation = generation
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.speaker_service = speaker_service
        self.dialog_service = dialog_service
        get_generator(generation.seeds)


    def _load_model(self, model_file: str = None,
                    config: str = None,
                    dtype: torch.dtype = torch.bfloat16,
                    attn_implementation: str = "sdpa"):
        config_dict = {}
        if config:
            with open(config, 'r') as f:
                import json
                config_dict = json.load(f)
        else:
            # Use default configuration
            print(f"Using default configuration: {DEFAULT_CONFIG}")
            config_dict = DEFAULT_CONFIG

        config = VibeVoiceConfig.from_dict(config_dict,
                                           torch_dtype=dtype,
                                           device_map="cuda",
                                           attn_implementation=attn_implementation)
        # Load model with device-specific logic
        model = VibeVoiceForConditionalInference.from_pretrain(model_file, config)
        return model

    def run_inference(self):
        txt_content, scripts, unique_speaker_names = self.dialog_service.parse_session_txt_script(self.generation.dialog_session_id)
        if not scripts or not unique_speaker_names:
            raise RuntimeError("No scripts, speaker_numbers found for the specified dialog session.")

        self.speaker_service.get_speakers_filepath(unique_speaker_names)

        # Proceed with inference using txt_content, scripts, and speaker_numbers
        # ...
