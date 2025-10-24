import os
import re
from typing import Union
import time
import torch

from abc import ABC, abstractmethod
from flask import current_app
from pathlib import Path
from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalInference, VibeVoiceGenerationOutput
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
from transformers.utils import logging
from config.configuration_vibevoice import VibeVoiceConfig, DEFAULT_CONFIG
from backend.models.generation import Generation, InferencePhase, UpdateStatusCallable
from backend.services.speaker_service import SpeakerService
from backend.services.dialog_session_service import DialogSessionService
from util.rand_init import get_generator

logging.set_verbosity_info()
logger = logging.get_logger(__name__)

class FakeModel:
    def __init__(self):
        pass

    def generate(self, update_status: UpdateStatusCallable, **kwargs) -> Union[torch.LongTensor, VibeVoiceGenerationOutput]:
        for i in range(100):
            time.sleep(0.5)  # Simulate some processing time
            update_status(InferencePhase.INFERENCING, current=i + 1, total_step=100)

        return torch.randn(1, 16000 * 5)  # Simulate 5 seconds of audio at 16kHz

class InferenceBase(ABC):
    def __init__(self, generation: Generation, speaker_service: SpeakerService,
                 dialog_service: DialogSessionService):
        self.generation = generation
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.speaker_service = speaker_service
        self.dialog_service = dialog_service
        self.model_file = current_app.config['MODEL_FILE_PATH']

    @staticmethod
    def create(generation: Generation, speaker_service: SpeakerService,
               dialog_service: DialogSessionService, meta_file_path: str, fake: bool = False) -> 'InferenceBase':

        if fake:
            return FakeInferenceEngine(generation, speaker_service, dialog_service, meta_file_path)

        return InferenceEngine(generation, speaker_service, dialog_service, meta_file_path)

    @abstractmethod
    def _load_model(self, dtype: torch.dtype, config: str = None):
        pass

    @abstractmethod
    def _save_audio(self, outputs: Union[torch.LongTensor, VibeVoiceGenerationOutput],
                    processor: VibeVoiceProcessor, update_status: UpdateStatusCallable) -> None:
        pass

    def run_inference(self, status_update: UpdateStatusCallable = lambda phase, *args, **kwargs: None):

        get_generator(self.generation.seeds)

        txt_content, scripts, unique_speaker_names = self.dialog_service.parse_session_txt_script(self.generation.dialog_session_id)
        if not scripts or not unique_speaker_names:
            raise RuntimeError("No scripts, speaker_numbers found for the specified dialog session.")

        voice_sample = self.speaker_service.get_speakers_filepath(unique_speaker_names)
        logger.info(f"Loaded voice samples for speakers: {unique_speaker_names}")

        # Proceed with inference using txt_content, scripts, and speaker_numbers
        # ...
        full_script = '\n'.join(scripts)
        full_script = full_script.replace("’", "'")

        status_update(InferencePhase.PREPROCESSING, scripts=scripts,
                      unique_speaker_names=unique_speaker_names,
                      voice_sample=voice_sample)

        processor = VibeVoiceProcessor.from_pretrained(None)
        inputs = processor(text=[full_script],
                           voice_samples=[voice_sample],
                           padding=True,
                           return_tensors="pt",
                           return_attention_mask=True)

        for k, v in inputs.items():
            inputs[k] = v.to(self.device)

        load_dtype = torch.bfloat16
        if self.generation.model_dtype == "float8_e4m3fn":
            load_dtype = torch.float8_e4m3fn
        logger.info(f"Loading model with dtype: {load_dtype}, "
                    f"attn_implementation: {self.generation.attn_implementation}")

        # only for testing purpose, load a fake model
        model = self._load_model(dtype=load_dtype)

        start_time = time.time()
        outputs = model.generate(**inputs,
                                 max_new_tokens=None,
                                 cfg_scale=self.generation.cfg_scale,
                                 tokenizer=processor.tokenizer,
                                 generation_config={'do_sample': False},
                                 verbose=True)

        status_update(InferencePhase.SAVING_AUDIO, generation_time=f"{time.time() - start_time:.2f} seconds")
        self.save_audio(outputs, processor, status_update)


class InferenceEngine(InferenceBase):
    def __init__(self, generation, speaker_service, dialog_service, meta_file_path: str):
        super().__init__(generation, speaker_service, dialog_service, meta_file_path)

    def _load_model(self, dtype: torch.dtype, config: str = None):
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
                                           attn_implementation=self.generation.attn_implementation)
        # Load model with device-specific logic
        model_file = Path(self.model_file) / Path(f"vibevoice7b_{'bf16' if dtype == torch.bfloat16 else 'float8_e4m3fn'}.safetensors")
        model = VibeVoiceForConditionalInference.from_pretrain(str(model_file.resolve()), config)
        model.to(self.device)
        model.eval()
        return model


class FakeInferenceEngine(InferenceBase):
    def __init__(self, generation, speaker_service, dialog_service, meta_file_path: str):
        super().__init__(generation, speaker_service, dialog_service, meta_file_path)

    def _load_model(self, dtype: torch.dtype, config: str = None):
        return FakeModel()

    def _save_audio(self, outputs: Union[torch.LongTensor, VibeVoiceGenerationOutput], 
                    processor: VibeVoiceProcessor, status_update: UpdateStatusCallable) -> None:
        output_audio_path = current_app.config['AUDIO_OUTPUT_DIR'] / Path(self.generation.output_filename)
        status_update(InferencePhase.SAVING_AUDIO, output_audio_path=str(output_audio_path))
        logger.info(f"Saving generated audio to {output_audio_path}")
        return

