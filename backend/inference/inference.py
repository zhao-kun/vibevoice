import base64
from typing import Union
import time
import torch

from abc import ABC, abstractmethod
from flask import current_app
from pathlib import Path
from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalInference, VibeVoiceGenerationOutput
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
from transformers.utils import logging
from config.configuration_vibevoice import DEFAULT_CONFIG, VibeVoiceConfig, InferencePhase
from backend.models.generation import Generation, UpdateStatusCallable
from backend.services.speaker_service import SpeakerService
from backend.services.dialog_session_service import DialogSessionService
from util.rand_init import get_generator

logging.set_verbosity_info()
logger = logging.get_logger(__name__)

class FakeModel:
    def __init__(self):
        pass

    def generate(self, **kwargs) -> Union[torch.LongTensor, VibeVoiceGenerationOutput]:
        for i in range(100):
            time.sleep(0.5)  # Simulate some processing time
            kwargs.get("status_update", lambda phase, **kwargs: None)(InferencePhase.INFERENCING, current=i + 1, total_step=100)

        return torch.randn(1, 16000 * 5)  # Simulate 5 seconds of audio at 16kHz

class InferenceBase(ABC):
    def __init__(self, generation: Generation, speaker_service: SpeakerService,
                 dialog_service: DialogSessionService, meta_file_path: str):
        self.generation = generation
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.speaker_service = speaker_service
        self.dialog_service = dialog_service
        self.model_file = current_app.config['MODEL_PATH']
        self.meta_file_path = meta_file_path

    @staticmethod
    def create(generation: Generation, speaker_service: SpeakerService,
               dialog_service: DialogSessionService, meta_file_path: str, fake: bool = False,
               enable_offloading: bool = None, num_gpu_layers: int = None,
               offload_preset: str = None) -> 'InferenceBase':
        """
        Create inference engine instance.

        Args:
            generation: Generation object
            speaker_service: Speaker service
            dialog_service: Dialog service
            meta_file_path: Path to metadata
            fake: Use fake inference engine for testing
            enable_offloading: Enable layer offloading (default: auto-detect)
            num_gpu_layers: Number of layers on GPU (default: auto-detect)
            offload_preset: Use preset config ('high_end', 'mid_range', 'consumer', 'budget', 'minimal')

        Returns:
            InferenceBase instance
        """
        if fake:
            return FakeInferenceEngine(generation, speaker_service, dialog_service, meta_file_path)

        return InferenceEngine(generation, speaker_service, dialog_service, meta_file_path,
                             enable_offloading=enable_offloading,
                             num_gpu_layers=num_gpu_layers,
                             offload_preset=offload_preset)

    @abstractmethod
    def _load_model(self, dtype: torch.dtype, config: str = None):
        pass

    @abstractmethod
    def _save_audio(self, outputs: Union[torch.LongTensor, VibeVoiceGenerationOutput],
                    processor: VibeVoiceProcessor, update_status: UpdateStatusCallable,
                    generation_time: float, input_tokens: int, **kwargs) -> None:
        pass

    def run_inference(self, status_update: UpdateStatusCallable = lambda phase, *args, **kwargs: None):

        get_generator(self.generation.seeds)

        txt_content, scripts, unique_speaker_names = self.dialog_service.parse_session_txt_script(self.generation.session_id)
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
            if torch.is_tensor(v):
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
                                 verbose=False,
                                 status_update=status_update)

        generation_time = time.time() - start_time
        self._save_audio(outputs, processor, status_update, generation_time, 
                         inputs['input_ids'].shape[1],
                         unique_speaker_names=unique_speaker_names,
                         number_of_segments=len(scripts))


class InferenceEngine(InferenceBase):
    def __init__(self, generation, speaker_service, dialog_service, meta_file_path: str,
                 enable_offloading: bool = None, num_gpu_layers: int = None,
                 offload_preset: str = None):
        """
        Initialize inference engine with optional layer offloading.

        Args:
            generation: Generation object with parameters
            speaker_service: Service for managing speakers
            dialog_service: Service for managing dialogs
            meta_file_path: Path to metadata file
            enable_offloading: Enable layer offloading (default: auto-detect)
            num_gpu_layers: Number of layers to keep on GPU (default: auto-detect)
            offload_preset: Use preset config ('high_end', 'mid_range', 'consumer', 'budget', 'minimal')
        """
        super().__init__(generation, speaker_service, dialog_service, meta_file_path)

        # Offloading configuration
        self.enable_offloading = enable_offloading
        self.num_gpu_layers = num_gpu_layers
        self.offload_preset = offload_preset

    def _load_model(self, dtype: torch.dtype, config: str = None):
        from vibevoice.modular.adaptive_offload import AdaptiveOffloadManager
        from vibevoice.modular.custom_offloading_utils import OffloadConfig

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

        # Determine offload configuration
        offload_config = None
        use_float8 = dtype == torch.float8_e4m3fn

        if self.offload_preset:
            # Use preset configuration
            logger.info(f"Using offload preset: {self.offload_preset}")
            offload_config = AdaptiveOffloadManager.get_preset_config(self.offload_preset)
        elif self.enable_offloading or (self.enable_offloading is None and self.num_gpu_layers is not None):
            # Auto-configure based on available VRAM or use specified num_gpu_layers
            if self.num_gpu_layers is not None:
                logger.info(f"Manual offload config: {self.num_gpu_layers} layers on GPU")
                offload_config = OffloadConfig(
                    enabled=True,
                    num_layers_on_gpu=self.num_gpu_layers,
                    pin_memory=True,
                    prefetch_next_layer=True,
                    verbose=False
                )
            else:
                logger.info("Auto-detecting offload configuration...")
                offload_config = AdaptiveOffloadManager.auto_configure(
                    total_layers=28,
                    batch_size=1,
                    max_seq_len=4096,
                    use_float8=use_float8,
                    target_utilization=0.80,
                    device=self.device,
                    logger=logger
                )
        else:
            logger.info("Layer offloading disabled")

        # Print VRAM estimates
        if offload_config and offload_config.enabled:
            AdaptiveOffloadManager.print_vram_table(use_float8=use_float8, logger=logger)

        # Load model with device-specific logic
        model_file = Path(self.model_file) / Path(f"vibevoice7b_{'bf16' if dtype == torch.bfloat16 else 'float8_e4m3fn'}.safetensors")
        model = VibeVoiceForConditionalInference.from_pretrain(
            str(model_file.resolve()),
            config,
            device=self.device,
            offload_config=offload_config
        )

        # Print offloading statistics if enabled
        if model.offloader is not None:
            model.offloader.print_stats()

        model.eval()
        return model

    def _save_audio(self, outputs: Union[torch.LongTensor, VibeVoiceGenerationOutput],
                    processor: VibeVoiceProcessor, status_update: UpdateStatusCallable,
                    generation_time: float, input_tokens: int, **kwargs) -> None:
        if outputs.speech_outputs is None or len(outputs.speech_outputs) == 0:
            raise RuntimeError("No audio output generated.")

        sample_rate = 24000
        audio_samples = outputs.speech_outputs[0].shape[-1] \
            if len(outputs.speech_outputs[0].shape) > 0 else len(outputs.speech_outputs[0])
        audio_duration = audio_samples / sample_rate
        rtf = generation_time / audio_duration if audio_duration > 0 else float('inf')

        output_tokens = outputs.sequences.shape[1]  # Total tokens (input + generated)
        generated_tokens = output_tokens - input_tokens

        # Generate output filename and set it in the generation object
        output_filename = f"{self.generation.request_id}.wav"
        self.generation.output_filename = output_filename

        # Save output (processor handles device internally)
        output_audio_path = Path(self.generation.project_dir) / output_filename
        processor.save_audio(outputs.speech_outputs[0], output_path=output_audio_path)

        status_update(InferencePhase.SAVING_AUDIO,
                      output_audio_path=str(output_audio_path),
                      generation_time=generation_time,
                      prefilling_tokens=input_tokens,
                      total_tokens=output_tokens,
                      generated_tokens=generated_tokens,
                      audio_duration_seconds=audio_duration,
                      real_time_factor=rtf,
                      **kwargs)
        logger.info(f"Saving generated audio to {output_audio_path}")
        return

class FakeInferenceEngine(InferenceBase):
    def __init__(self, generation, speaker_service, dialog_service, meta_file_path: str):
        super().__init__(generation, speaker_service, dialog_service, meta_file_path)

    def _load_model(self, dtype: torch.dtype, config: str = None):
        return FakeModel()

    def _save_audio(self, outputs: Union[torch.LongTensor, VibeVoiceGenerationOutput],
                    processor: VibeVoiceProcessor, status_update: UpdateStatusCallable,
                    generation_time: float, input_tokens: int, **kwargs) -> None:
        base64_wav_audio = "UklGRiUAAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQEAAACA"  # Fake short audio for test
        audio_data = base64.b64decode(base64_wav_audio)

        # Generate output filename and set it in the generation object
        output_filename = f"{self.generation.request_id}.wav"
        self.generation.output_filename = output_filename

        output_audio_path = Path(self.generation.project_dir) / output_filename
        with open(output_audio_path, 'wb') as f:
            f.write(audio_data)
        status_update(InferencePhase.SAVING_AUDIO,
                      output_audio_path=str(output_audio_path),
                      prefilling_tokens=input_tokens,
                      generation_time=generation_time,
                      total_tokens=1024,                        # fake value
                      generated_tokens=1024,                    # fake value
                      audio_duration_seconds=5.0,               # fake value
                      real_time_factor=1.0,                     # fake value
                      **kwargs
                      )
        logger.info(f"Saving generated audio to {output_audio_path}")
        return
