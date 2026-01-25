from typing import List, Optional, Tuple, Union, Callable, Any
import torch
import torch.nn as nn

from accelerate import init_empty_weights
from transformers.generation import GenerationConfig, LogitsProcessor, LogitsProcessorList, StoppingCriteriaList
from transformers.generation.utils import GenerateOutput, GenerationMixin
from transformers.modeling_outputs import CausalLMOutput, BaseModelOutputWithPast
from transformers import modeling_utils

from util.model_utils import merge_lora_weights
from util.logger import get_logger
from config.configuration_vibevoice import VibeVoiceASRConfig

from .modular_vibevoice_qwen import QwenModel, QwenConfig
from .modeling_vibevoice import VibeVoiceCausalLMOutputWithPast, SpeechConnector
from .modular_vibevoice_tokenizer import (
    VibeVoiceSemanticTokenizerModel, VibeVoiceAcousticTokenizerModel, 
    VibeVoiceTokenizerStreamingCache, VibeVoiceTokenizerEncoderOutput
)


logger = get_logger(__name__)

if not hasattr(modeling_utils, "ALL_PARALLEL_STYLES") or modeling_utils.ALL_PARALLEL_STYLES is None:
    modeling_utils.ALL_PARALLEL_STYLES = ["tp", "none", "colwise", "rowwise"]

# @auto_docstring
class VibeVoiceASRPreTrainedModel(nn.Module):
    config_class = VibeVoiceASRConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _skip_keys_device_placement = "past_key_values"
    _supports_cache_class = True
    _supports_flash_attn = True
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_quantized_cache = True
    _supports_static_cache = True
    _supports_attention_backend = True

    def _init_weights(self, module):

        # Use the language model's initializer_range if available
        if hasattr(self.config, 'language_model_config') and hasattr(self.config.language_model_config, 'initializer_range'):
            std = self.config.language_model_config.initializer_range
        elif hasattr(self.config, 'decoder_config') and hasattr(self.config.decoder_config, 'initializer_range'):
            std = self.config.decoder_config.initializer_range
        else:
            std = 0.02  # Default value

        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()

# @auto_docstring
class VibeVoiceASRModel(VibeVoiceASRPreTrainedModel):
    def __init__(self, config):
        super().__init__()

        if hasattr(config, 'torch_dtype') and config.torch_dtype is not None:
            if isinstance(config.torch_dtype, str):
                dtype = getattr(torch, config.torch_dtype)
            else:
                dtype = config.torch_dtype
        else:
            dtype = torch.float32

        # Initialize Qwen2 model for language modeling
        lm_config = config.decoder_config 
        lm_config = QwenConfig.from_config(lm_config)
        self.language_model = QwenModel(lm_config, dtype=dtype)

        # Initialize speech components if needed
        self.acoustic_tokenizer = VibeVoiceAcousticTokenizerModel(config.acoustic_tokenizer_config, dtype=dtype).to(dtype)
        self.semantic_tokenizer = VibeVoiceSemanticTokenizerModel(config.semantic_tokenizer_config, dtype=dtype).to(dtype)

        self.acoustic_connector = SpeechConnector(config.acoustic_vae_dim, lm_config.hidden_size, dtype=dtype).to(dtype)
        self.semantic_connector = SpeechConnector(config.semantic_vae_dim, lm_config.hidden_size, dtype=dtype).to(dtype)

    def get_input_embeddings(self):
        if hasattr(self.language_model, 'embed_tokens'):
            # If the language model has an embed_tokens attribute, return it
            return self.language_model.embed_tokens

        for name, attr in self.language_model.fullmap.items():  # parallel by nnscaler, the name is changed
            if attr.orig_name == 'embed_tokens.weight':
                return getattr(self.language_model, name)
        assert False, 'should not arrive here'

    def set_input_embeddings(self, value):
        self.language_model.embed_tokens = value

    def set_speech_tokenizers(self, acoustic_tokenizer=None, semantic_tokenizer=None):
        """Set the speech tokenizers used for encoding and decoding speech."""
        self.acoustic_tokenizer = acoustic_tokenizer
        self.semantic_tokenizer = semantic_tokenizer

        # Reset the encoder to evaluation mode
        if self.acoustic_tokenizer is not None:
            self.acoustic_tokenizer.eval()

        if self.semantic_tokenizer is not None:
            self.semantic_tokenizer.eval()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Forward through language model
        outputs = self.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )

        if not return_dict:
            return outputs

        return BaseModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class VibeVoiceASRForConditionalGeneration(VibeVoiceASRPreTrainedModel, GenerationMixin):
    """
    VibeVoice model for Automatic Speech Recognition (ASR) with language modeling head for conditional generation.
    This class is designed for inference and generation tasks.
    """
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _is_stateful = False

    def __init__(self, config):
        super().__init__()
        self.model = VibeVoiceASRModel(config)
        self.vocab_size = config.decoder_config.vocab_size

        # Determine the dtype to use
        if hasattr(config, 'torch_dtype') and config.torch_dtype is not None:
            if isinstance(config.torch_dtype, str):
                dtype = getattr(torch, config.torch_dtype)
            else:
                dtype = config.torch_dtype
        else:
            dtype = torch.float32

        # Initialize lm_head with the correct dtype
        self.lm_head = nn.Linear(config.decoder_config.hidden_size, self.vocab_size, bias=False).to(dtype)
        self.generation_config = GenerationConfig.from_model_config(config)
        self.config = config
        self.main_input_name = "input_ids"
        self.device = torch.device("cuda")
        # Initialize weights and apply final processing
        #self.post_init()
    

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model.language_model = decoder

    def get_decoder(self):
        return self.model.language_model

    def tie_weights(self):
        """Tie the weights between the input embeddings and the output embeddings."""
        if getattr(self.config.decoder_config, 'tie_word_embeddings', False):
            output_embeddings = self.get_output_embeddings()
            input_embeddings = self.get_input_embeddings()
            if hasattr(input_embeddings, 'weight'):
                output_embeddings.weight = input_embeddings.weight
            else:
                output_embeddings.weight = input_embeddings

    def encode_speech(
        self,
        speech_tensors: torch.FloatTensor,
        speech_masks: Optional[torch.BoolTensor] = None,
        speech_semantic_tensors: Optional[torch.FloatTensor] = None,
        streaming_segment_duration: float = 60.0,  # seconds
    ):
        """
        Encode speech input into features that can be used by the language model.
        This method is called once before generation to process the speech input.

        For long audio (>600s by default), uses streaming processing to avoid conv overflow (>2^32).
        Segments are processed independently, then concatenated before final sampling.

        Args:
            speech_tensors: Input audio tensor [batch_size, samples]
            speech_masks: Optional mask for speech features
            speech_semantic_tensors: Optional pre-computed semantic tokens
            streaming_segment_duration: Segment duration in seconds for streaming processing (default: 60s)
        """
        if hasattr(self.config, 'torch_dtype') and self.config.torch_dtype is not None:
            if isinstance(self.config.torch_dtype, str):
                dtype = getattr(torch, self.config.torch_dtype)
            else:
                dtype = self.config.torch_dtype
        else:
            dtype = torch.float32

        speech_tensors = speech_tensors.to(dtype)

        # Ensure proper shape: (batch, samples)
        if speech_tensors.ndim == 1:
            speech_tensors = speech_tensors.unsqueeze(0)

        batch_size, total_samples = speech_tensors.shape
        sample_rate = 24000  # fix 24kHz sample rate

        # Calculate segment size in samples
        segment_samples = int(streaming_segment_duration * sample_rate)

        # Decide whether to use streaming based on audio length
        use_streaming = total_samples > segment_samples

        with torch.no_grad():
            if not use_streaming:
                # Short audio: direct processing (original behavior)
                encoder_output = self.model.acoustic_tokenizer.encode(speech_tensors.unsqueeze(1))
                audio_tokens = encoder_output.sample(dist_type=self.model.acoustic_tokenizer.std_dist_type)[0]
                acoustic_features = self.model.acoustic_connector(audio_tokens)

                # Encode semantic features
                if speech_semantic_tensors is not None:
                    semantic_features = self.model.semantic_connector(speech_semantic_tensors)
                else:
                    semantic_tokens = self.model.semantic_tokenizer.encode(speech_tensors.unsqueeze(1)).mean
                    semantic_features = self.model.semantic_connector(semantic_tokens)
            else:
                # Long audio: streaming processing
                # print(f"Using streaming processing for long audio: {total_samples/sample_rate:.1f}s "
                #       f"(segment size: {streaming_segment_duration}s)")

                # Initialize caches for both tokenizers
                acoustic_encoder_cache = VibeVoiceTokenizerStreamingCache()
                semantic_encoder_cache = VibeVoiceTokenizerStreamingCache()
                acoustic_mean_segments = []
                semantic_mean_segments = []
                sample_indices = torch.arange(batch_size, device=speech_tensors.device)

                # Helper function from batch_asr_sft_cache.py
                def _iter_segments(total_length: int, segment_length: int):
                    """Iterate over audio segments with a given segment length."""
                    if segment_length <= 0:
                        raise ValueError("segment_length must be positive")
                    for start in range(0, total_length, segment_length):
                        end = min(start + segment_length, total_length)
                        if end > start:
                            yield start, end

                # Process each segment for both acoustic and semantic tokenizers
                segments = list(_iter_segments(total_samples, segment_samples))
                num_segments = len(segments)
                for seg_idx, (start, end) in enumerate(segments):
                    chunk = speech_tensors[:, start:end].contiguous()
                    if chunk.numel() == 0:
                        continue

                    # Check if this is the final segment
                    is_final = (seg_idx == num_segments - 1)

                    # Encode chunk for acoustic tokenizer (don't sample yet)
                    acoustic_encoder_output = self.model.acoustic_tokenizer.encode(
                        chunk.unsqueeze(1),
                        cache=acoustic_encoder_cache,
                        sample_indices=sample_indices,
                        use_cache=True,
                        is_final_chunk=is_final,
                    )
                    acoustic_mean_segments.append(acoustic_encoder_output.mean)

                    # Encode chunk for semantic tokenizer (take mean directly)
                    semantic_encoder_output = self.model.semantic_tokenizer.encode(
                        chunk.unsqueeze(1),
                        cache=semantic_encoder_cache,
                        sample_indices=sample_indices,
                        use_cache=True,
                        is_final_chunk=is_final,
                    )
                    semantic_mean_segments.append(semantic_encoder_output.mean)

                # print(f"Processed {len(acoustic_mean_segments)} segments.")
                # Concatenate all acoustic means and sample once
                acoustic_mean_full = torch.cat(acoustic_mean_segments, dim=1).contiguous()
                acoustic_encoder_output = VibeVoiceTokenizerEncoderOutput(
                    mean=acoustic_mean_full,
                    std=self.model.acoustic_tokenizer.fix_std
                )
                audio_tokens = acoustic_encoder_output.sample(
                    dist_type=self.model.acoustic_tokenizer.std_dist_type
                )[0]
                acoustic_features = self.model.acoustic_connector(audio_tokens)

                # Concatenate all semantic means
                semantic_tokens = torch.cat(semantic_mean_segments, dim=1).contiguous()
                semantic_features = self.model.semantic_connector(semantic_tokens)

            # Combine acoustic and semantic features
            if speech_masks is not None:
                combined_features = acoustic_features[speech_masks] + semantic_features[speech_masks]
            else:
                combined_features = acoustic_features + semantic_features

        return combined_features

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        # Speech-specific arguments
        speech_tensors: Optional[torch.FloatTensor] = None,
        speech_masks: Optional[torch.BoolTensor] = None,
        speech_semantic_tensors: Optional[torch.FloatTensor] = None,
        acoustic_input_mask: Optional[torch.BoolTensor] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutput]:
        """
        Forward pass for the model. Handles both training and generation scenarios.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        # Process inputs
        if inputs_embeds is None and input_ids is not None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        # If we have speech input and acoustic_input_mask, encode and insert speech features
        if speech_tensors is not None and acoustic_input_mask is not None:
            speech_features = self.encode_speech(
                speech_tensors=speech_tensors,
                speech_masks=speech_masks,
                speech_semantic_tensors=speech_semantic_tensors,
            )
            inputs_embeds[acoustic_input_mask] = speech_features

        # Forward through the model
        outputs = self.model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0] if not return_dict else outputs.last_hidden_state
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return VibeVoiceCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        speech_tensors=None,
        speech_masks=None,
        speech_semantic_tensors=None,
        acoustic_input_mask=None,
        **kwargs,
    ):
        """
        Prepare inputs for generation step. This method is called by generate() 
        for each token generation step.

        Following Qwen2-VL's approach: speech inputs are only forwarded on the first pass
        (when cache_position[0] == 0), and are excluded in subsequent generation steps.
        """
        # If we have past key values, we only need to process the new tokens
        if past_key_values is not None:
            if isinstance(past_key_values, tuple):
                past_length = past_key_values[0][0].shape[2]
            else:
                past_length = past_key_values.get_seq_length()

            # Keep only the new tokens
            if input_ids is not None and input_ids.shape[1] > past_length:
                input_ids = input_ids[:, past_length:]

        # Prepare position ids
        if position_ids is None and attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values is not None and input_ids is not None:
                position_ids = position_ids[:, -input_ids.shape[1]:]

        # Prepare cache position
        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + (input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]),
                device=input_ids.device if input_ids is not None else inputs_embeds.device
            )

        # Prepare model inputs
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
            }
        )

        # Following Qwen2-VL pattern: only include speech inputs on the first forward pass
        # (when cache_position[0] == 0), exclude them in subsequent generation steps
        if cache_position is not None and len(cache_position) > 0 and cache_position[0] == 0:
            # First forward pass - include speech inputs if provided
            model_inputs.update({
                "speech_tensors": speech_tensors,
                "speech_masks": speech_masks,
                "speech_semantic_tensors": speech_semantic_tensors,
                "acoustic_input_mask": acoustic_input_mask,
            })
        else:
            # Subsequent generation steps - exclude speech inputs
            model_inputs.update({
                "speech_tensors": None,
                "speech_masks": None,
                "speech_semantic_tensors": None,
                "acoustic_input_mask": None,
            })

        # Include any remaining kwargs that might be needed
        model_inputs.update(kwargs)
        return model_inputs
    
    # @torch.no_grad()
    # def generate(
    #     self,
    #     inputs: Optional[torch.Tensor] = None,
    #     generation_config: Optional[GenerationConfig] = None,
    #     logits_processor: Optional[LogitsProcessorList] = None,
    #     stopping_criteria: Optional[StoppingCriteriaList] = None,
    #     prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], list[int]]] = None,
    #     synced_gpus: Optional[bool] = None,
    #     negative_prompt_ids: Optional[torch.Tensor] = None,
    #     negative_prompt_attention_mask: Optional[torch.Tensor] = None,
    #     use_model_defaults: Optional[bool] = None,
    #     custom_generate: Optional[Union[str, Callable]] = None,
    #     **kwargs,
    # ) -> Tuple[GenerateOutput, torch.Tensor]:
    #     generation_config, model_kwargs = self._prepare_for_generation(generation_config, use_model_defaults, **kwargs)
    #     pass

    # def _prepare_generation_config(
    #     self,
    #     generation_config: Optional[GenerationConfig],
    #     use_model_defaults: Optional[bool] = None,
    #     **kwargs: Any,
    # ) -> tuple[GenerationConfig, dict]:
    #     """
    #     Prepares the base generation config, then applies any generation configuration options from kwargs. This
    #     function handles retrocompatibility with respect to configuration files.
    #     """
    #     # parameterization priority:
    #     # kwargs > non-global default values in `generation_config` > `model.generation_config` > GenerationConfig()
    #     # TODO (joao): per-model generation config classes.

    #     using_model_generation_config = False
    #     if generation_config is None:
    #         # legacy: users may modify the model configuration to control generation. To trigger this legacy behavior,
    #         # the following conditions must be met
    #         # 1) the generation config must have been created from the model config (`_from_model_config` field);
    #         # 2) the generation config must have seen no modification since its creation (the hash is the same);
    #         # 3) there are non-default generation parameters in the model config.
    #         # 4) the user must have set new generation parameters in the model config.
    #         if (
    #             self.generation_config._from_model_config  # 1)
    #             and self.generation_config._original_object_hash == hash(self.generation_config)  # 2)
    #             and len(self.config._get_non_default_generation_parameters()) > 0  # 3)
    #         ):
    #             new_generation_config = GenerationConfig.from_model_config(self.config)
    #             if new_generation_config != self.generation_config:  # 4)
    #                 warnings.warn(
    #                     "You have modified the pretrained model configuration to control generation. This is a"
    #                     " deprecated strategy to control generation and will be removed in v5."
    #                     " Please use and modify the model generation configuration (see"
    #                     " https://huggingface.co/docs/transformers/generation_strategies#default-text-generation-configuration )",
    #                     UserWarning,
    #                 )
    #                 self.generation_config = new_generation_config

    #         generation_config = self.generation_config
    #         using_model_generation_config = True

    #         # Related to #40039: prior to this PR, models with sliding window attention were forced to have
    #         # `cache_implementation="hybrid"` (the static sliding window cache). For these models, we now want to use
    #         # the dynamic sliding window cache by default, so we UNSET `cache_implementation` if it is a default value.
    #         # (if we're inside this branch, then it is because we're using default values from the Hub)
    #         if generation_config.cache_implementation == "hybrid":
    #             generation_config.cache_implementation = None

    #     # `torch.export.export` usually raises an exception if it is called
    #     # with ``strict=True``. deepcopy can only be processed if ``strict=False``.
    #     generation_config = copy.deepcopy(generation_config)

    #     if not using_model_generation_config:
    #         # If `generation_config` is provided:
    #         # - `use_model_defaults`: let's fallback ALL default values to the model's generation config
    #         # - otherwise: legacy behavior, let's just make sure we have the tokens defined
    #         model_base_version = version.parse(version.parse(self.generation_config.transformers_version).base_version)
    #         if use_model_defaults is True or (
    #             use_model_defaults is None and model_base_version >= version.parse("4.50.0")
    #         ):
    #             modified_values = {}
    #             global_default_generation_config = GenerationConfig()
    #             model_generation_config = self.generation_config
    #             # we iterate over the model's generation config: it may hold custom keys, which we'll want to copy
    #             for key, model_gen_config_value in model_generation_config.__dict__.items():
    #                 if key.startswith("_") or key == "transformers_version":  # metadata
    #                     continue
    #                 # Don't set `cache_implementation = 'hybrid'` from the model defaults, see #40135
    #                 if key == "cache_implementation" and model_generation_config.cache_implementation == "hybrid":
    #                     continue
    #                 global_default_value = getattr(global_default_generation_config, key, None)
    #                 custom_gen_config_value = getattr(generation_config, key, None)
    #                 if (
    #                     custom_gen_config_value == global_default_value
    #                     and model_gen_config_value != global_default_value
    #                 ):
    #                     modified_values[key] = model_gen_config_value
    #                     setattr(generation_config, key, model_gen_config_value)
    #             # edge case: we may set `temperature=0.0` and `do_sample=False`, but the model defaults to
    #             # `do_sample=True`
    #             if generation_config.temperature == 0.0:
    #                 generation_config.do_sample = False
    #             if use_model_defaults is None and len(modified_values) > 0:
    #                 logger.warning_once(
    #                     f"`generation_config` default values have been modified to match model-specific defaults: "
    #                     f"{modified_values}. If this is not desired, please set these values explicitly."
    #                 )
    #         else:
    #             if generation_config.bos_token_id is None:
    #                 generation_config.bos_token_id = self.generation_config.bos_token_id
    #             if generation_config.eos_token_id is None:
    #                 generation_config.eos_token_id = self.generation_config.eos_token_id
    #             if generation_config.pad_token_id is None:
    #                 generation_config.pad_token_id = self.generation_config.pad_token_id
    #             if generation_config.decoder_start_token_id is None:
    #                 generation_config.decoder_start_token_id = self.generation_config.decoder_start_token_id

    #     # Finally, apply any passed kwargs
    #     model_kwargs = generation_config.update(**kwargs)
    #     # And keep in model_kwargs variable output controls
    #     output_attentions = generation_config.output_attentions
    #     output_hidden_states = generation_config.output_hidden_states
    #     model_kwargs.update({"output_attentions": output_attentions} if output_attentions else {})
    #     model_kwargs.update({"output_hidden_states": output_hidden_states} if output_hidden_states else {})

    #     return generation_config, model_kwargs        

    @classmethod
    def from_pretrain(cls, model_path: str, config: VibeVoiceASRConfig, device="cuda", offload_config=None, lora_model_path: str = None, lora_weight: float = 1.0):
        """
        Load model from pretrained weights.

        Args:
            model_path: Path to model weights (directory or single file)
            config: Model configuration
            device: Target device (default: "cuda")
            offload_config: Optional OffloadConfig for layer offloading

        Returns:
            Loaded model with optional offloading enabled
        """
        import os
        from util.safetensors_util import MultipleSafetensorLoader, MemoryEfficientSafeOpen
        from vibevoice.modular.custom_offloading_utils import LayerOffloader

        with init_empty_weights():
            model = cls(config)

        state_dict = {}
        if os.path.isdir(model_path):
            print(f"Begin to load model from model path {model_path}")
            state_dict = MultipleSafetensorLoader(os.path.join(model_path, "model.safetensors.index.json")).load_dict()
        else:
            print(f"Begin to load model from mono model file {model_path}")
            with MemoryEfficientSafeOpen(model_path) as safe:
                for key in safe.keys():
                    state_dict[key] = safe.get_tensor(key)

        model.load_state_dict(state_dict, strict=False, assign=True)
        print("Model loaded")

        if lora_model_path is not None and lora_model_path != "":
            model = merge_lora_weights(model, lora_model_path, lora_weight)

        # Setup layer offloading if enabled
        if offload_config is not None and offload_config.enabled:
            print(f"Setting up layer offloading: {offload_config.num_layers_on_gpu} layers on GPU")

            # First, move all non-transformer components to GPU
            # (LayerOffloader only handles transformer layers)

            # Top-level components
            model.lm_head.to(device)

            # Language model components (except transformer layers)
            # Convert embedding from Float8 to BF16 to avoid OOM during forward pass
            if model.model.language_model.embed_tokens.weight.dtype == torch.float8_e4m3fn:
                print("Converting embedding layer from Float8 to BF16...")
                # Move to CPU first, convert, then move to GPU
                model.model.language_model.embed_tokens.cpu()
                embed_weight_data = model.model.language_model.embed_tokens.weight.data.to(torch.bfloat16)
                model.model.language_model.embed_tokens.weight.data = embed_weight_data

            model.model.language_model.embed_tokens.to(device)
            model.model.language_model.norm.to(device)

            # Speech processing components
            model.model.acoustic_tokenizer.to(device)
            model.model.semantic_tokenizer.to(device)
            model.model.acoustic_connector.to(device)
            model.model.semantic_connector.to(device)

            # Prediction head: conditionally offload to CPU
            if offload_config.offload_prediction_head:
                model.model.prediction_head.cpu()
                print("Prediction head offloaded to CPU (will transfer on-demand)")
            else:
                model.model.prediction_head.to(device)

            # Now setup layer offloading for transformer layers only
            model.offloader = LayerOffloader(
                language_model=model.model.language_model,
                config=offload_config,
                device=torch.device(device),
                logger=logger
            )
            print(f"Layer offloading enabled: {len(model.offloader.offloaded_layers)} layers offloaded")
        else:
            # No offloading - move entire model to device
            model.to(device)
            print(f"Model moved to {device} (no offloading)")

        return model


__all__ = [
    "VibeVoiceASRPreTrainedModel",
    "VibeVoiceASRModel",
    "VibeVoiceASRForConditionalGeneration",
]
