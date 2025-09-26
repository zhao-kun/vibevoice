from pyexpat import model
import torch
from transformers import Qwen2Tokenizer

from vibevoice.modular.modular_vibevoice_qwen import Qwen2ForCausalLM
from config.configuration_vibevoice import QwenConfig


def inference(model_path: str, model_name: str="Qwen/Qwen2-0.5B", config: QwenConfig=None):

    model_name = "Qwen/Qwen2-0.5B"
    tokenizer = Qwen2Tokenizer.from_pretrained(model_name)

    model = Qwen2ForCausalLM.from_pretrained(model_path, config)
    model.to("cuda")

    # Prompt
    prompt = "Once upon a time"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    # Initialize sequence with prompt
    input_ids = inputs["input_ids"]

    # Greedy decoding loop
    max_new_tokens = 350
    for loop in range(max_new_tokens):
        logits = model.forward(input_ids=input_ids)
        next_token_logits = logits[:, -1, :]  # last step
        next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
        
        # Append prediction
        input_ids = torch.cat([input_ids, next_token_id], dim=-1)
    
        # Stop if EOS token is generated
        if next_token_id.item() == tokenizer.eos_token_id:
            break
    
    # Decode to string
    generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    print(generated_text)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Memory-efficient safetensors reader example")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model directory", default="./models/qwen2-0.5b/model.safetensors")
    parser.add_argument("--model_name", type=str, required=False, help="Model name", default="Qwen/Qwen2-0.5B")
    args = parser.parse_args()
    config = QwenConfig(
        attention_dropout=0.0,
        bos_token_id=151643,
        eos_token_id=151643,
        hidden_act="silu",
        hidden_size=896,
        initializer_range=0.02,
        intermediate_size=4864,
        max_position_embeddings=131072,
        max_window_layers=24,
        num_attention_heads=14,
        num_hidden_layers=24,
        num_key_value_heads=2,
        rms_norm_eps=1e-06,
        rope_theta=1000000.0,
        sliding_window=131072,
        torch_dtype="bfloat16",
        use_cache=True,
        use_sliding_window= False,
        vocab_size=151936
    )
    inference(args.model_path, model_name=args.model_name, config=config)