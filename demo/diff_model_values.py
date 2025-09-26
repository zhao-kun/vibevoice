import torch
import torch.nn as nn

def compare_model_weights(model1, model2, tolerance=1e-6):
    """Compare weights between two models"""
    state_dict1 = model1.state_dict()
    state_dict2 = model2.state_dict()
    
    differences = {}
    
    # Check if both models have same parameter names
    keys1, keys2 = set(state_dict1.keys()), set(state_dict2.keys())
    
    if keys1 != keys2:
        print(f"Different parameter names:")
        print(f"Only in model1: {keys1 - keys2}")
        print(f"Only in model2: {keys2 - keys1}")
    
    # Compare common parameters
    common_keys = keys1 & keys2
    
    for name in common_keys:
        param1, param2 = state_dict1[name], state_dict2[name]
        
        # Check shape compatibility
        if param1.shape != param2.shape:
            differences[name] = {
                'type': 'shape_mismatch',
                'shape1': param1.shape,
                'shape2': param2.shape
            }
            continue
        
        # Check if parameters are equal
        if torch.equal(param1, param2):
            continue
        
        # Calculate differences
        diff = param1 - param2
        abs_diff = torch.abs(diff)
        
        differences[name] = {
            'type': 'value_difference',
            'max_abs_diff': abs_diff.max().item(),
            'mean_abs_diff': abs_diff.mean().item(),
            'relative_diff': (abs_diff / (torch.abs(param1) + 1e-8)).mean().item(),
            'num_different': (abs_diff > tolerance).sum().item(),
            'total_elements': param1.numel()
        }
    
    return differences


def load_local_model():
    from vibevoice.modular.modular_vibevoice_qwen import Qwen2ForCausalLM, QwenConfig
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
    model = Qwen2ForCausalLM.from_pretrained("./models/qwen2-0.5b/model.safetensors", config)
    return model

def load_transformer_model():
    from transformers import Qwen2ForCausalLM
    model = Qwen2ForCausalLM.from_pretrained("Qwen/Qwen2-0.5B")
    return model

if __name__ == "__main__":
    model_local = load_local_model()
    model_transformer = load_transformer_model()
    differences = compare_model_weights(model_local, model_transformer, tolerance=1e-6)
    if len(differences) == 0:
        print("Models are identical.")
    else:    
        for name, diff in differences.items():
            for k, v in diff.items():
                print(f"  {k}: {v}")