import argparse
import os

import torch

from safetensors.torch import save_model

from config.configuration_vibevoice import VibeVoiceConfig
from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalInference


def parse_args():
    parser = argparse.ArgumentParser(description="VibeVoice Model Convert and Save")
    parser.add_argument("--model_path", type=str, required=True , help="Original model path, must 7B (large) version")
    parser.add_argument("--type", type=str, default="bfloat16", help="Data type for the new convert model, could be bfloat16 or torch.float8_e4m3fn")
    parser.add_argument("--converted_model_name", type=str, default="./vibvoice7b_", help="Path to the converted model file")
    args = parser.parse_args()
    return args

def main(): 
    args = parse_args()
    os.path.join(args.model_path, "config.json")
    with open(os.path.join(args.model_path, "config.json"), 'r') as f:
        import json
        config_dict = json.load(f)

    load_dtype = torch.float16
    attn_implementation = "sdpa"
 
    config = VibeVoiceConfig.from_dict(config_dict, 
                                       torch_dtype=load_dtype, 
                                       device_map="cuda", 
                                       attn_implementation=attn_implementation)


    # Load model with device-specific logic
    model = VibeVoiceForConditionalInference.from_pretrain(args.model_path, config)
    model.eval()
    target_dtype = None
    if args.type == "float8_e4m3fn":
        target_dtype = torch.float8_e4m3fn
        save_model_name = args.converted_model_name + "float8_e4m3fn.safetensors"
    elif args.type == "float8_e5m2":
        target_dtype = torch.float8_e5m2
        save_model_name = args.converted_model_name + "float8_e5m2.safetensors"
    else:
        target_dtype = torch.bfloat16
        save_model_name = args.converted_model_name + "bf16.safetensors"

    model.to(dtype=target_dtype)

    print(f"Model converted and saved to {save_model_name} with dtype {args.type}")
    save_model(model, save_model_name)



if __name__ == "__main__":
    main()