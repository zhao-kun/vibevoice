import torch

def rand_seed(seeds: int) -> torch.Generator:
    return torch.Generator(device='cuda').manual_seed(42)


random_generator = rand_seed(42)


