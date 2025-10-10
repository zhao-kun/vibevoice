import torch

def rand_seed(seeds: int) -> torch.Generator:
    return torch.Generator(device='cpu').manual_seed(seeds)


#random_generator = rand_seed(42)


