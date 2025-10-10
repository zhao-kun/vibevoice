import torch

from typing import Optional

_random_generator: Optional[torch.Generator] = None

def get_generator(seeds: int = 42) -> torch.Generator:
    global _random_generator
    
    def rand_seed() -> torch.Generator:
        torch.manual_seed(seeds)
        return torch.Generator(device='cpu').manual_seed(seeds)

    if _random_generator is None:
        _random_generator = rand_seed()
    return _random_generator

