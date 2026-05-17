import random

import numpy as np
import torch


def set_random_seed(seed: int = 0) -> None:
    """Set seeds across torch/numpy/random for reproducibility."""

    torch.manual_seed(seed + 0)
    torch.cuda.manual_seed(seed + 1)
    torch.cuda.manual_seed_all(seed + 2)
    np.random.seed(seed + 3)
    torch.cuda.manual_seed_all(seed + 4)
    random.seed(seed + 5)
