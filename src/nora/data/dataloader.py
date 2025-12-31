"""Deterministic dataloader utilities"""

import torch
from torch.utils.data import DataLoader


def get_worker_init_fn(seed: int):
    """Get worker initialization function for deterministic dataloading"""

    def init_fn(worker_id):
        import random
        import numpy as np

        random.seed(seed + worker_id)
        np.random.seed(seed + worker_id)
        torch.manual_seed(seed + worker_id)

    return init_fn
