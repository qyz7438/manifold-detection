from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resolve_device(config: dict) -> torch.device:
    requested = str(config.get("device", "cuda"))
    if requested in ("auto", "cuda"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)
