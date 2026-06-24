from __future__ import annotations

import torch


def resolve_device(config: dict) -> torch.device:
    requested = str(config.get("device", "cuda"))
    if requested == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def make_optimizer(parameters, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    return torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)


def set_trainable(model: torch.nn.Module, mode: str) -> None:
    if mode == "full":
        for param in model.parameters():
            param.requires_grad = True
        return
    if mode == "head_norm":
        for param in model.parameters():
            param.requires_grad = False
        for name, param in model.named_parameters():
            if "head" in name or "norm" in name:
                param.requires_grad = True
        return
    raise ValueError(f"Unknown trainable mode: {mode}")
