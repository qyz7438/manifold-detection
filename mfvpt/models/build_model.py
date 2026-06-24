from __future__ import annotations

import torch
import timm


def build_model(config: dict) -> torch.nn.Module:
    model_cfg = config["model"]
    try:
        return timm.create_model(
            model_cfg["name"],
            pretrained=bool(model_cfg.get("pretrained", True)),
            num_classes=int(model_cfg.get("num_classes", 100)),
        )
    except Exception:
        if model_cfg.get("allow_random_init_fallback", False):
            return timm.create_model(
                model_cfg["name"],
                pretrained=False,
                num_classes=int(model_cfg.get("num_classes", 100)),
            )
        raise


def normalize_for_imagenet(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x - mean) / std
