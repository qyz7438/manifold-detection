from __future__ import annotations

import torch
from torch import nn


class ResidualBBoxPredictorAdapter(nn.Module):
    """Wrap a Fast R-CNN predictor with zero-initialized residual adapters."""

    def __init__(
        self,
        base_predictor: nn.Module,
        hidden_dim: int = 128,
        scale: float = 1.0,
        *,
        enable_cls_adapter: bool = False,
        cls_scale: float = 1.0,
    ):
        super().__init__()
        if not hasattr(base_predictor, "cls_score") or not hasattr(base_predictor, "bbox_pred"):
            raise AttributeError("base_predictor must expose cls_score and bbox_pred")
        self.base_predictor = base_predictor
        in_features = int(base_predictor.bbox_pred.in_features)
        out_features = int(base_predictor.bbox_pred.out_features)
        cls_out_features = int(base_predictor.cls_score.out_features)
        hidden_dim = max(1, int(hidden_dim))
        self.scale = float(scale)
        self.cls_scale = float(cls_scale)
        self.bbox_adapter = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_features),
        )
        nn.init.zeros_(self.bbox_adapter[-1].weight)
        nn.init.zeros_(self.bbox_adapter[-1].bias)
        self.cls_adapter = None
        if enable_cls_adapter:
            self.cls_adapter = nn.Sequential(
                nn.Linear(in_features, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, cls_out_features),
            )
            nn.init.zeros_(self.cls_adapter[-1].weight)
            nn.init.zeros_(self.cls_adapter[-1].bias)

    @property
    def cls_score(self) -> nn.Module:
        return self.base_predictor.cls_score

    @property
    def bbox_pred(self) -> nn.Module:
        return self.base_predictor.bbox_pred

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        class_logits, box_regression = self.base_predictor(x)
        if self.cls_adapter is not None:
            class_logits = class_logits + self.cls_scale * self.cls_adapter(x)
        return class_logits, box_regression + self.scale * self.bbox_adapter(x)


def install_residual_bbox_adapter(
    model: nn.Module,
    *,
    hidden_dim: int = 128,
    scale: float = 1.0,
    enable_cls_adapter: bool = False,
    cls_scale: float = 1.0,
) -> ResidualBBoxPredictorAdapter:
    current = model.roi_heads.box_predictor
    if isinstance(current, ResidualBBoxPredictorAdapter):
        return current
    device = next(current.parameters()).device
    adapter = ResidualBBoxPredictorAdapter(
        current,
        hidden_dim=hidden_dim,
        scale=scale,
        enable_cls_adapter=enable_cls_adapter,
        cls_scale=cls_scale,
    )
    adapter.to(device)
    model.roi_heads.box_predictor = adapter
    return adapter


def freeze_bbox_adapter_only(model: nn.Module) -> list[str]:
    return freeze_selected_adapters(model, train_bbox_adapter=True, train_cls_adapter=False)


def freeze_selected_adapters(
    model: nn.Module,
    *,
    train_bbox_adapter: bool,
    train_cls_adapter: bool,
) -> list[str]:
    for parameter in model.parameters():
        parameter.requires_grad = False
    predictor = model.roi_heads.box_predictor
    if not isinstance(predictor, ResidualBBoxPredictorAdapter):
        raise TypeError("model.roi_heads.box_predictor is not a ResidualBBoxPredictorAdapter")

    trainable_names = []
    if train_bbox_adapter:
        for name, parameter in predictor.bbox_adapter.named_parameters():
            parameter.requires_grad = True
            trainable_names.append(f"roi_heads.box_predictor.bbox_adapter.{name}")
    if train_cls_adapter:
        if predictor.cls_adapter is None:
            raise RuntimeError("Classifier adapter was requested but is not installed.")
        for name, parameter in predictor.cls_adapter.named_parameters():
            parameter.requires_grad = True
            trainable_names.append(f"roi_heads.box_predictor.cls_adapter.{name}")
    if not trainable_names:
        raise RuntimeError("No adapter parameters were selected for training.")
    return trainable_names


def freeze_adapters_and_predictor(
    model: nn.Module,
    *,
    train_bbox_adapter: bool = True,
    train_cls_adapter: bool,
    train_cls_score: bool = True,
    train_bbox_pred: bool = True,
) -> list[str]:
    for parameter in model.parameters():
        parameter.requires_grad = False
    predictor = model.roi_heads.box_predictor
    if not isinstance(predictor, ResidualBBoxPredictorAdapter):
        raise TypeError("model.roi_heads.box_predictor is not a ResidualBBoxPredictorAdapter")

    trainable_names = []
    if train_bbox_adapter:
        for name, parameter in predictor.bbox_adapter.named_parameters():
            parameter.requires_grad = True
            trainable_names.append(f"roi_heads.box_predictor.bbox_adapter.{name}")
    if train_cls_adapter:
        if predictor.cls_adapter is None:
            raise RuntimeError("Classifier adapter was requested but is not installed.")
        for name, parameter in predictor.cls_adapter.named_parameters():
            parameter.requires_grad = True
            trainable_names.append(f"roi_heads.box_predictor.cls_adapter.{name}")
    if train_cls_score:
        for name, parameter in predictor.base_predictor.cls_score.named_parameters():
            parameter.requires_grad = True
            trainable_names.append(f"roi_heads.box_predictor.base_predictor.cls_score.{name}")
    if train_bbox_pred:
        for name, parameter in predictor.base_predictor.bbox_pred.named_parameters():
            parameter.requires_grad = True
            trainable_names.append(f"roi_heads.box_predictor.base_predictor.bbox_pred.{name}")
    if not trainable_names:
        raise RuntimeError("No parameters were selected for training.")
    return trainable_names
