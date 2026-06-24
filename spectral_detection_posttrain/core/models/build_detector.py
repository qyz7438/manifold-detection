from __future__ import annotations

import torch
import torch.nn as nn
from spectral_detection_posttrain.experiments.schema import resolve_model_name, validate_experiment_config
from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
    FasterRCNN_ResNet50_FPN_Weights,
)
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn, fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

MODEL_REGISTRY = {
    "fasterrcnn_mobilenet_v3_large_320_fpn": (
        fasterrcnn_mobilenet_v3_large_320_fpn,
        FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
    ),
    "fasterrcnn_resnet50_fpn": (
        fasterrcnn_resnet50_fpn,
        FasterRCNN_ResNet50_FPN_Weights,
    ),
}


def build_detector(config: dict) -> torch.nn.Module:
    config = validate_experiment_config(config, formal=False)
    model_cfg = config["model"]
    num_classes = int(model_cfg.get("num_classes", 2))
    pretrained = bool(model_cfg.get("pretrained", True))
    model_kwargs = {
        "min_size": int(model_cfg.get("min_size", 320)),
        "max_size": int(model_cfg.get("max_size", 320)),
    }
    model_name = resolve_model_name(model_cfg)
    build_fn, weights_cls = MODEL_REGISTRY[model_name]
    weights = weights_cls.DEFAULT if pretrained else None
    try:
        if pretrained:
            model = build_fn(weights=weights, **model_kwargs)
        else:
            model = build_fn(weights=None, weights_backbone=None, **model_kwargs)
    except Exception:
        if not bool(model_cfg.get("allow_random_init_fallback", True)):
            raise
        model = build_fn(weights=None, weights_backbone=None, **model_kwargs)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    afm_channels = int(model_cfg.get("afm_channels", 0))
    afm_fpn = bool(model_cfg.get("afm_fpn", False))

    if afm_fpn:
        from spectral_detection_posttrain.methods.afm.micro_afm import MultiScaleAFM

        fpn_channels = [256, 256, 256, 256]
        afm_fpn_gate = float(model_cfg.get("afm_fpn_gate", 0.6))
        multi_afm = MultiScaleAFM(channels=fpn_channels, gate_strength=afm_fpn_gate)
        original_backbone_forward = model.backbone.forward

        def _patched_backbone_forward(x):
            features = original_backbone_forward(x)
            if isinstance(features, torch.Tensor):
                features = {"0": features}
            out = {}
            for i, (key, feat) in enumerate(features.items()):
                out[key] = multi_afm(feat, level=i)
            return out

        model.backbone.forward = _patched_backbone_forward
        model._multi_afm = multi_afm

    elif afm_channels > 0:
        from spectral_detection_posttrain.methods.afm.micro_afm import build_afm_block

        afm_type = str(model_cfg.get("afm_type", "identity"))
        afm_residual_mode = str(model_cfg.get("afm_residual_mode", "current"))
        afm = build_afm_block(afm_type=afm_type, channels=afm_channels, residual_mode=afm_residual_mode)
        if afm is not None:
            original_box_head = model.roi_heads.box_head

            class AFMThenHead(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.afm = afm
                    self.head = original_box_head

                def forward(self, x):
                    x = self.afm(x)
                    return self.head(x)

            model.roi_heads.box_head = AFMThenHead()
            model._afm_type = afm_type
            model._afm_residual_mode = afm_residual_mode

    return model


def freeze_backbone(model: torch.nn.Module) -> None:
    for param in model.backbone.parameters():
        param.requires_grad = False


def freeze_rpn(model: torch.nn.Module) -> None:
    for param in model.rpn.parameters():
        param.requires_grad = False


def freeze_box_head(model: torch.nn.Module) -> None:
    for param in model.roi_heads.box_head.parameters():
        param.requires_grad = False


def freeze_box_predictor(model: torch.nn.Module) -> None:
    for param in model.roi_heads.box_predictor.parameters():
        param.requires_grad = False


def freeze_detector_for_rlvr(model: torch.nn.Module, unfreeze_mode: str = "cls") -> None:
    """Freeze all detector params except those specified by unfreeze_mode.

    unfreeze_mode:
        cls  - only roi_heads.box_predictor.cls_score is trainable
        box  - entire roi_heads.box_predictor is trainable
    """
    freeze_backbone(model)
    freeze_rpn(model)
    freeze_box_head(model)
    for param in model.roi_heads.box_predictor.parameters():
        param.requires_grad = False

    model.roi_heads.box_predictor.cls_score.weight.requires_grad = True
    model.roi_heads.box_predictor.cls_score.bias.requires_grad = True

    if unfreeze_mode == "box":
        model.roi_heads.box_predictor.bbox_pred.weight.requires_grad = True
        model.roi_heads.box_predictor.bbox_pred.bias.requires_grad = True


def set_rlvr_trainable_params(model: torch.nn.Module, mode: str = "box") -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    if mode in {"box", "roi"}:
        for parameter in model.roi_heads.box_head.parameters():
            parameter.requires_grad = True
        for parameter in model.roi_heads.box_predictor.parameters():
            parameter.requires_grad = True
    elif mode == "cls":
        for parameter in model.roi_heads.box_predictor.cls_score.parameters():
            parameter.requires_grad = True
    else:
        raise ValueError(f"Unknown RLVR trainable mode: {mode}")


def set_detector_eval_except_trainable(model: torch.nn.Module) -> None:
    """Keep frozen detector state stable while allowing trainable leaf modules to get gradients.

    All BatchNorm sub-modules stay in eval mode regardless of requires_grad to prevent
    running-stat drift. Only leaf modules with at least one trainable parameter enter
    train mode.
    """
    model.eval()
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()
    for module in model.modules():
        has_trainable_param = any(
            parameter.requires_grad for parameter in module.parameters(recurse=False)
        )
        if has_trainable_param and not isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.train()
