from __future__ import annotations

import math
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


def _get_roi_feature_channels(model: torch.nn.Module) -> int:
    """Return the channel depth of the tensor that enters box_head.

    TorchVision Faster R-CNN keeps this in ``model.backbone.out_channels`` for
    FPN backbones.  We fall back to a config override or the heuristic 256.
    """
    out_ch = getattr(model.backbone, "out_channels", None)
    if isinstance(out_ch, int) and out_ch > 0:
        return out_ch
    return 256


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

    # ------------------------------------------------------------------
    # Resolve feature dimensions safely.
    # ------------------------------------------------------------------
    roi_channels = int(model_cfg.get("roi_channels", 0))
    if roi_channels <= 0:
        roi_channels = _get_roi_feature_channels(model)

    box_in_features = model.roi_heads.box_predictor.cls_score.in_features

    # ------------------------------------------------------------------
    # Optional PAH: keep a reference to the old predictor so we can
    # initialise the residual prototype head from pretrained weights.
    # ------------------------------------------------------------------
    use_pah = bool(model_cfg.get("use_pah", False))
    old_predictor = model.roi_heads.box_predictor
    if use_pah:
        from spectral_detection_posttrain.methods.detection.pah import ResidualPrototypeHead

        pah_temperature = float(model_cfg.get("pah_temperature", 0.1))
        pah_learnable_temp = bool(model_cfg.get("pah_learnable_temp", False))
        pah_num_bg = int(model_cfg.get("pah_num_bg", 4))
        pah_gamma_init = float(model_cfg.get("pah_gamma_init", 0.0))
        model.roi_heads.box_predictor = ResidualPrototypeHead(
            in_features=box_in_features,
            num_classes=num_classes,
            temperature=pah_temperature,
            learnable_temp=pah_learnable_temp,
            num_background_prototypes=pah_num_bg,
            gamma_init=pah_gamma_init,
            old_predictor=old_predictor,
        )
    else:
        model.roi_heads.box_predictor = FastRCNNPredictor(box_in_features, num_classes)

    # ------------------------------------------------------------------
    # Optional multi-scale AFM on FPN features (legacy / special use).
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Optional structural blocks around the box head.
    # ------------------------------------------------------------------
    use_pbg = bool(model_cfg.get("use_pbg", False))
    use_lsg = bool(model_cfg.get("use_lsg", False))
    use_tam = bool(model_cfg.get("use_tam", False))

    pbg = None
    if use_pbg:
        from spectral_detection_posttrain.methods.detection.pbg import PhaseBoundaryGate

        pbg_alpha_init = float(model_cfg.get("pbg_alpha_init", 1e-2))
        pbg_phase_mask = str(model_cfg.get("pbg_phase_mask", "soft"))
        pbg = PhaseBoundaryGate(
            channels=roi_channels,
            alpha_init=pbg_alpha_init,
            phase_mask=pbg_phase_mask,
        )

    lsg = None
    if use_lsg:
        from spectral_detection_posttrain.methods.detection.pbg import LearnedSpectralGate

        lsg_alpha_init = float(model_cfg.get("lsg_alpha_init", 0.1))
        lsg_use_radius = bool(model_cfg.get("lsg_use_radius", True))
        lsg = LearnedSpectralGate(
            channels=roi_channels,
            alpha_init=lsg_alpha_init,
            use_radius=lsg_use_radius,
        )

    spatial_afm = None
    afm_channels = int(model_cfg.get("afm_channels", 0))
    if afm_channels > 0:
        afm_type = str(model_cfg.get("afm_type", "identity"))
        from spectral_detection_posttrain.methods.afm.micro_afm import build_afm_block
        afm_residual_mode = str(model_cfg.get("afm_residual_mode", "current"))
        spatial_afm = build_afm_block(
            afm_type=afm_type, channels=afm_channels, residual_mode=afm_residual_mode
        )
        model._afm_type = afm_type
        model._afm_residual_mode = afm_residual_mode

    tam = None
    if use_tam:
        from spectral_detection_posttrain.methods.detection.tam import TaskAlignedManifold

        tam_latent_dim = int(model_cfg.get("tam_latent_dim", 256))
        tam_spectral_quality = bool(model_cfg.get("tam_spectral_quality", False))
        tam_contrastive = bool(model_cfg.get("tam_contrastive", False))
        tam = TaskAlignedManifold(
            in_features=box_in_features,
            latent_dim=tam_latent_dim,
            use_spectral_quality=tam_spectral_quality,
            use_contrastive=tam_contrastive,
        )

    # Optionally increase ROI Align resolution for better frequency-domain
    # processing.  box_head is trained for 7x7 inputs, so we downsample back
    # before feeding it if a larger ROI size is requested.
    roi_align_size = int(model_cfg.get("roi_align_size", 7))
    if roi_align_size != 7:
        if hasattr(model.roi_heads.box_roi_pool, "output_size"):
            model.roi_heads.box_roi_pool.output_size = (roi_align_size, roi_align_size)

    # Insert blocks around the box head: PBG -> LSG -> AFM -> downsample -> head -> TAM.
    if pbg is not None or lsg is not None or spatial_afm is not None or tam is not None or roi_align_size != 7:
        original_box_head = model.roi_heads.box_head
        downsample = nn.AdaptiveAvgPool2d((7, 7)) if roi_align_size != 7 else nn.Identity()

        class RefinedBoxHead(nn.Module):
            def __init__(self):
                super().__init__()
                self.pbg = pbg
                self.lsg = lsg
                self.spatial_afm = spatial_afm
                # Backward-compatible alias used by some trainable-mode helpers.
                self.afm = spatial_afm
                self.downsample = downsample
                self.head = original_box_head
                self.tam = tam

            def forward(self, x, proposals=None):
                if self.pbg is not None:
                    x = self.pbg(x)
                if self.lsg is not None:
                    x = self.lsg(x)
                if self.spatial_afm is not None:
                    x = self.spatial_afm(x)
                x = self.downsample(x)
                z = self.head(x)
                if self.tam is not None:
                    labels = getattr(self, "_last_labels", None)
                    z = self.tam(z, roi_feature=x, labels=labels)
                    self._last_labels = None
                return z

        model.roi_heads.box_head = RefinedBoxHead()

        # If TAM contrastive learning is enabled, patch the training-sample
        # selector so that the matched labels are available inside the wrapper.
        if tam is not None and tam.use_contrastive:
            original_select = model.roi_heads.select_training_samples

            def _patched_select(proposals, targets):
                proposals_out, matched_idxs, labels, regression_targets = original_select(
                    proposals, targets
                )
                if labels is not None and len(labels) > 0:
                    model.roi_heads.box_head._last_labels = torch.cat(labels, dim=0)
                return proposals_out, matched_idxs, labels, regression_targets

            model.roi_heads.select_training_samples = _patched_select

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
