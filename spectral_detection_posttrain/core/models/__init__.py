"""Canonical model components shared across experiment families."""

from .box_heads import (
    AttentionPoolBoxHead,
    BottleneckBoxHead,
    BottleneckTwoMLPHead,
    ConvLowDimBoxHead,
    get_box_head_type,
    replace_box_head,
)
from .build_detector import (
    build_detector,
    freeze_backbone,
    freeze_box_head,
    freeze_box_predictor,
    freeze_detector_for_rlvr,
    freeze_rpn,
    set_detector_eval_except_trainable,
    set_rlvr_trainable_params,
)
from .bbox_adapter import (
    ResidualBBoxPredictorAdapter,
    freeze_adapters_and_predictor,
    freeze_bbox_adapter_only,
    freeze_selected_adapters,
    install_residual_bbox_adapter,
)
from .spectral_quality_head import SpectralQualityHead

__all__ = [
    "AttentionPoolBoxHead",
    "BottleneckBoxHead",
    "BottleneckTwoMLPHead",
    "ConvLowDimBoxHead",
    "ResidualBBoxPredictorAdapter",
    "SpectralQualityHead",
    "build_detector",
    "freeze_adapters_and_predictor",
    "freeze_backbone",
    "freeze_bbox_adapter_only",
    "freeze_box_head",
    "freeze_box_predictor",
    "freeze_detector_for_rlvr",
    "freeze_rpn",
    "freeze_selected_adapters",
    "get_box_head_type",
    "install_residual_bbox_adapter",
    "replace_box_head",
    "set_detector_eval_except_trainable",
    "set_rlvr_trainable_params",
]
