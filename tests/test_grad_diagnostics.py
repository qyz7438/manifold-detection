import pytest
import torch
from torch import nn

from spectral_detection_posttrain.utils.grad_diagnostics import (
    parameter_component,
    summarize_current_parameter_gradients,
    summarize_loss_component_gradients,
)


def test_parameter_component_classifies_trainable_detector_parts():
    assert parameter_component("roi_heads.box_predictor.bbox_adapter.0.weight") == "bbox_adapter"
    assert parameter_component("roi_heads.box_predictor.cls_adapter.0.weight") == "cls_adapter"
    assert parameter_component("roi_heads.box_predictor.base_predictor.cls_score.weight") == "cls_score"
    assert parameter_component("roi_heads.box_predictor.base_predictor.bbox_pred.weight") == "bbox_pred"
    assert parameter_component("roi_heads.box_head.fc6.weight") == "box_head"
    assert parameter_component("backbone.body.0.weight") == "backbone"


def test_summarize_current_parameter_gradients_reports_l2_per_component():
    bbox = nn.Parameter(torch.tensor([1.0, 2.0]))
    cls = nn.Parameter(torch.tensor([3.0]))
    bbox.grad = torch.tensor([3.0, 4.0])
    cls.grad = torch.tensor([12.0])

    metrics = summarize_current_parameter_gradients(
        [
            ("roi_heads.box_predictor.bbox_adapter.0.weight", bbox),
            ("roi_heads.box_predictor.cls_adapter.0.weight", cls),
        ]
    )

    assert metrics["grad_total_bbox_adapter_l2"] == pytest.approx(5.0)
    assert metrics["grad_total_cls_adapter_l2"] == pytest.approx(12.0)
    assert metrics["grad_total_total_l2"] == pytest.approx(13.0)
    assert metrics["grad_total_bbox_adapter_max_abs"] == pytest.approx(4.0)
    assert metrics["grad_total_cls_adapter_elem_count"] == 1


def test_summarize_loss_component_gradients_keeps_components_separate():
    bbox = nn.Parameter(torch.tensor([2.0]))
    cls = nn.Parameter(torch.tensor([3.0]))
    losses = {
        "bbox": bbox.pow(2).sum(),
        "cls": (2.0 * cls).sum(),
    }

    metrics = summarize_loss_component_gradients(
        losses,
        [
            ("roi_heads.box_predictor.bbox_adapter.0.weight", bbox),
            ("roi_heads.box_predictor.cls_adapter.0.weight", cls),
        ],
    )

    assert metrics["grad_bbox_bbox_adapter_l2"] == pytest.approx(4.0)
    assert metrics["grad_bbox_cls_adapter_l2"] == pytest.approx(0.0)
    assert metrics["grad_cls_bbox_adapter_l2"] == pytest.approx(0.0)
    assert metrics["grad_cls_cls_adapter_l2"] == pytest.approx(2.0)
