from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn

from analyze_layerwise_geometry import install_active_correction_from_checkpoint_state
from spectral_detection_posttrain.methods.manifold import ManifoldCorrectionPredictor


class DummyPredictor(nn.Module):
    def __init__(self, feature_dim: int, num_classes: int) -> None:
        super().__init__()
        self.cls_score = nn.Linear(feature_dim, num_classes)
        self.bbox_pred = nn.Linear(feature_dim, num_classes * 4)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.cls_score(features), self.bbox_pred(features)


class DummyRoiHeads(nn.Module):
    def __init__(self, feature_dim: int, num_classes: int) -> None:
        super().__init__()
        self.box_predictor = DummyPredictor(feature_dim, num_classes)


class DummyDetector(nn.Module):
    def __init__(self, feature_dim: int, num_classes: int) -> None:
        super().__init__()
        self.roi_heads = DummyRoiHeads(feature_dim, num_classes)


def _active_checkpoint_state(feature_dim: int, num_classes: int, num_prototypes: int) -> dict[str, torch.Tensor]:
    hidden_dim = feature_dim
    flattened = num_classes * num_prototypes
    return {
        "roi_heads.box_predictor.base_predictor.cls_score.weight": torch.randn(num_classes, feature_dim),
        "roi_heads.box_predictor.base_predictor.cls_score.bias": torch.randn(num_classes),
        "roi_heads.box_predictor.base_predictor.bbox_pred.weight": torch.randn(num_classes * 4, feature_dim),
        "roi_heads.box_predictor.base_predictor.bbox_pred.bias": torch.randn(num_classes * 4),
        "roi_heads.box_predictor.prototype_bank.prototypes": torch.randn(num_classes, num_prototypes, feature_dim),
        "roi_heads.box_predictor.prototype_bank.ema_sums": torch.randn(num_classes, num_prototypes, feature_dim),
        "roi_heads.box_predictor.prototype_bank.ema_counts": torch.ones(num_classes, num_prototypes),
        "roi_heads.box_predictor.transport_head.mlp.0.weight": torch.randn(hidden_dim, feature_dim),
        "roi_heads.box_predictor.transport_head.mlp.0.bias": torch.randn(hidden_dim),
        "roi_heads.box_predictor.transport_head.mlp.2.weight": torch.randn(flattened * feature_dim, hidden_dim),
        "roi_heads.box_predictor.transport_head.mlp.2.bias": torch.randn(flattened * feature_dim),
        "roi_heads.box_predictor.endpoint_gate.weight": torch.randn(1, feature_dim),
        "roi_heads.box_predictor.endpoint_gate.bias": torch.randn(1),
    }


def test_install_active_correction_from_checkpoint_state_wraps_detector_for_gated_endpoint(
    tmp_path: Path,
) -> None:
    feature_dim = 8
    num_classes = 3
    num_prototypes = 2
    checkpoint_path = tmp_path / "checkpoint_best_ap75.pth"
    (tmp_path / "manifold_config.json").write_text(
        json.dumps(
            {
                "active_correction_gamma": 0.35,
                "active_correction_mode": "gated_endpoint",
                "tau": 0.07,
            }
        ),
        encoding="utf-8",
    )

    model = DummyDetector(feature_dim=feature_dim, num_classes=num_classes)
    state_dict = _active_checkpoint_state(feature_dim, num_classes, num_prototypes)

    installed = install_active_correction_from_checkpoint_state(
        model,
        state_dict,
        checkpoint_path=checkpoint_path,
        device=torch.device("cpu"),
    )

    assert installed is True
    predictor = model.roi_heads.box_predictor
    assert isinstance(predictor, ManifoldCorrectionPredictor)
    assert predictor.gamma == 0.35
    assert predictor.correction_mode == "gated_endpoint"
    assert predictor.transport_head.tau == 0.07
    model.load_state_dict(state_dict, strict=True)


def test_install_active_correction_uses_checkpoint_epoch_gamma_metadata(tmp_path: Path) -> None:
    feature_dim = 8
    num_classes = 3
    num_prototypes = 2
    checkpoint_path = tmp_path / "checkpoint_best_ap75.pth"
    (tmp_path / "manifold_config.json").write_text(
        json.dumps(
            {
                "active_correction_gamma": 0.35,
                "active_correction_mode": "gated_endpoint",
                "tau": 0.07,
            }
        ),
        encoding="utf-8",
    )

    model = DummyDetector(feature_dim=feature_dim, num_classes=num_classes)
    state_dict = _active_checkpoint_state(feature_dim, num_classes, num_prototypes)

    installed = install_active_correction_from_checkpoint_state(
        model,
        state_dict,
        checkpoint_path=checkpoint_path,
        device=torch.device("cpu"),
        checkpoint_metadata={"active_correction_gamma_epoch": 0.2},
    )

    assert installed is True
    assert isinstance(model.roi_heads.box_predictor, ManifoldCorrectionPredictor)
    assert model.roi_heads.box_predictor.gamma == 0.2


def test_install_active_correction_from_checkpoint_state_ignores_baseline_predictor() -> None:
    model = DummyDetector(feature_dim=8, num_classes=3)

    installed = install_active_correction_from_checkpoint_state(
        model,
        {"roi_heads.box_predictor.cls_score.weight": torch.randn(3, 8)},
        checkpoint_path=Path("checkpoint.pth"),
        device=torch.device("cpu"),
    )

    assert installed is False
    assert not isinstance(model.roi_heads.box_predictor, ManifoldCorrectionPredictor)
