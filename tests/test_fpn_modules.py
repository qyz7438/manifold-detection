"""Smoke tests for FPN spectral manifold, real adapter, and attention baselines."""

from __future__ import annotations

import sys

import pytest
import torch

from spectral_detection_posttrain.core.models.build_detector import build_detector


def _minimal_config(**overrides):
    cfg = {
        "model": {
            "name": "fasterrcnn_mobilenet_v3_large_320_fpn",
            "num_classes": 21,
            "pretrained": False,
            "trainable_backbone_layers": 3,
        },
        "data": {"dataset": "voc"},
    }
    cfg["model"].update(overrides)
    return cfg


def _dummy_input(device="cpu"):
    return torch.zeros(1, 3, 320, 320, device=device)


@pytest.mark.parametrize("model_name", ["fasterrcnn_mobilenet_v3_large_320_fpn"])
def test_baseline_forward(model_name):
    config = _minimal_config(name=model_name, num_classes=21)
    model = build_detector(config)
    model.eval()
    with torch.no_grad():
        out = model.backbone(_dummy_input())
    assert "0" in out


def test_fpn_spectral_manifold_forward():
    config = _minimal_config(
        fpn_spectral_manifold=True,
        fpn_sm_channels=256,
        fpn_sm_levels=4,
        fpn_sm_latent_dim=16,
        fpn_sm_hidden_dim=32,
        fpn_sm_use_freq_coords=False,
        fpn_sm_use_level_coords=False,
        fpn_sm_init_alpha=0.01,
    )
    model = build_detector(config)
    model.eval()
    assert hasattr(model, "_fpn_spectral_manifold")
    with torch.no_grad():
        out = model.backbone(_dummy_input())
    assert "0" in out


def test_fpn_real_adapter_forward():
    config = _minimal_config(
        fpn_real_adapter=True,
        fpn_real_channels=256,
        fpn_real_levels=4,
        fpn_real_init_alpha=0.01,
    )
    model = build_detector(config)
    model.eval()
    assert hasattr(model, "_fpn_real_adapter")
    with torch.no_grad():
        out = model.backbone(_dummy_input())
    assert "0" in out


@pytest.mark.parametrize("attention_type", ["se", "fcanet", "eca"])
@pytest.mark.parametrize("reduction", [4, 16])
def test_fpn_attention_baseline_forward(attention_type, reduction):
    config = _minimal_config(
        fpn_attention_type=attention_type,
        fpn_attention_reduction=reduction,
    )
    model = build_detector(config)
    model.eval()
    assert hasattr(model, "_fpn_attention")
    assert model._fpn_attention.reduction == reduction
    with torch.no_grad():
        out = model.backbone(_dummy_input())
    assert "0" in out


def test_coco_loader_is_lazy_import():
    """Importing the datasets package must not require pycocotools."""
    # Ensure pycocotools is not preloaded; if it is, skip rather than fail.
    if "pycocotools" in sys.modules:
        pytest.skip("pycocotools already imported in this process")

    import spectral_detection_posttrain.datasets as datasets_module

    assert "pycocotools" not in sys.modules
    assert hasattr(datasets_module, "build_detection_loaders")


def test_coco_loader_raises_without_pycocotools(monkeypatch):
    """If pycocotools is absent, dataset=coco should raise ImportError lazily."""
    # Simulate pycocotools not being installed by hiding it from importlib.
    monkeypatch.setitem(sys.modules, "pycocotools", None)
    monkeypatch.setitem(sys.modules, "pycocotools.coco", None)

    from spectral_detection_posttrain.datasets import build_detection_loaders

    config = {"data": {"dataset": "coco", "root": "./data/coco"}}
    with pytest.raises(ImportError):
        build_detection_loaders(config)
