import torch
from torch import nn

from spectral_detection_posttrain.models.bbox_adapter import (
    ResidualBBoxPredictorAdapter,
    freeze_bbox_adapter_only,
    freeze_adapters_and_predictor,
    freeze_selected_adapters,
    install_residual_bbox_adapter,
)


class _FakeBoxPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.cls_score = nn.Linear(4, 2)
        self.bbox_pred = nn.Linear(4, 8)

    def forward(self, x):
        return self.cls_score(x), self.bbox_pred(x)


class _FakeRoiHeads(nn.Module):
    def __init__(self):
        super().__init__()
        self.box_head = nn.Linear(4, 4)
        self.box_predictor = _FakeBoxPredictor()


class _FakeDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(4, 4)
        self.rpn = nn.Linear(4, 4)
        self.roi_heads = _FakeRoiHeads()


def test_residual_bbox_adapter_is_exact_noop_at_initialization():
    predictor = _FakeBoxPredictor()
    adapter = ResidualBBoxPredictorAdapter(predictor, hidden_dim=4, scale=1.0)
    x = torch.randn(5, 4)

    base_cls, base_box = predictor(x)
    cls, box = adapter(x)

    assert torch.allclose(cls, base_cls)
    assert torch.allclose(box, base_box)


def test_install_residual_bbox_adapter_preserves_detector_outputs():
    model = _FakeDetector()
    x = torch.randn(3, 4)

    base_cls, base_box = model.roi_heads.box_predictor(x)
    adapter = install_residual_bbox_adapter(model, hidden_dim=4, scale=1.0)
    cls, box = model.roi_heads.box_predictor(x)

    assert isinstance(adapter, ResidualBBoxPredictorAdapter)
    assert torch.allclose(cls, base_cls)
    assert torch.allclose(box, base_box)


def test_residual_cls_adapter_is_exact_noop_at_initialization():
    predictor = _FakeBoxPredictor()
    adapter = ResidualBBoxPredictorAdapter(
        predictor,
        hidden_dim=4,
        scale=1.0,
        enable_cls_adapter=True,
        cls_scale=0.5,
    )
    x = torch.randn(5, 4)

    base_cls, base_box = predictor(x)
    cls, box = adapter(x)

    assert adapter.cls_adapter is not None
    assert torch.allclose(cls, base_cls)
    assert torch.allclose(box, base_box)


def test_freeze_bbox_adapter_only_trains_adapter_not_base_predictor():
    model = _FakeDetector()
    install_residual_bbox_adapter(model, hidden_dim=4, scale=1.0)

    trainable = freeze_bbox_adapter_only(model)

    assert trainable
    assert all("bbox_adapter" in name for name in trainable)
    for name, parameter in model.named_parameters():
        if "bbox_adapter" in name:
            assert parameter.requires_grad, name
        else:
            assert not parameter.requires_grad, name


def test_freeze_selected_adapters_can_train_only_cls_adapter():
    model = _FakeDetector()
    install_residual_bbox_adapter(
        model,
        hidden_dim=4,
        scale=1.0,
        enable_cls_adapter=True,
        cls_scale=0.5,
    )

    trainable = freeze_selected_adapters(model, train_bbox_adapter=False, train_cls_adapter=True)

    assert trainable
    assert all("cls_adapter" in name for name in trainable)
    for name, parameter in model.named_parameters():
        if "cls_adapter" in name:
            assert parameter.requires_grad, name
        else:
            assert not parameter.requires_grad, name


def test_freeze_adapters_and_predictor_trains_adapters_and_base_predictor_only():
    model = _FakeDetector()
    install_residual_bbox_adapter(
        model,
        hidden_dim=4,
        scale=1.0,
        enable_cls_adapter=True,
        cls_scale=0.5,
    )

    trainable = freeze_adapters_and_predictor(model, train_cls_adapter=True)

    assert any("bbox_adapter" in name for name in trainable)
    assert any("cls_adapter" in name for name in trainable)
    assert any("base_predictor.cls_score" in name for name in trainable)
    assert any("base_predictor.bbox_pred" in name for name in trainable)
    for name, parameter in model.named_parameters():
        if (
            "bbox_adapter" in name
            or "cls_adapter" in name
            or "box_predictor.base_predictor.cls_score" in name
            or "box_predictor.base_predictor.bbox_pred" in name
        ):
            assert parameter.requires_grad, name
        else:
            assert not parameter.requires_grad, name


def test_freeze_adapters_and_predictor_can_train_cls_branch_only():
    model = _FakeDetector()
    install_residual_bbox_adapter(
        model,
        hidden_dim=4,
        scale=1.0,
        enable_cls_adapter=True,
        cls_scale=0.5,
    )

    trainable = freeze_adapters_and_predictor(
        model,
        train_bbox_adapter=False,
        train_cls_adapter=True,
        train_cls_score=True,
        train_bbox_pred=False,
    )

    assert trainable
    assert any("cls_adapter" in name for name in trainable)
    assert any("base_predictor.cls_score" in name for name in trainable)
    assert all("bbox_adapter" not in name for name in trainable)
    assert all("base_predictor.bbox_pred" not in name for name in trainable)
    for name, parameter in model.named_parameters():
        if "cls_adapter" in name or "box_predictor.base_predictor.cls_score" in name:
            assert parameter.requires_grad, name
        else:
            assert not parameter.requires_grad, name


def test_install_residual_bbox_adapter_preserves_existing_device():
    model = _FakeDetector().to(torch.device("meta"))

    adapter = install_residual_bbox_adapter(model, hidden_dim=4, scale=1.0)

    assert next(adapter.bbox_adapter.parameters()).device.type == "meta"
