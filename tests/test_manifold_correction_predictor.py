from __future__ import annotations

import torch
import torch.nn as nn

from spectral_detection_posttrain.methods.manifold import (
    ManifoldCorrectionPredictor,
    PrototypeBank,
    TransportHead,
)


class RecordingPredictor(nn.Module):
    def __init__(self, feature_dim: int, num_classes: int) -> None:
        super().__init__()
        self.cls_score = nn.Linear(feature_dim, num_classes)
        self.bbox_pred = nn.Linear(feature_dim, num_classes * 4)
        self.last_input: torch.Tensor | None = None

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.last_input = features
        return self.cls_score(features), self.bbox_pred(features)


def test_manifold_correction_predictor_alters_features_before_prediction() -> None:
    torch.manual_seed(0)
    feature_dim = 4
    num_classes = 3
    num_prototypes = 2
    base = RecordingPredictor(feature_dim, num_classes)
    bank = PrototypeBank(num_classes, num_prototypes, feature_dim)
    head = TransportHead(feature_dim, num_classes * num_prototypes, tau=0.2)

    with torch.no_grad():
        bank.prototypes.zero_()
        head.mlp[-1].weight.zero_()
        head.mlp[-1].bias.zero_()
        first_foreground_offset = num_prototypes * feature_dim
        head.mlp[-1].bias[
            first_foreground_offset : first_foreground_offset + feature_dim
        ] = torch.tensor([0.2, -0.1, 0.0, 0.1])

    predictor = ManifoldCorrectionPredictor(
        base,
        prototype_bank=bank,
        transport_head=head,
        gamma=1.0,
        tau=0.2,
        normalize_features=False,
    )

    features = torch.randn(5, feature_dim, requires_grad=True)
    class_logits, box_regression = predictor(features)

    assert class_logits.shape == (5, num_classes)
    assert box_regression.shape == (5, num_classes * 4)
    assert base.last_input is not None
    assert not torch.allclose(base.last_input, features)

    loss = class_logits.sum() + box_regression.sum()
    loss.backward()

    assert features.grad is not None
    assert head.mlp[-1].bias.grad is not None
    assert head.mlp[-1].bias.grad.abs().sum() > 0


def test_manifold_correction_predictor_gamma_zero_matches_base_predictor() -> None:
    torch.manual_seed(0)
    feature_dim = 4
    num_classes = 3
    num_prototypes = 2
    base = RecordingPredictor(feature_dim, num_classes)
    bank = PrototypeBank(num_classes, num_prototypes, feature_dim)
    head = TransportHead(feature_dim, num_classes * num_prototypes)
    predictor = ManifoldCorrectionPredictor(
        base,
        prototype_bank=bank,
        transport_head=head,
        gamma=0.0,
    )

    features = torch.randn(5, feature_dim)
    expected_logits, expected_boxes = base(features)
    actual_logits, actual_boxes = predictor(features)

    assert torch.allclose(actual_logits, expected_logits)
    assert torch.allclose(actual_boxes, expected_boxes)


def test_correction_field_uses_predicted_foreground_class_gate() -> None:
    feature_dim = 2
    num_classes = 3
    num_prototypes = 1
    base = RecordingPredictor(feature_dim, num_classes)
    bank = PrototypeBank(num_classes, num_prototypes, feature_dim)
    head = TransportHead(feature_dim, num_classes * num_prototypes)

    with torch.no_grad():
        bank.prototypes.zero_()
        head.mlp[-1].weight.zero_()
        head.mlp[-1].bias.zero_()
        # Residual slots are flattened as class0/bg, class1, class2.
        head.mlp[-1].bias[0:2] = torch.tensor([10.0, 0.0])
        head.mlp[-1].bias[2:4] = torch.tensor([1.0, 0.0])
        head.mlp[-1].bias[4:6] = torch.tensor([0.0, 2.0])

    predictor = ManifoldCorrectionPredictor(
        base,
        prototype_bank=bank,
        transport_head=head,
        gamma=1.0,
        normalize_features=False,
    )

    features = torch.zeros(1, feature_dim)
    logits = torch.tensor([[-10.0, -10.0, 10.0]])
    field = predictor.correction_field(features, logits)

    assert torch.allclose(field, torch.tensor([[0.0, 2.0]]), atol=1e-4)


def test_correction_field_from_class_weights_reuses_flattened_transport_slots() -> None:
    feature_dim = 2
    num_classes = 3
    num_prototypes = 2
    base = RecordingPredictor(feature_dim, num_classes)
    bank = PrototypeBank(num_classes, num_prototypes, feature_dim)
    head = TransportHead(feature_dim, num_classes * num_prototypes, tau=0.1)

    with torch.no_grad():
        bank.prototypes.zero_()
        head.mlp[-1].weight.zero_()
        head.mlp[-1].bias.zero_()
        # Flattened slots are class-major: c0k0, c0k1, c1k0, c1k1, c2k0, c2k1.
        head.mlp[-1].bias[4:6] = torch.tensor([1.0, 0.0])
        head.mlp[-1].bias[6:8] = torch.tensor([3.0, 0.0])
        head.mlp[-1].bias[8:10] = torch.tensor([0.0, 5.0])
        head.mlp[-1].bias[10:12] = torch.tensor([0.0, 7.0])

    predictor = ManifoldCorrectionPredictor(
        base,
        prototype_bank=bank,
        transport_head=head,
        gamma=1.0,
        normalize_features=False,
    )
    field_from_weights = getattr(predictor, "correction_field_from_class_weights", None)
    assert field_from_weights is not None

    features = torch.zeros(1, feature_dim)
    class_weights = torch.tensor([[0.0, 1.0, 0.0]])
    field = field_from_weights(features, class_weights)

    assert torch.allclose(field, torch.tensor([[2.0, 0.0]]), atol=1e-4)


def test_endpoint_correction_field_points_to_class_prototype_endpoint() -> None:
    feature_dim = 2
    num_classes = 3
    num_prototypes = 1
    base = RecordingPredictor(feature_dim, num_classes)
    bank = PrototypeBank(num_classes, num_prototypes, feature_dim)
    head = TransportHead(feature_dim, num_classes * num_prototypes)

    with torch.no_grad():
        bank.prototypes.zero_()
        bank.prototypes[1, 0] = torch.tensor([2.0, 0.0])
        bank.prototypes[2, 0] = torch.tensor([0.0, 3.0])

    predictor = ManifoldCorrectionPredictor(
        base,
        prototype_bank=bank,
        transport_head=head,
        gamma=1.0,
        normalize_features=False,
        correction_mode="endpoint",
    )

    features = torch.tensor([[0.5, -1.0]])
    class_weights = torch.tensor([[0.0, 1.0, 0.0]])
    field = predictor.correction_field_from_class_weights(features, class_weights)

    assert torch.allclose(field, torch.tensor([[1.5, 1.0]]), atol=1e-6)


def test_gated_endpoint_correction_has_trainable_gate() -> None:
    feature_dim = 2
    num_classes = 3
    num_prototypes = 1
    base = RecordingPredictor(feature_dim, num_classes)
    bank = PrototypeBank(num_classes, num_prototypes, feature_dim)
    head = TransportHead(feature_dim, num_classes * num_prototypes)

    with torch.no_grad():
        bank.prototypes.zero_()
        bank.prototypes[1, 0] = torch.tensor([2.0, 0.0])

    predictor = ManifoldCorrectionPredictor(
        base,
        prototype_bank=bank,
        transport_head=head,
        gamma=1.0,
        normalize_features=False,
        correction_mode="gated_endpoint",
        endpoint_gate_init=0.25,
    )

    features = torch.tensor([[0.5, 0.0]], requires_grad=True)
    class_weights = torch.tensor([[0.0, 1.0, 0.0]])
    field = predictor.correction_field_from_class_weights(features, class_weights)

    assert torch.allclose(field, torch.tensor([[0.375, 0.0]]), atol=1e-6)

    field.sum().backward()

    gate_params = list(predictor.endpoint_gate_parameters())
    assert gate_params
    assert gate_params[-1].grad is not None
    assert gate_params[-1].grad.abs().sum() > 0.0
