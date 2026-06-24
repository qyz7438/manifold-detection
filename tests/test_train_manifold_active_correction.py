from __future__ import annotations

from collections import OrderedDict
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from spectral_detection_posttrain.methods.manifold import (
    ManifoldCorrectionPredictor,
    PrototypeBank,
    SinkhornAssigner,
)
import spectral_detection_posttrain.trainers.detection.train_manifold_posttrain as trainer
from spectral_detection_posttrain.trainers.detection.train_manifold_posttrain import (
    install_active_manifold_correction,
)


class DummyPredictor(nn.Module):
    def __init__(self, feature_dim: int, num_classes: int) -> None:
        super().__init__()
        self.cls_score = nn.Linear(feature_dim, num_classes)
        self.bbox_pred = nn.Linear(feature_dim, num_classes * 4)

    def forward(self, x):
        return self.cls_score(x), self.bbox_pred(x)


class DummyModel(nn.Module):
    def __init__(self, feature_dim: int, num_classes: int) -> None:
        super().__init__()
        self.roi_heads = nn.Module()
        self.roi_heads.box_predictor = DummyPredictor(feature_dim, num_classes)


class BatchNormProbeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.BatchNorm2d(3, momentum=1.0)
        self.roi_heads = nn.Module()
        self.roi_heads.box_roi_pool = self._box_roi_pool
        self.roi_heads.box_head = nn.Flatten(start_dim=1)

    def transform(self, images, targets):
        tensors = torch.stack(images, dim=0)
        image_sizes = [tuple(image.shape[-2:]) for image in images]
        return SimpleNamespace(tensors=tensors, image_sizes=image_sizes), targets

    def _box_roi_pool(self, features, boxes, image_sizes):
        feature = features["0"] if isinstance(features, OrderedDict) else features
        return feature.mean(dim=(-2, -1), keepdim=True)


def test_install_active_manifold_correction_wraps_box_predictor() -> None:
    model = DummyModel(feature_dim=8, num_classes=4)
    bank = PrototypeBank(num_classes=4, num_prototypes_per_class=3, feature_dim=8)

    active_head = install_active_manifold_correction(
        model,
        prototype_bank=bank,
        gamma=0.05,
        tau=0.2,
        normalize_features=True,
    )

    assert isinstance(model.roi_heads.box_predictor, ManifoldCorrectionPredictor)
    assert model.roi_heads.box_predictor.base_predictor.cls_score.out_features == 4
    assert active_head.num_prototypes == 12


def test_parse_args_accepts_corrected_feature_geometry_weights(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_manifold_posttrain.py",
            "--config",
            "config.yaml",
            "--baseline",
            "checkpoint.pth",
            "--run-name",
            "corrected_geometry",
            "--active-correction-gamma-schedule",
            "linear_decay",
            "--active-correction-gamma-final",
            "0.1",
            "--lambda-corrected-intra",
            "0.05",
            "--lambda-corrected-inter",
            "0.1",
            "--lambda-corrected-inter-preserve",
            "0.2",
            "--lambda-corrected-center-preserve",
            "0.3",
            "--lambda-corrected-memory-center-preserve",
            "0.4",
            "--lambda-corrected-memory-inter-preserve",
            "0.5",
            "--lambda-correction-field-preserve",
            "0.6",
            "--corrected-memory-momentum",
            "0.8",
            "--corrected-inter-margin",
            "0.75",
        ],
    )

    args = trainer.parse_args()

    assert args.active_correction_gamma_schedule == "linear_decay"
    assert args.active_correction_gamma_final == 0.1
    assert args.lambda_corrected_intra == 0.05
    assert args.lambda_corrected_inter == 0.1
    assert args.lambda_corrected_inter_preserve == 0.2
    assert args.lambda_corrected_center_preserve == 0.3
    assert args.lambda_corrected_memory_center_preserve == 0.4
    assert args.lambda_corrected_memory_inter_preserve == 0.5
    assert args.lambda_correction_field_preserve == 0.6
    assert args.corrected_memory_momentum == 0.8
    assert args.corrected_inter_margin == 0.75


def test_correction_field_preservation_loss_penalizes_drift_from_reference() -> None:
    preserve_fn = getattr(trainer, "correction_field_preservation_loss", None)
    assert preserve_fn is not None

    current = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    reference = current.detach().clone()

    zero_loss = preserve_fn(current, reference, lambda_preserve=10.0)
    assert zero_loss.item() == pytest.approx(0.0)

    drifted = torch.tensor([[1.2, 0.0], [0.0, 0.7]], requires_grad=True)
    loss = preserve_fn(drifted, reference, lambda_preserve=10.0)

    assert loss.item() > 0.0
    loss.backward()
    assert drifted.grad is not None
    assert drifted.grad[0, 0] > 0.0
    assert drifted.grad[1, 1] < 0.0


def test_initial_sanity_payload_separates_raw_and_active_initial_metrics() -> None:
    payload_fn = getattr(trainer, "initial_sanity_payload", None)
    assert payload_fn is not None

    raw_metrics = {"ap50": 0.11, "ap75": 0.22}
    active_metrics = {"ap50": 0.33, "ap75": 0.44}

    payload = payload_fn(raw_metrics, active_metrics)

    assert payload["raw_baseline_val_ap50"] == pytest.approx(0.11)
    assert payload["raw_baseline_val_ap75"] == pytest.approx(0.22)
    assert payload["active_initial_val_ap50"] == pytest.approx(0.33)
    assert payload["active_initial_val_ap75"] == pytest.approx(0.44)
    assert payload["initial_val_ap50"] == pytest.approx(0.33)
    assert payload["initial_val_ap75"] == pytest.approx(0.44)


def test_initial_checkpoint_metadata_marks_epoch_zero_active_metrics() -> None:
    metadata_fn = getattr(trainer, "initial_checkpoint_extra_metadata", None)
    assert metadata_fn is not None

    raw_metrics = {"ap50": 0.11, "ap75": 0.22}
    active_metrics = {"ap50": 0.33, "ap75": 0.44}

    metadata = metadata_fn(raw_metrics, active_metrics)

    assert metadata["epoch"] == 0
    assert metadata["val_ap50"] == pytest.approx(0.33)
    assert metadata["val_ap75"] == pytest.approx(0.44)
    assert metadata["raw_baseline_val_ap50"] == pytest.approx(0.11)
    assert metadata["active_initial_val_ap75"] == pytest.approx(0.44)


def test_seeded_prototype_warmup_is_independent_of_prior_rng_use() -> None:
    warmup_fn = getattr(trainer, "initialize_prototypes_from_centers_with_seed", None)
    assert warmup_fn is not None

    centers = torch.arange(24, dtype=torch.float32).reshape(3, 2, 4).mean(dim=1)
    first = PrototypeBank(num_classes=3, num_prototypes_per_class=2, feature_dim=4)
    second = PrototypeBank(num_classes=3, num_prototypes_per_class=2, feature_dim=4)

    warmup_fn(first, centers, noise_scale=0.05, seed=123)
    _ = torch.randn(97)
    warmup_fn(second, centers, noise_scale=0.05, seed=123)

    assert torch.allclose(first.prototypes, second.prototypes)


def test_infer_box_feature_dim_does_not_pollute_batchnorm_stats() -> None:
    model = BatchNormProbeModel()
    model.train()
    before_mean = model.backbone.running_mean.clone()
    before_var = model.backbone.running_var.clone()

    infer_dim = getattr(trainer, "infer_box_feature_dim", None)
    assert infer_dim is not None
    feature_dim = infer_dim(
        model,
        torch.device("cpu"),
        {"model": {"min_size": 8, "max_size": 8}},
    )

    assert feature_dim == 3
    assert model.training
    assert torch.allclose(model.backbone.running_mean, before_mean)
    assert torch.allclose(model.backbone.running_var, before_var)


def test_install_active_manifold_correction_starts_as_identity_field() -> None:
    torch.manual_seed(0)
    model = DummyModel(feature_dim=8, num_classes=4)
    bank = PrototypeBank(num_classes=4, num_prototypes_per_class=3, feature_dim=8)

    install_active_manifold_correction(
        model,
        prototype_bank=bank,
        gamma=0.15,
        tau=0.2,
        normalize_features=False,
    )

    features = torch.randn(5, 8)
    logits = torch.randn(5, 4)
    field = model.roi_heads.box_predictor.correction_field(features, logits)

    assert torch.allclose(field, torch.zeros_like(field))


def test_active_manifold_losses_reuse_installed_correction_field_and_energy() -> None:
    torch.manual_seed(0)
    feature_dim = 4
    num_classes = 4
    num_prototypes = 2
    model = DummyModel(feature_dim=feature_dim, num_classes=num_classes)
    bank = PrototypeBank(num_classes, num_prototypes, feature_dim)
    active_head = install_active_manifold_correction(
        model,
        prototype_bank=bank,
        gamma=0.2,
        tau=0.3,
        normalize_features=False,
    )
    with torch.no_grad():
        active_head.mlp[-1].bias.fill_(0.01)
    active_losses = getattr(trainer, "active_manifold_losses", None)
    assert active_losses is not None

    features = torch.randn(6, feature_dim, requires_grad=True)
    labels = torch.tensor([1, 2, 3, 1, 2, 3])
    sinkhorn = SinkhornAssigner(eps=0.1, max_iter=5)

    losses = active_losses(
        features,
        labels,
        bank,
        sinkhorn,
        model.roi_heads.box_predictor,
        lambda_tr=0.01,
        lambda_en=0.1,
        normalize=False,
    )

    assert losses["loss_manifold_total"].requires_grad
    assert losses["loss_energy"].item() > 0.0

    losses["loss_manifold_total"].backward()

    assert active_head.mlp[-1].weight.grad is not None
    assert active_head.mlp[-1].weight.grad.abs().sum() > 0.0


def test_active_correction_gamma_schedule_interpolates_over_epochs() -> None:
    gamma_for_epoch = getattr(trainer, "active_correction_gamma_for_epoch", None)
    assert gamma_for_epoch is not None

    assert gamma_for_epoch(
        initial_gamma=0.35,
        final_gamma=0.1,
        epoch=1,
        total_epochs=5,
        schedule="constant",
    ) == pytest.approx(0.35)
    assert gamma_for_epoch(
        initial_gamma=0.35,
        final_gamma=0.1,
        epoch=1,
        total_epochs=5,
        schedule="linear_decay",
    ) == pytest.approx(0.35)
    assert gamma_for_epoch(
        initial_gamma=0.35,
        final_gamma=0.1,
        epoch=3,
        total_epochs=5,
        schedule="linear_decay",
    ) == pytest.approx(0.225)
    assert gamma_for_epoch(
        initial_gamma=0.35,
        final_gamma=0.1,
        epoch=5,
        total_epochs=5,
        schedule="linear_decay",
    ) == pytest.approx(0.1)
    assert gamma_for_epoch(
        initial_gamma=0.35,
        final_gamma=0.1,
        epoch=1,
        total_epochs=1,
        schedule="linear_decay",
    ) == pytest.approx(0.35)


def test_set_active_correction_gamma_updates_wrapped_predictor() -> None:
    model = DummyModel(feature_dim=4, num_classes=3)
    bank = PrototypeBank(num_classes=3, num_prototypes_per_class=1, feature_dim=4)
    install_active_manifold_correction(
        model,
        prototype_bank=bank,
        gamma=0.35,
        tau=0.3,
        normalize_features=False,
    )

    set_gamma = getattr(trainer, "set_active_correction_gamma", None)
    assert set_gamma is not None

    updated = set_gamma(model, 0.1)

    assert updated is True
    assert model.roi_heads.box_predictor.gamma == pytest.approx(0.1)
    assert set_gamma(DummyModel(feature_dim=4, num_classes=3), 0.2) is False


def test_active_manifold_losses_train_gated_endpoint_gate() -> None:
    torch.manual_seed(0)
    feature_dim = 2
    num_classes = 3
    num_prototypes = 1
    model = DummyModel(feature_dim=feature_dim, num_classes=num_classes)
    bank = PrototypeBank(num_classes, num_prototypes, feature_dim)
    install_active_manifold_correction(
        model,
        prototype_bank=bank,
        gamma=0.5,
        tau=0.3,
        normalize_features=False,
        correction_mode="gated_endpoint",
        endpoint_gate_init=0.25,
    )
    with torch.no_grad():
        bank.prototypes.zero_()
        bank.prototypes[1, 0] = torch.tensor([1.0, 0.0])
        bank.prototypes[2, 0] = torch.tensor([0.0, 1.0])

    features = torch.tensor([[0.0, 0.0], [0.2, 0.1]], requires_grad=True)
    labels = torch.tensor([1, 2])
    sinkhorn = SinkhornAssigner(eps=0.1, max_iter=5)

    losses = trainer.active_manifold_losses(
        features,
        labels,
        bank,
        sinkhorn,
        model.roi_heads.box_predictor,
        lambda_tr=1.0,
        lambda_en=0.01,
        normalize=False,
    )

    losses["loss_manifold_total"].backward()

    gate_params = list(model.roi_heads.box_predictor.endpoint_gate_parameters())
    assert gate_params
    assert gate_params[-1].grad is not None
    assert gate_params[-1].grad.abs().sum() > 0.0


def test_maybe_update_prototypes_can_freeze_warmup_endpoints() -> None:
    bank = PrototypeBank(num_classes=3, num_prototypes_per_class=1, feature_dim=2)
    sinkhorn = SinkhornAssigner(eps=0.1, max_iter=5)
    features = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    labels = torch.tensor([1, 2])
    before = bank.prototypes.clone()

    updated = trainer.maybe_update_prototypes(
        bank,
        features,
        labels,
        sinkhorn,
        normalize=False,
        freeze=True,
    )

    assert updated is False
    assert torch.allclose(bank.prototypes, before)


def test_maybe_update_prototypes_updates_when_not_frozen() -> None:
    bank = PrototypeBank(num_classes=3, num_prototypes_per_class=1, feature_dim=2)
    sinkhorn = SinkhornAssigner(eps=0.1, max_iter=5)
    features = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    labels = torch.tensor([1, 2])
    before = bank.prototypes.clone()

    updated = trainer.maybe_update_prototypes(
        bank,
        features,
        labels,
        sinkhorn,
        normalize=False,
        freeze=False,
    )

    assert updated is True
    assert not torch.allclose(bank.prototypes, before)


def test_update_best_metric_tracks_ap75_independently() -> None:
    current_best = {"ap50": 0.6, "ap75": 0.30}
    candidate = {"ap50": 0.58, "ap75": 0.35}

    best_value, best_epoch, best_metrics, improved = trainer.update_best_metric(
        metric_name="ap75",
        candidate_metrics=candidate,
        epoch=3,
        best_value=0.30,
        best_epoch=1,
        best_metrics=current_best,
    )

    assert improved is True
    assert best_value == 0.35
    assert best_epoch == 3
    assert best_metrics is candidate


def test_update_best_metric_keeps_existing_when_candidate_is_not_better() -> None:
    current_best = {"ap50": 0.6, "ap75": 0.30}
    candidate = {"ap50": 0.58, "ap75": 0.29}

    best_value, best_epoch, best_metrics, improved = trainer.update_best_metric(
        metric_name="ap75",
        candidate_metrics=candidate,
        epoch=3,
        best_value=0.30,
        best_epoch=1,
        best_metrics=current_best,
    )

    assert improved is False
    assert best_value == 0.30
    assert best_epoch == 1
    assert best_metrics is current_best


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device placement check")
def test_install_active_gated_endpoint_moves_gate_to_model_device() -> None:
    device = torch.device("cuda")
    model = DummyModel(feature_dim=4, num_classes=3).to(device)
    bank = PrototypeBank(num_classes=3, num_prototypes_per_class=1, feature_dim=4).to(device)

    install_active_manifold_correction(
        model,
        prototype_bank=bank,
        gamma=0.2,
        tau=0.3,
        normalize_features=False,
        correction_mode="gated_endpoint",
        endpoint_gate_init=0.25,
    )

    gate_params = list(model.roi_heads.box_predictor.endpoint_gate_parameters())
    assert gate_params
    expected_device = bank.prototypes.device
    assert all(parameter.device == expected_device for parameter in gate_params)


def test_prototype_projection_targets_are_soft_class_endpoints() -> None:
    projection_targets = getattr(trainer, "prototype_projection_targets", None)
    assert projection_targets is not None

    class FixedAssigner:
        def __call__(self, distances):
            assert distances.shape == (2, 2)
            return torch.tensor(
                [[0.25, 0.75], [1.0, 0.0]],
                device=distances.device,
                dtype=distances.dtype,
            )

    bank = PrototypeBank(num_classes=3, num_prototypes_per_class=2, feature_dim=2)
    with torch.no_grad():
        bank.prototypes.zero_()
        bank.prototypes[1] = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    features = torch.tensor([[0.2, 0.8], [1.0, 0.1]], requires_grad=True)
    labels = torch.tensor([1, 1])

    projection = projection_targets(
        features,
        labels,
        bank,
        FixedAssigner(),
        normalize=False,
    )

    assert torch.allclose(
        projection["target_features"],
        torch.tensor([[0.25, 0.75], [1.0, 0.0]]),
    )
    assert torch.allclose(projection["assignments"].sum(dim=1), torch.ones(2))


def test_projection_geometry_intra_uses_assignment_similarity() -> None:
    geometry_losses = getattr(trainer, "projection_geometry_losses", None)
    assert geometry_losses is not None

    bank = PrototypeBank(num_classes=3, num_prototypes_per_class=2, feature_dim=2)
    endpoints = torch.tensor([[0.0, 0.0], [2.0, 0.0]], requires_grad=True)
    labels = torch.tensor([1, 1])

    same_mode = geometry_losses(
        endpoints,
        torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
        labels,
        bank,
        lambda_intra=1.0,
        lambda_proto_div=0.0,
        lambda_inter=0.0,
    )
    different_mode = geometry_losses(
        endpoints,
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        labels,
        bank,
        lambda_intra=1.0,
        lambda_proto_div=0.0,
        lambda_inter=0.0,
    )

    assert same_mode["loss_projection_intra"].item() > 3.9
    assert different_mode["loss_projection_intra"].item() == 0.0
    same_mode["loss_projection_geometry_total"].backward()
    assert endpoints.grad is not None
    assert endpoints.grad.abs().sum() > 0.0


def test_projection_geometry_penalizes_collapsed_prototypes_more_than_spread() -> None:
    geometry_losses = getattr(trainer, "projection_geometry_losses", None)
    assert geometry_losses is not None

    collapsed = PrototypeBank(num_classes=3, num_prototypes_per_class=2, feature_dim=2)
    spread = PrototypeBank(num_classes=3, num_prototypes_per_class=2, feature_dim=2)
    with torch.no_grad():
        collapsed.prototypes.zero_()
        collapsed.prototypes[1] = torch.tensor([[0.0, 0.0], [0.01, 0.0]])
        spread.prototypes.zero_()
        spread.prototypes[1] = torch.tensor([[0.0, 0.0], [2.0, 0.0]])

    endpoints = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
    assignments = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    labels = torch.tensor([1, 1])

    collapsed_losses = geometry_losses(
        endpoints,
        assignments,
        labels,
        collapsed,
        lambda_intra=0.0,
        lambda_proto_div=1.0,
        lambda_inter=0.0,
        proto_div_temperature=0.1,
    )
    spread_losses = geometry_losses(
        endpoints,
        assignments,
        labels,
        spread,
        lambda_intra=0.0,
        lambda_proto_div=1.0,
        lambda_inter=0.0,
        proto_div_temperature=0.1,
    )

    assert collapsed_losses["loss_projection_proto_div"].item() > 0.9
    assert spread_losses["loss_projection_proto_div"].item() < 1e-3


def test_projection_geometry_inter_margin_penalizes_near_wrong_class() -> None:
    geometry_losses = getattr(trainer, "projection_geometry_losses", None)
    assert geometry_losses is not None

    bank = PrototypeBank(num_classes=3, num_prototypes_per_class=1, feature_dim=2)
    with torch.no_grad():
        bank.prototypes[1] = torch.tensor([[0.0, 0.0]])
        bank.prototypes[2] = torch.tensor([[0.2, 0.0]])

    endpoints = torch.tensor([[0.0, 0.0]], requires_grad=True)
    assignments = torch.tensor([[1.0]])
    labels = torch.tensor([1])

    losses = geometry_losses(
        endpoints,
        assignments,
        labels,
        bank,
        lambda_intra=0.0,
        lambda_proto_div=0.0,
        lambda_inter=1.0,
        inter_margin=0.5,
    )

    assert losses["loss_projection_inter"].item() > 0.0
    losses["loss_projection_geometry_total"].backward()
    assert endpoints.grad is not None
    assert endpoints.grad.abs().sum() > 0.0


def test_corrected_feature_geometry_inter_margin_penalizes_wrong_class_proximity() -> None:
    corrected_losses = getattr(trainer, "corrected_feature_geometry_losses", None)
    assert corrected_losses is not None

    bank = PrototypeBank(num_classes=3, num_prototypes_per_class=1, feature_dim=2)
    with torch.no_grad():
        bank.prototypes.zero_()
        bank.prototypes[1, 0] = torch.tensor([0.0, 0.0])
        bank.prototypes[2, 0] = torch.tensor([0.2, 0.0])

    near_wrong = torch.tensor([[0.15, 0.0]], requires_grad=True)
    far_wrong = torch.tensor([[-2.0, 0.0]], requires_grad=True)
    labels = torch.tensor([1])

    near_losses = corrected_losses(
        near_wrong,
        labels,
        bank,
        lambda_intra=0.0,
        lambda_inter=1.0,
        inter_margin=0.5,
        normalize=False,
    )
    far_losses = corrected_losses(
        far_wrong,
        labels,
        bank,
        lambda_intra=0.0,
        lambda_inter=1.0,
        inter_margin=0.5,
        normalize=False,
    )

    assert near_losses["loss_corrected_inter"].item() > 0.0
    assert far_losses["loss_corrected_inter"].item() == 0.0
    near_losses["loss_corrected_geometry_total"].backward()
    assert near_wrong.grad is not None
    assert near_wrong.grad.abs().sum() > 0.0


def test_corrected_feature_geometry_inter_enforces_relative_true_class_margin() -> None:
    corrected_losses = getattr(trainer, "corrected_feature_geometry_losses", None)
    assert corrected_losses is not None

    bank = PrototypeBank(num_classes=3, num_prototypes_per_class=1, feature_dim=2)
    with torch.no_grad():
        bank.prototypes.zero_()
        bank.prototypes[1, 0] = torch.tensor([0.0, 0.0])
        bank.prototypes[2, 0] = torch.tensor([10.0, 0.0])

    closer_to_wrong = torch.tensor([[8.0, 0.0]], requires_grad=True)
    closer_to_true = torch.tensor([[1.0, 0.0]], requires_grad=True)
    labels = torch.tensor([1])

    wrong_side = corrected_losses(
        closer_to_wrong,
        labels,
        bank,
        lambda_intra=0.0,
        lambda_inter=1.0,
        inter_margin=0.5,
        normalize=False,
    )
    true_side = corrected_losses(
        closer_to_true,
        labels,
        bank,
        lambda_intra=0.0,
        lambda_inter=1.0,
        inter_margin=0.5,
        normalize=False,
    )

    assert wrong_side["loss_corrected_inter"].item() > 0.0
    assert true_side["loss_corrected_inter"].item() == 0.0
    wrong_side["loss_corrected_geometry_total"].backward()
    assert closer_to_wrong.grad is not None
    assert closer_to_wrong.grad.abs().sum() > 0.0


def test_corrected_feature_geometry_intra_penalizes_same_class_spread() -> None:
    corrected_losses = getattr(trainer, "corrected_feature_geometry_losses", None)
    assert corrected_losses is not None

    bank = PrototypeBank(num_classes=3, num_prototypes_per_class=1, feature_dim=2)
    spread = torch.tensor([[0.0, 0.0], [2.0, 0.0]], requires_grad=True)
    compact = torch.tensor([[1.0, 0.0], [1.0, 0.0]], requires_grad=True)
    labels = torch.tensor([1, 1])

    spread_losses = corrected_losses(
        spread,
        labels,
        bank,
        lambda_intra=1.0,
        lambda_inter=0.0,
        inter_margin=0.5,
        normalize=False,
    )
    compact_losses = corrected_losses(
        compact,
        labels,
        bank,
        lambda_intra=1.0,
        lambda_inter=0.0,
        inter_margin=0.5,
        normalize=False,
    )

    assert spread_losses["loss_corrected_intra"].item() > compact_losses["loss_corrected_intra"].item()
    spread_losses["loss_corrected_geometry_total"].backward()
    assert spread.grad is not None
    assert spread.grad.abs().sum() > 0.0


def test_corrected_feature_geometry_inter_penalizes_close_class_centroids() -> None:
    corrected_losses = getattr(trainer, "corrected_feature_geometry_losses", None)
    assert corrected_losses is not None

    bank = PrototypeBank(num_classes=3, num_prototypes_per_class=1, feature_dim=2)
    with torch.no_grad():
        bank.prototypes.zero_()
        bank.prototypes[1, 0] = torch.tensor([10.0, 10.0])
        bank.prototypes[2, 0] = torch.tensor([10.0, -10.0])
    close_centroids = torch.tensor(
        [[0.0, 0.0], [0.1, 0.0], [0.2, 0.0], [0.3, 0.0]],
        requires_grad=True,
    )
    far_centroids = torch.tensor(
        [[0.0, 0.0], [0.0, 0.0], [2.0, 0.0], [2.0, 0.0]],
        requires_grad=True,
    )
    labels = torch.tensor([1, 1, 2, 2])

    close_losses = corrected_losses(
        close_centroids,
        labels,
        bank,
        lambda_intra=0.0,
        lambda_inter=1.0,
        inter_margin=0.5,
        normalize=False,
    )
    far_losses = corrected_losses(
        far_centroids,
        labels,
        bank,
        lambda_intra=0.0,
        lambda_inter=1.0,
        inter_margin=0.5,
        normalize=False,
    )

    assert close_losses["loss_corrected_inter"].item() > far_losses["loss_corrected_inter"].item()
    close_losses["loss_corrected_geometry_total"].backward()
    assert close_centroids.grad is not None
    assert close_centroids.grad.abs().sum() > 0.0


def test_corrected_feature_geometry_preserves_reference_class_centroid_distances() -> None:
    corrected_losses = getattr(trainer, "corrected_feature_geometry_losses", None)
    assert corrected_losses is not None

    bank = PrototypeBank(num_classes=3, num_prototypes_per_class=1, feature_dim=2)
    reference = torch.tensor(
        [[0.0, 0.0], [0.0, 0.0], [2.0, 0.0], [2.0, 0.0]],
    )
    collapsed = torch.tensor(
        [[0.0, 0.0], [0.0, 0.0], [0.2, 0.0], [0.2, 0.0]],
        requires_grad=True,
    )
    preserved = torch.tensor(
        [[0.0, 0.0], [0.0, 0.0], [2.1, 0.0], [2.1, 0.0]],
        requires_grad=True,
    )
    labels = torch.tensor([1, 1, 2, 2])

    collapsed_losses = corrected_losses(
        collapsed,
        labels,
        bank,
        lambda_intra=0.0,
        lambda_inter=0.0,
        inter_margin=0.5,
        normalize=False,
        reference_features=reference,
        lambda_inter_preserve=1.0,
    )
    preserved_losses = corrected_losses(
        preserved,
        labels,
        bank,
        lambda_intra=0.0,
        lambda_inter=0.0,
        inter_margin=0.5,
        normalize=False,
        reference_features=reference,
        lambda_inter_preserve=1.0,
    )

    assert collapsed_losses["loss_corrected_inter_preserve"].item() > 0.0
    assert preserved_losses["loss_corrected_inter_preserve"].item() == 0.0
    collapsed_losses["loss_corrected_geometry_total"].backward()
    assert collapsed.grad is not None
    assert collapsed.grad.abs().sum() > 0.0


def test_corrected_feature_geometry_preserves_reference_class_centroid_positions() -> None:
    corrected_losses = getattr(trainer, "corrected_feature_geometry_losses", None)
    assert corrected_losses is not None

    bank = PrototypeBank(num_classes=3, num_prototypes_per_class=1, feature_dim=2)
    reference = torch.tensor(
        [[0.0, 0.0], [0.0, 0.0], [2.0, 0.0], [2.0, 0.0]],
    )
    shifted = torch.tensor(
        [[1.0, 0.0], [1.0, 0.0], [3.0, 0.0], [3.0, 0.0]],
        requires_grad=True,
    )
    preserved = torch.tensor(
        [[0.0, 0.0], [0.0, 0.0], [2.0, 0.0], [2.0, 0.0]],
        requires_grad=True,
    )
    labels = torch.tensor([1, 1, 2, 2])

    shifted_losses = corrected_losses(
        shifted,
        labels,
        bank,
        lambda_intra=0.0,
        lambda_inter=0.0,
        inter_margin=0.5,
        normalize=False,
        reference_features=reference,
        lambda_center_preserve=1.0,
    )
    preserved_losses = corrected_losses(
        preserved,
        labels,
        bank,
        lambda_intra=0.0,
        lambda_inter=0.0,
        inter_margin=0.5,
        normalize=False,
        reference_features=reference,
        lambda_center_preserve=1.0,
    )

    assert shifted_losses["loss_corrected_center_preserve"].item() > 0.0
    assert shifted_losses["loss_corrected_center_preserve"].item() == pytest.approx(1.0)
    assert preserved_losses["loss_corrected_center_preserve"].item() == 0.0
    shifted_losses["loss_corrected_geometry_total"].backward()
    assert shifted.grad is not None
    assert shifted.grad.abs().sum() > 0.0


def test_corrected_feature_geometry_uses_memory_class_centroid_anchor() -> None:
    corrected_losses = getattr(trainer, "corrected_feature_geometry_losses", None)
    centroid_memory_cls = getattr(trainer, "ClassCentroidMemory", None)
    assert corrected_losses is not None
    assert centroid_memory_cls is not None

    memory = centroid_memory_cls(num_classes=3, feature_dim=2, momentum=0.5, device=torch.device("cpu"))
    memory.update(
        torch.tensor([[0.0, 0.0], [0.0, 0.0], [2.0, 0.0], [2.0, 0.0]]),
        torch.tensor([1, 1, 2, 2]),
        normalize=False,
    )
    memory.update(
        torch.tensor([[2.0, 0.0], [2.0, 0.0]]),
        torch.tensor([1, 1]),
        normalize=False,
    )

    assert torch.allclose(memory.centers[1], torch.tensor([1.0, 0.0]))
    assert torch.allclose(memory.centers[2], torch.tensor([2.0, 0.0]))

    bank = PrototypeBank(num_classes=3, num_prototypes_per_class=1, feature_dim=2)
    shifted = torch.tensor(
        [[2.0, 0.0], [2.0, 0.0], [3.0, 0.0], [3.0, 0.0]],
        requires_grad=True,
    )
    preserved = torch.tensor(
        [[1.0, 0.0], [1.0, 0.0], [2.0, 0.0], [2.0, 0.0]],
        requires_grad=True,
    )
    labels = torch.tensor([1, 1, 2, 2])

    shifted_losses = corrected_losses(
        shifted,
        labels,
        bank,
        lambda_intra=0.0,
        lambda_inter=0.0,
        inter_margin=0.5,
        normalize=False,
        centroid_memory=memory,
        lambda_memory_center_preserve=1.0,
    )
    preserved_losses = corrected_losses(
        preserved,
        labels,
        bank,
        lambda_intra=0.0,
        lambda_inter=0.0,
        inter_margin=0.5,
        normalize=False,
        centroid_memory=memory,
        lambda_memory_center_preserve=1.0,
    )

    assert shifted_losses["loss_corrected_memory_center_preserve"].item() == pytest.approx(1.0)
    assert preserved_losses["loss_corrected_memory_center_preserve"].item() == 0.0
    shifted_losses["loss_corrected_geometry_total"].backward()
    assert shifted.grad is not None
    assert shifted.grad.abs().sum() > 0.0


def test_corrected_feature_geometry_preserves_memory_class_centroid_distances() -> None:
    corrected_losses = getattr(trainer, "corrected_feature_geometry_losses", None)
    centroid_memory_cls = getattr(trainer, "ClassCentroidMemory", None)
    assert corrected_losses is not None
    assert centroid_memory_cls is not None

    memory = centroid_memory_cls(num_classes=3, feature_dim=2, momentum=0.5, device=torch.device("cpu"))
    memory.update(
        torch.tensor([[0.0, 0.0], [0.0, 0.0], [2.0, 0.0], [2.0, 0.0]]),
        torch.tensor([1, 1, 2, 2]),
        normalize=False,
    )

    bank = PrototypeBank(num_classes=3, num_prototypes_per_class=1, feature_dim=2)
    collapsed = torch.tensor(
        [[0.0, 0.0], [0.0, 0.0], [0.2, 0.0], [0.2, 0.0]],
        requires_grad=True,
    )
    preserved = torch.tensor(
        [[0.0, 0.0], [0.0, 0.0], [2.1, 0.0], [2.1, 0.0]],
        requires_grad=True,
    )
    labels = torch.tensor([1, 1, 2, 2])

    collapsed_losses = corrected_losses(
        collapsed,
        labels,
        bank,
        lambda_intra=0.0,
        lambda_inter=0.0,
        inter_margin=0.5,
        normalize=False,
        centroid_memory=memory,
        lambda_memory_inter_preserve=1.0,
    )
    preserved_losses = corrected_losses(
        preserved,
        labels,
        bank,
        lambda_intra=0.0,
        lambda_inter=0.0,
        inter_margin=0.5,
        normalize=False,
        centroid_memory=memory,
        lambda_memory_inter_preserve=1.0,
    )

    assert collapsed_losses["loss_corrected_memory_inter_preserve"].item() > 0.0
    assert preserved_losses["loss_corrected_memory_inter_preserve"].item() == 0.0
    collapsed_losses["loss_corrected_geometry_total"].backward()
    assert collapsed.grad is not None
    assert collapsed.grad.abs().sum() > 0.0


def test_spectral_tail_loss_penalizes_extra_active_dimensions() -> None:
    tail_loss = getattr(trainer, "spectral_tail_loss", None)
    assert tail_loss is not None

    base = torch.randn(16, 3)
    low_rank = torch.cat([base, torch.zeros(16, 5)], dim=1)
    high_rank = torch.randn(16, 8)

    low = tail_loss(low_rank, rank=3, normalize=False)
    high = tail_loss(high_rank, rank=3, normalize=False)

    assert low.item() < 1e-6
    assert high.item() > low.item() + 0.05


def test_fc1_geometry_losses_backpropagate_rank_and_compactness_terms() -> None:
    fc1_losses = getattr(trainer, "fc1_geometry_losses", None)
    assert fc1_losses is not None

    features = torch.randn(12, 8, requires_grad=True)
    labels = torch.tensor([1, 1, 1, 2, 2, 2, 3, 3, 3, 1, 2, 3])

    losses = fc1_losses(
        features,
        labels,
        rank=3,
        lambda_rank=0.5,
        lambda_compact=0.25,
        normalize=False,
    )

    assert losses["loss_fc1_geometry_total"].requires_grad
    assert losses["loss_fc1_rank"].item() > 0.0
    assert losses["loss_fc1_compact"].item() > 0.0

    losses["loss_fc1_geometry_total"].backward()

    assert features.grad is not None
    assert features.grad.abs().sum() > 0.0


def test_prediction_preservation_losses_detach_teacher_and_backprop_student() -> None:
    preserve_losses = getattr(trainer, "prediction_preservation_losses", None)
    assert preserve_losses is not None

    student_logits = torch.randn(5, 4, requires_grad=True)
    student_bbox = torch.randn(5, 16, requires_grad=True)
    teacher_logits = torch.randn(5, 4, requires_grad=True)
    teacher_bbox = torch.randn(5, 16, requires_grad=True)
    labels = torch.tensor([0, 1, 2, 3, 1])

    losses = preserve_losses(
        student_logits,
        student_bbox,
        teacher_logits,
        teacher_bbox,
        labels,
        lambda_logits=0.3,
        lambda_bbox=0.7,
        temperature=2.0,
    )

    assert losses["loss_preserve_total"].requires_grad
    assert losses["loss_preserve_logits"].item() > 0.0
    assert losses["loss_preserve_bbox"].item() > 0.0

    losses["loss_preserve_total"].backward()

    assert student_logits.grad is not None
    assert student_logits.grad.abs().sum() > 0.0
    assert student_bbox.grad is not None
    assert student_bbox.grad.abs().sum() > 0.0
    assert teacher_logits.grad is None
    assert teacher_bbox.grad is None


def test_prediction_preservation_bbox_uses_only_foreground_label_slice() -> None:
    preserve_losses = getattr(trainer, "prediction_preservation_losses", None)
    assert preserve_losses is not None

    student_logits = torch.zeros(2, 3, requires_grad=True)
    teacher_logits = torch.zeros(2, 3)
    student_bbox = torch.zeros(2, 12, requires_grad=True)
    teacher_bbox = torch.zeros(2, 12)
    labels = torch.tensor([1, 2])

    # Other class slices differ, but each sample's labelled slice is identical.
    teacher_bbox[0, 8:12] = 10.0
    teacher_bbox[1, 4:8] = -10.0

    losses = preserve_losses(
        student_logits,
        student_bbox,
        teacher_logits,
        teacher_bbox,
        labels,
        lambda_logits=0.0,
        lambda_bbox=1.0,
        temperature=1.0,
    )

    assert losses["loss_preserve_bbox"].item() == 0.0


def test_eval_metrics_returns_full_detection_metrics(monkeypatch) -> None:
    class EvalModel(nn.Module):
        def forward(self, images):
            return [
                {
                    "boxes": torch.zeros((0, 4)),
                    "scores": torch.zeros((0,)),
                    "labels": torch.zeros((0,), dtype=torch.long),
                }
                for _ in images
            ]

    def fake_evaluate(predictions, targets, **kwargs):
        return {"ap50": 0.5, "ap75": 0.25, "ece": 0.1}

    monkeypatch.setattr(trainer, "evaluate_detection_predictions", fake_evaluate)
    metrics_fn = getattr(trainer, "eval_metrics", None)
    assert metrics_fn is not None

    val_loader = [
        (
            [torch.zeros(3, 8, 8)],
            [{"boxes": torch.zeros((0, 4)), "labels": torch.zeros((0,), dtype=torch.long)}],
        )
    ]
    config = {"matching": {}, "eval": {}}
    metrics = metrics_fn(EvalModel(), val_loader, torch.device("cpu"), config, num_classes=4)

    assert metrics["ap50"] == 0.5
    assert metrics["ap75"] == 0.25
    assert metrics["ece"] == 0.1
