"""Unit tests for Plan F remote-sensing spectral methods."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from spectral_detection_posttrain.methods.remote_sensing import (
    MultiScaleSpectralHead,
    RemoteSensingAFM,
    RemoteSensingManifold,
    RotationEquivariantFFT,
    compute_ap,
    evaluate_remote_sensing_ap,
)


def test_remote_sensing_afm_shape_and_gradient():
    """RemoteSensingAFM preserves spatial resolution and supports backprop."""
    module = RemoteSensingAFM(channels=16, scales=[1, 2, 4], gate_strength=0.6)
    x = torch.randn(2, 16, 32, 32, requires_grad=True)
    y = module(x)

    assert y.shape == x.shape
    assert torch.isfinite(y).all()

    loss = y.abs().pow(2).sum()
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    for p in module.parameters():
        assert p.grad is not None
        assert torch.isfinite(p.grad).all()


def test_remote_sensing_afm_multi_scale_fusion_preserves_size():
    """A pyramid with downsampling still produces an output at the input size."""
    module = RemoteSensingAFM(channels=8, scales=[1, 2])
    for size in [(16, 16), (15, 31), (8, 8)]:
        x = torch.randn(1, 8, *size)
        y = module(x)
        assert y.shape == x.shape


def test_remote_sensing_afm_invalid_input():
    """RemoteSensingAFM rejects non-4D inputs and invalid scales."""
    with pytest.raises(ValueError):
        RemoteSensingAFM(channels=8, scales=[0, 1])

    module = RemoteSensingAFM(channels=8)
    with pytest.raises(ValueError):
        module(torch.randn(8, 8, 8))


def test_rotation_equivariant_fft_shape_and_gradient():
    """RotationEquivariantFFT returns an equally-sized feature map and gradients."""
    module = RotationEquivariantFFT(channels=8, n_angles=4, pool="mean")
    x = torch.randn(2, 8, 16, 16, requires_grad=True)
    y = module(x)

    assert y.shape == x.shape
    assert torch.isfinite(y).all()

    loss = y.abs().pow(2).sum()
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_rotation_equivariant_fft_single_angle_identity():
    """With one angle the module reduces to its internal spectral residual."""
    module = RotationEquivariantFFT(channels=4, n_angles=1, pool="mean", gate_strength=0.0)
    x = torch.randn(1, 4, 16, 16)

    with torch.no_grad():
        y = module(x)
        expected = module._spectral_residual(x)

    assert y.shape == x.shape
    assert torch.allclose(y, expected, atol=1e-5)


def test_rotation_equivariant_fft_invalid_args():
    """RotationEquivariantFFT validates its hyperparameters."""
    with pytest.raises(ValueError):
        RotationEquivariantFFT(channels=4, n_angles=0)
    with pytest.raises(ValueError):
        RotationEquivariantFFT(channels=4, n_angles=4, pool="median")


def test_remote_sensing_manifold_inherits_identity():
    """The manifold without coordinate inputs is the Plan A identity manifold."""
    dim = 16
    manifold = RemoteSensingManifold(
        in_dim=dim,
        latent_dim=dim,
        n_scales=4,
        n_orientations=8,
        n_classes=3,
    )
    F = torch.randn(2, dim, dtype=torch.cfloat)
    F_recon = manifold(F)
    assert F_recon.shape == F.shape
    mse = (F_recon - F).abs().pow(2).mean().item()
    assert mse < 1e-3


def test_remote_sensing_manifold_coordinate_embeddings():
    """Scale/orientation/class embeddings modify the encoded latent coordinate."""
    dim = 16
    manifold = RemoteSensingManifold(
        in_dim=dim,
        latent_dim=dim,
        n_scales=3,
        n_orientations=4,
        n_classes=2,
    )
    # Initialize embeddings to non-zero values so the test can observe a change.
    for emb in (manifold.scale_embed, manifold.orientation_embed, manifold.class_embed):
        nn.init.normal_(emb.weight)

    F = torch.randn(2, dim, dtype=torch.cfloat)

    z_plain = manifold.encode(F)
    z_rich = manifold.encode(
        F,
        scale_idx=torch.tensor([0, 2]),
        orientation_idx=torch.tensor([1, 3]),
        class_idx=torch.tensor([0, 1]),
    )

    assert z_rich.shape == z_plain.shape
    assert not torch.allclose(z_plain, z_rich)

    # Auto-encoding with coordinates is differentiable.
    manifold.zero_grad()
    F_recon = manifold(F, torch.tensor([0]), torch.tensor([1]), torch.tensor([0]))
    loss = F_recon.abs().pow(2).sum()
    loss.backward()
    for emb in (manifold.scale_embed, manifold.orientation_embed, manifold.class_embed):
        assert emb.weight.grad is not None
        assert torch.isfinite(emb.weight.grad).all()


def test_remote_sensing_manifold_invalid_input():
    """The manifold still rejects non-complex inputs inherited from Plan A."""
    manifold = RemoteSensingManifold(
        in_dim=8,
        latent_dim=8,
        n_scales=2,
        n_orientations=2,
    )
    with pytest.raises(ValueError):
        manifold.encode(torch.randn(2, 8))


def test_multiscale_spectral_head_outputs():
    """The head applies spectral augmentation to every FPN level."""
    in_channels = [16, 32, 64]
    features = [
        torch.randn(2, 16, 32, 32),
        torch.randn(2, 32, 16, 16),
        torch.randn(2, 64, 8, 8),
    ]

    head_no_proj = MultiScaleSpectralHead(in_channels)
    out_no_proj = head_no_proj(features)
    assert len(out_no_proj) == len(features)
    for y, x in zip(out_no_proj, features):
        assert y.shape == x.shape

    head_proj = MultiScaleSpectralHead(in_channels, out_channels=128)
    out_proj = head_proj(features)
    assert len(out_proj) == len(features)
    for y in out_proj:
        assert y.shape[1] == 128


def test_multiscale_spectral_head_gradient():
    """The multi-scale head supports backpropagation."""
    in_channels = [8, 16]
    features = [
        torch.randn(1, 8, 8, 8, requires_grad=True),
        torch.randn(1, 16, 4, 4, requires_grad=True),
    ]
    head = MultiScaleSpectralHead(in_channels, out_channels=8)
    outputs = head(features)
    loss = sum(o.abs().pow(2).sum() for o in outputs)
    loss.backward()
    for x in features:
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()


def test_multiscale_spectral_head_length_mismatch():
    """The head rejects a feature list of the wrong length."""
    head = MultiScaleSpectralHead([8, 16])
    with pytest.raises(ValueError):
        head([torch.randn(1, 8, 8, 8)])


def test_compute_ap_perfect():
    """compute_ap returns 1.0 for a perfect precision-recall curve."""
    recall = torch.tensor([0.0, 0.5, 1.0])
    precision = torch.tensor([1.0, 1.0, 1.0])
    assert compute_ap(recall, precision) == pytest.approx(1.0, abs=1e-6)


def test_compute_ap_empty():
    """compute_ap returns 0.0 for empty inputs."""
    assert compute_ap(torch.tensor([]), torch.tensor([])) == 0.0


def test_box_iou():
    """Pairwise IoU is correct for overlapping and disjoint boxes."""
    boxes1 = torch.tensor([[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 10.0, 10.0]])
    boxes2 = torch.tensor(
        [
            [5.0, 5.0, 15.0, 15.0],  # 25 / 175
            [20.0, 20.0, 30.0, 30.0],  # disjoint
        ]
    )
    ious = evaluate_remote_sensing_ap.__module__  # not an iou helper
    # Reach the internal helper through the module object.
    from spectral_detection_posttrain.methods.remote_sensing import eval_remote_sensing

    iou_matrix = eval_remote_sensing._box_iou(boxes1, boxes2)
    assert iou_matrix.shape == (2, 2)
    assert iou_matrix[0, 0].item() == pytest.approx(25.0 / 175.0, abs=1e-5)
    assert iou_matrix[0, 1].item() == pytest.approx(0.0, abs=1e-6)


def test_evaluate_remote_sensing_ap_perfect():
    """A perfect prediction set yields AP = 1.0 at IoU 0.5."""
    gt = {
        "img1": [
            {"bbox": torch.tensor([0.0, 0.0, 10.0, 10.0]), "score": torch.tensor(1.0)}
        ]
    }
    pred = {
        "img1": [
            {"bbox": torch.tensor([0.0, 0.0, 10.0, 10.0]), "score": torch.tensor(0.9)}
        ]
    }
    metrics = evaluate_remote_sensing_ap(pred, gt, iou_threshold=0.5)
    assert metrics["AP"] == pytest.approx(1.0, abs=1e-6)
    assert metrics["recall"] == pytest.approx(1.0, abs=1e-6)


def test_evaluate_remote_sensing_ap_empty_predictions():
    """No predictions yields zero AP/recall."""
    gt = {
        "img1": [
            {"bbox": torch.tensor([0.0, 0.0, 10.0, 10.0]), "score": torch.tensor(1.0)}
        ]
    }
    pred = {"img1": []}
    metrics = evaluate_remote_sensing_ap(pred, gt)
    assert metrics["AP"] == 0.0
    assert metrics["recall"] == 0.0
