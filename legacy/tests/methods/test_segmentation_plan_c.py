"""Unit tests for Plan C semantic-segmentation modules."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from spectral_detection_posttrain.methods.segmentation import (
    ManifoldAFMBlock,
    ManifoldAFMStack,
    OTSegmentationLoss,
    boundary_iou,
    eval_segmentation,
    mean_iou,
    per_class_iou,
    pixel_accuracy,
)


def _image_like(batch: int, channels: int, height: int, width: int) -> torch.Tensor:
    """Create a deterministic real-valued feature map for tests."""
    torch.manual_seed(42)
    return torch.randn(batch, channels, height, width, requires_grad=True)


def _labels_like(batch: int, height: int, width: int, num_classes: int) -> torch.Tensor:
    """Create deterministic integer labels for tests."""
    torch.manual_seed(43)
    return torch.randint(0, num_classes, (batch, height, width))


class TestManifoldAFMBlock:
    """Tests for the manifold-aware AFM block."""

    def test_forward_shape(self):
        """Output shape matches input shape."""
        block = ManifoldAFMBlock(channels=64, latent_dim=16)
        x = _image_like(2, 64, 8, 8)
        y = block(x)
        assert y.shape == x.shape

    def test_channel_grouping(self):
        """Block accepts channels that are multiples of latent_dim."""
        block = ManifoldAFMBlock(channels=128, latent_dim=32)
        x = _image_like(1, 128, 4, 4)
        y = block(x)
        assert y.shape == x.shape

    def test_invalid_channels_raises(self):
        """Channels not divisible by latent_dim raises an error."""
        with pytest.raises(ValueError):
            ManifoldAFMBlock(channels=100, latent_dim=32)

    def test_identity_at_init(self):
        """With zero gates and identity manifold, output is close to input."""
        block = ManifoldAFMBlock(channels=64, latent_dim=16, gate_strength=0.0)
        x = _image_like(2, 64, 8, 8)
        with torch.no_grad():
            y = block(x)
        rel_err = (y - x).abs().mean().item() / (x.abs().mean().item() + 1e-8)
        assert rel_err < 0.05

    def test_gradient_flow(self):
        """Backpropagation reaches every parameter used in the forward path."""
        block = ManifoldAFMBlock(channels=64, latent_dim=16)
        x = _image_like(2, 64, 8, 8)
        y = block(x)
        loss = y.pow(2).sum()
        loss.backward()

        assert x.grad is not None
        assert torch.isfinite(x.grad).all()
        for name, p in block.named_parameters():
            # ChordTransport stores the metric for API compatibility but does not
            # use it in its single-step forward, so its parameters stay unused.
            if "chord_transport.metric" in name:
                continue
            assert p.grad is not None, f"{name} has no gradient"
            assert torch.isfinite(p.grad).all(), f"{name} gradient non-finite"

    def test_chord_transport_toggle(self):
        """Block works with and without Chord transport."""
        for use_transport in (True, False):
            block = ManifoldAFMBlock(
                channels=64, latent_dim=16, use_chord_transport=use_transport
            )
            x = _image_like(1, 64, 8, 8)
            y = block(x)
            assert y.shape == x.shape

    def test_spatial_dims_preserved(self):
        """Non-square inputs are handled correctly."""
        block = ManifoldAFMBlock(channels=32, latent_dim=16)
        x = _image_like(1, 32, 4, 7)
        y = block(x)
        assert y.shape == (1, 32, 4, 7)


class TestManifoldAFMStack:
    """Tests for the multi-level manifold AFM stack."""

    def test_stack_forward(self):
        """Stack applies the correct per-level block."""
        stack = ManifoldAFMStack(channels=[32, 64, 128], latent_dim=32)
        for level, c in enumerate([32, 64, 128]):
            x = _image_like(1, c, 8, 8)
            y = stack(x, level=level)
            assert y.shape == x.shape

    def test_per_level_gate_strength(self):
        """Per-level gate strengths are accepted."""
        stack = ManifoldAFMStack(
            channels=[32, 64], latent_dim=32, gate_strength=[0.3, 0.6]
        )
        assert stack.blocks["0"].gate_strength == pytest.approx(0.3)
        assert stack.blocks["1"].gate_strength == pytest.approx(0.6)

    def test_mismatched_gate_length_raises(self):
        """Mismatched gate_strength length raises an error."""
        with pytest.raises(ValueError):
            ManifoldAFMStack(channels=[32, 64], latent_dim=32, gate_strength=[0.3])


class TestOTSegmentationLoss:
    """Tests for the OT segmentation loss."""

    def test_forward_shape_and_type(self):
        """Loss returns a scalar tensor."""
        loss_fn = OTSegmentationLoss(num_classes=4, sample_count=16)
        pred = torch.randn(2, 4, 8, 8, requires_grad=True)
        target = _labels_like(2, 8, 8, 4)
        loss = loss_fn(pred, target)
        assert loss.shape == ()
        assert torch.isfinite(loss)

    def test_random_and_uncertain_sampling(self):
        """Both sampling modes run without error."""
        for mode in ("random", "uncertain"):
            loss_fn = OTSegmentationLoss(
                num_classes=4, sample_count=16, sample_mode=mode
            )
            pred = torch.randn(2, 4, 8, 8)
            target = _labels_like(2, 8, 8, 4)
            loss = loss_fn(pred, target)
            assert torch.isfinite(loss)

    def test_gradient_flow(self):
        """Gradients flow back to the logits."""
        loss_fn = OTSegmentationLoss(num_classes=4, sample_count=16)
        pred = torch.randn(2, 4, 8, 8, requires_grad=True)
        target = _labels_like(2, 8, 8, 4)
        loss = loss_fn(pred, target)
        loss.backward()
        assert pred.grad is not None
        assert torch.isfinite(pred.grad).all()

    def test_perfect_prediction_low_loss(self):
        """Perfect predictions give lower loss than random ones."""
        torch.manual_seed(0)
        loss_fn = OTSegmentationLoss(num_classes=4, sample_count=16)
        target = _labels_like(2, 8, 8, 4)
        # Perfect logits: large value at the target class.
        perfect = torch.zeros(2, 4, 8, 8)
        perfect.scatter_(1, target.unsqueeze(1), 10.0)
        random = torch.randn(2, 4, 8, 8)
        loss_perfect = loss_fn(perfect, target).item()
        loss_random = loss_fn(random, target).item()
        assert loss_perfect < loss_random

    def test_ignore_index(self):
        """Ignored pixels are excluded from sampling."""
        loss_fn = OTSegmentationLoss(
            num_classes=4, sample_count=16, sample_mode="uncertain", ignore_index=255
        )
        pred = torch.randn(2, 4, 8, 8)
        target = _labels_like(2, 8, 8, 4)
        target[:, 0, :] = 255
        loss = loss_fn(pred, target)
        assert torch.isfinite(loss)

    def test_invalid_num_classes_raises(self):
        """num_classes <= 1 raises an error."""
        with pytest.raises(ValueError):
            OTSegmentationLoss(num_classes=1)


class TestEvalSegmentation:
    """Tests for segmentation evaluation metrics."""

    def test_perfect_mean_iou(self):
        """Perfect predictions yield mIoU of 1."""
        target = _labels_like(2, 8, 8, 4)
        pred = target.clone()
        iou = mean_iou(pred, target, num_classes=4)
        assert iou.item() == pytest.approx(1.0, abs=1e-6)

    def test_per_class_iou_shape(self):
        """per_class_iou returns one value per class."""
        target = _labels_like(2, 8, 8, 4)
        pred = target.clone()
        iou = per_class_iou(pred, target, num_classes=4)
        assert iou.shape == (4,)

    def test_pixel_accuracy(self):
        """Pixel accuracy equals 1 for perfect predictions."""
        target = _labels_like(2, 8, 8, 4)
        pred = target.clone()
        acc = pixel_accuracy(pred, target)
        assert acc.item() == pytest.approx(1.0, abs=1e-6)

    def test_mean_iou_with_logits(self):
        """mean_iou accepts logits and converts them to labels."""
        target = _labels_like(2, 8, 8, 4)
        logits = F.one_hot(target, num_classes=4).permute(0, 3, 1, 2).float()
        logits = logits * 10.0
        iou = mean_iou(logits, target, num_classes=4)
        assert iou.item() == pytest.approx(1.0, abs=1e-6)

    def test_boundary_iou_runs(self):
        """boundary_iou returns a scalar for valid inputs."""
        target = _labels_like(2, 8, 8, 4)
        pred = target.clone()
        biou = boundary_iou(pred, target, num_classes=4, boundary_width=1)
        assert biou.shape == ()
        assert 0.0 <= biou.item() <= 1.0

    def test_eval_segmentation_returns_dict(self):
        """eval_segmentation returns all expected metrics."""
        target = _labels_like(2, 8, 8, 4)
        pred = target.clone()
        metrics = eval_segmentation(pred, target, num_classes=4)
        assert set(metrics.keys()) == {
            "mIoU",
            "boundary_iou",
            "pixel_accuracy",
            "per_class_iou",
        }
        assert metrics["mIoU"].item() == pytest.approx(1.0, abs=1e-6)

    def test_ignore_index(self):
        """ignore_index masks are respected by the metrics."""
        target = _labels_like(2, 8, 8, 4)
        pred = target.clone()
        # Set a strip to ignore; predictions remain correct for non-ignored.
        target[:, 0, :] = 255
        acc = pixel_accuracy(pred, target, ignore_index=255)
        assert acc.item() == pytest.approx(1.0, abs=1e-6)
