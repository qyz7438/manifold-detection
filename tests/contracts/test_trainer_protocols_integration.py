"""Fake-data CPU integration test for the trainer protocols (plan Task 15 step 4).

Exercises the full adapter pipeline end-to-end without downloading weights or
data: prepare -> train adapter -> eval adapter -> checkpoint reload -> metrics
-> manifest completion.

- The detector is a real torchvision Faster R-CNN (MobileNetV3-320) built via
  the maintained ``build_detector`` with ``pretrained=False`` — no downloads.
- The dataset is a deterministic fake loader injected through the adapters'
  ``loader_builder`` seam: one 320x320 image with a bright square and one
  matching GT box.
- ``ActionTransportTrainer`` runs its real train/eval loops (proposal extract
  -> supervised action loss -> optimizer step -> native-postprocess eval).
  ``StandardDetectionTrainer`` runs the maintained ``train_one_epoch`` over
  the same fake loader.

Scope: the full-featured action CLI (high-water-mark teacher, zero-action
parity diagnostics) and the Penn-Fudan/NWPU loader paths still need real
datasets; this test covers the protocol adapter path only, on purpose.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import pytest
import torch
import yaml

from spectral_detection_posttrain.core.models.build_detector import build_detector
from spectral_detection_posttrain.experiments.artifacts import load_artifact_manifest
from spectral_detection_posttrain.experiments.lifecycle import RUNTIME_MANIFEST_NAME
from spectral_detection_posttrain.methods.energy_transport import (
    ActionLocalTransportHead,
    ROIActionState,
)
from spectral_detection_posttrain.trainers.detection.action_local_transport import (
    ProposalActionBatch,
    SupervisedActionLossConfig,
    supervised_action_transport_loss,
)
from spectral_detection_posttrain.trainers.detection.action_transport import ActionTransportTrainer
from spectral_detection_posttrain.trainers.detection.protocols import TrainRequest, TrainResult
from spectral_detection_posttrain.trainers.detection.standard import StandardDetectionTrainer
from spectral_detection_posttrain.utils.config import load_config
from spectral_detection_posttrain.utils.io import load_checkpoint

IMAGE_SIZE = 320


def _fake_config() -> dict:
    return {
        "seed": 42,
        "device": "cpu",
        "data": {"root": "./data", "train_fraction": 0.8},
        "model": {
            "name": "fasterrcnn_mobilenet_v3_large_320_fpn",
            "pretrained": False,
            "allow_random_init_fallback": False,
            "num_classes": 2,
            "min_size": IMAGE_SIZE,
            "max_size": IMAGE_SIZE,
        },
        "train": {"batch_size": 1, "epochs": 1, "lr": 1e-3},
        "matching": {"iou_threshold": 0.5, "score_threshold": 0.05},
        "eval": {"batch_size": 1, "high_conf_threshold": 0.7},
    }


def _write_config(tmp_path: Path) -> Path:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(_fake_config(), sort_keys=False), encoding="utf-8")
    return config_path


def _fake_batch() -> tuple[list[torch.Tensor], list[dict]]:
    generator = torch.Generator().manual_seed(7)
    image = 0.05 * torch.rand(3, IMAGE_SIZE, IMAGE_SIZE, generator=generator)
    image[:, 80:240, 80:240] = 0.9
    target = {
        "boxes": torch.tensor([[80.0, 80.0, 240.0, 240.0]], dtype=torch.float32),
        "labels": torch.tensor([1], dtype=torch.int64),
    }
    return [image], [target]


def _fake_loader_builder(config: dict, *, limit_train: int | None, limit_val: int | None, batch_size: int):
    batch = _fake_batch()
    return [batch], [batch]


def _synthetic_action_batch(feature_dim: int) -> ProposalActionBatch:
    generator = torch.Generator().manual_seed(3)
    features = torch.randn(4, feature_dim, generator=generator)
    state = ROIActionState(
        features=features,
        boxes=torch.tensor(
            [[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 40.0, 40.0],
             [50.0, 50.0, 80.0, 80.0], [5.0, 5.0, 15.0, 15.0]]
        ),
        scores=torch.tensor([0.9, 0.8, 0.7, 0.1]),
        labels=torch.tensor([1, 1, 1, 0]),
        image_indices=torch.zeros(4, dtype=torch.int64),
        proposal_indices=torch.arange(4),
        matched_gt_indices=torch.tensor([0, 1, 2, -1]),
        ious=torch.tensor([0.70, 0.90, 0.74, 0.0]),
    )
    return ProposalActionBatch(
        state=state,
        matched_gt_boxes=torch.tensor(
            [[1.0, 1.0, 11.0, 11.0], [20.0, 20.0, 40.0, 40.0],
             [52.0, 52.0, 82.0, 82.0], [0.0, 0.0, 0.0, 0.0]]
        ),
        matched_gt_labels=torch.tensor([1, 1, 1, 0]),
        image_sizes=[(100, 100)],
    )


# ---------------------------------------------------------------------------
# Action-transport adapter end-to-end
# ---------------------------------------------------------------------------


def test_action_transport_trainer_end_to_end_on_fake_data(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    request = TrainRequest(
        config_path=str(config_path),
        run_name="action_fake",
        runs_root=str(tmp_path / "runs"),
        seed=42,
        epochs=1,
        limit_train=1,
        limit_val=1,
    )
    trainer = ActionTransportTrainer(loader_builder=_fake_loader_builder)

    result = trainer.train(request)

    assert isinstance(result, TrainResult)
    run_dir = tmp_path / "runs" / "action_fake"
    checkpoint_path = run_dir / "action_head_last.pth"

    # Checkpoints are real manifest artifact refs describing the saved head.
    assert len(result.checkpoints) == 1
    ref = result.checkpoints[0]
    assert ref.semantic_kind == "action_head_checkpoint"
    assert ref.logical_path == "runs/action_fake/action_head_last.pth"
    assert ref.sha256 == hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    assert ref.size_bytes == checkpoint_path.stat().st_size

    # Metrics are the scalar detector measurements plus the final train loss.
    for key in ("ap50", "ap75", "precision", "recall", "num_predictions", "train_loss_final", "epochs"):
        assert key in result.metrics, f"missing metric {key}"
    assert math.isfinite(result.metrics["train_loss_final"])
    assert 0.0 <= result.metrics["ap50"] <= 1.0
    assert 0.0 <= result.metrics["ap75"] <= 1.0
    assert result.metrics["epochs"] == 1

    # The runtime manifest was explicitly completed with both outputs hashed.
    manifest = load_artifact_manifest(run_dir / RUNTIME_MANIFEST_NAME)
    assert manifest.completion == "completed"
    assert {output.semantic_kind for output in manifest.outputs} == {
        "action_head_checkpoint",
        "eval_metrics",
    }

    # The saved checkpoint reloads into a fresh head with identical weights.
    feature_dim = int(result.diagnostics["feature_dim"])
    reloaded = ActionLocalTransportHead(feature_dim=feature_dim)
    load_checkpoint(reloaded, checkpoint_path, torch.device("cpu"))
    saved_state = torch.load(checkpoint_path, map_location="cpu")["model"]
    for name, tensor in reloaded.state_dict().items():
        assert torch.equal(tensor, saved_state[name]), f"reloaded weight mismatch: {name}"

    # The reloaded head produces a finite supervised action loss on a
    # deterministic synthetic proposal batch (no detector needed here).
    batch = _synthetic_action_batch(feature_dim)
    actions = reloaded(batch.state.features)
    loss = supervised_action_transport_loss(batch, actions, config=SupervisedActionLossConfig())
    assert torch.isfinite(loss["loss_total"])


# ---------------------------------------------------------------------------
# Standard adapter end-to-end
# ---------------------------------------------------------------------------


def test_standard_trainer_end_to_end_on_fake_data(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    request = TrainRequest(
        config_path=str(config_path),
        run_name="standard_fake",
        runs_root=str(tmp_path / "runs"),
        seed=42,
        epochs=1,
        limit_train=1,
        limit_val=1,
    )
    trainer = StandardDetectionTrainer(loader_builder=_fake_loader_builder)

    result = trainer.train(request)

    assert isinstance(result, TrainResult)
    run_dir = tmp_path / "runs" / "standard_fake"
    checkpoint_path = run_dir / "checkpoint_last.pth"

    assert math.isfinite(result.metrics["train_loss"])
    assert result.metrics["epoch"] == 1
    assert len(result.checkpoints) == 1
    ref = result.checkpoints[0]
    assert ref.semantic_kind == "final_checkpoint"
    assert ref.sha256 == hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()

    manifest = load_artifact_manifest(run_dir / RUNTIME_MANIFEST_NAME)
    assert manifest.completion == "completed"
    assert {output.semantic_kind for output in manifest.outputs} == {
        "final_checkpoint",
        "train_metrics",
    }

    # The saved detector checkpoint reloads into a fresh detector exactly.
    fresh_model = build_detector(load_config(config_path))
    load_checkpoint(fresh_model, checkpoint_path, torch.device("cpu"))
    saved_state = torch.load(checkpoint_path, map_location="cpu")["model"]
    for name, tensor in fresh_model.state_dict().items():
        assert torch.equal(tensor, saved_state[name]), f"reloaded weight mismatch: {name}"


# ---------------------------------------------------------------------------
# Failure path: an exploding loader must mark the manifest failed
# ---------------------------------------------------------------------------


class _ExplodingLoader:
    def __iter__(self):
        raise RuntimeError("fake loader failure")


def test_failed_run_marks_the_manifest_failed(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    request = TrainRequest(
        config_path=str(config_path),
        run_name="failing_fake",
        runs_root=str(tmp_path / "runs"),
        seed=42,
        epochs=1,
    )
    trainer = StandardDetectionTrainer(
        loader_builder=lambda config, **kwargs: (_ExplodingLoader(), _ExplodingLoader())
    )

    with pytest.raises(RuntimeError, match="fake loader failure"):
        trainer.train(request)

    manifest = load_artifact_manifest(tmp_path / "runs" / "failing_fake" / RUNTIME_MANIFEST_NAME)
    assert manifest.completion == "failed"
    assert manifest.unavailable_reason is not None
