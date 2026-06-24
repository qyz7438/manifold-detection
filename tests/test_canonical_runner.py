from __future__ import annotations

import json

import pytest
import yaml

from spectral_detection_posttrain.experiments.canonical_runner import (
    prepare_experiment,
    validate_checkpoint_path,
)


def _config() -> dict:
    return {
        "seed": 42,
        "device": "cpu",
        "data": {"root": "./data", "train_fraction": 0.8},
        "model": {
            "name": "fasterrcnn_mobilenet_v3_large_320_fpn",
            "pretrained": False,
            "allow_random_init_fallback": False,
            "num_classes": 2,
            "min_size": 320,
            "max_size": 320,
        },
        "train": {"batch_size": 1, "epochs": 1},
        "posttrain": {"batch_size": 1, "epochs": 1},
        "matching": {"iou_threshold": 0.5, "score_threshold": 0.05},
        "eval": {"batch_size": 1},
    }


def test_validate_checkpoint_path_rejects_missing_file(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        validate_checkpoint_path(tmp_path / "missing.pth")


def test_validate_checkpoint_path_rejects_empty_file(tmp_path) -> None:
    path = tmp_path / "empty.pth"
    path.write_bytes(b"")

    with pytest.raises(ValueError, match="empty"):
        validate_checkpoint_path(path)


def test_prepare_experiment_writes_normalized_config_and_metadata(tmp_path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(_config(), sort_keys=False), encoding="utf-8")
    checkpoint_path = tmp_path / "checkpoint.pth"
    checkpoint_path.write_bytes(b"checkpoint")

    context = prepare_experiment(
        config_path=config_path,
        run_name="canonical_smoke",
        phase="eval",
        checkpoint_path=checkpoint_path,
        runs_root=tmp_path / "runs",
    )

    saved_config = yaml.safe_load((context.run_dir / "config.yaml").read_text(encoding="utf-8"))
    metadata = json.loads((context.run_dir / "metadata.json").read_text(encoding="utf-8"))

    assert context.run_dir == tmp_path / "runs" / "canonical_smoke"
    assert saved_config["model"]["model_name"] == "fasterrcnn_mobilenet_v3_large_320_fpn"
    assert metadata["phase"] == "eval"
    assert metadata["checkpoint_hash"]
    assert metadata["config_hash"]
