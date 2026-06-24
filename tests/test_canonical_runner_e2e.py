from __future__ import annotations

import json
import subprocess
import sys

import yaml

from spectral_detection_posttrain.experiments.metadata import sha256_file


def test_canonical_runner_cli_prepares_run_metadata(tmp_path) -> None:
    config = {
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
            "afm_type": "mplseg_mid",
        },
        "train": {"batch_size": 1, "epochs": 1},
        "posttrain": {"batch_size": 1, "epochs": 1},
        "matching": {"iou_threshold": 0.5, "score_threshold": 0.05},
        "eval": {"batch_size": 1},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    checkpoint_path = tmp_path / "checkpoint.pth"
    checkpoint_path.write_bytes(b"fake-checkpoint-for-metadata")
    runs_root = tmp_path / "runs"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "spectral_detection_posttrain.experiments.canonical_runner",
            "--config",
            str(config_path),
            "--run-name",
            "e2e_smoke",
            "--phase",
            "eval",
            "--checkpoint",
            str(checkpoint_path),
            "--runs-root",
            str(runs_root),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    run_dir = runs_root / "e2e_smoke"
    saved_config = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))

    assert str(run_dir) in completed.stdout
    assert saved_config["model"]["afm_channels"] == 256
    assert metadata["phase"] == "eval"
    assert metadata["run_name"] == "e2e_smoke"
    assert metadata["checkpoint_hash"] == sha256_file(checkpoint_path)
    assert metadata["torch_version"]
    assert metadata["torchvision_version"]
