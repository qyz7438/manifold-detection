from __future__ import annotations

import hashlib

from spectral_detection_posttrain.experiments.metadata import (
    collect_experiment_metadata,
    hash_config,
    sha256_file,
)


def test_sha256_file_returns_known_digest(tmp_path) -> None:
    path = tmp_path / "payload.txt"
    path.write_text("canonical-runner", encoding="utf-8")

    assert sha256_file(path) == hashlib.sha256(b"canonical-runner").hexdigest()


def test_hash_config_is_order_stable() -> None:
    left = {"b": 2, "a": {"z": 1}}
    right = {"a": {"z": 1}, "b": 2}

    assert hash_config(left) == hash_config(right)


def test_collect_metadata_includes_versions_hashes_and_git_keys(tmp_path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("seed: 42\n", encoding="utf-8")
    checkpoint_path = tmp_path / "checkpoint.pth"
    checkpoint_path.write_bytes(b"checkpoint")
    config = {"seed": 42, "model": {"name": "fasterrcnn_mobilenet_v3_large_320_fpn"}}

    metadata = collect_experiment_metadata(config, config_path=config_path, checkpoint_path=checkpoint_path)

    assert metadata["config_hash"] == hash_config(config)
    assert metadata["config_file_hash"] == sha256_file(config_path)
    assert metadata["checkpoint_hash"] == sha256_file(checkpoint_path)
    assert metadata["torch_version"]
    assert metadata["torchvision_version"]
    assert "git_commit" in metadata
    assert "git_dirty" in metadata
    assert "git_status_hash" in metadata
    assert "git_status_entries" in metadata
    assert "git_status_truncated" in metadata
    assert len(metadata["git_status_short"].splitlines()) <= 50
