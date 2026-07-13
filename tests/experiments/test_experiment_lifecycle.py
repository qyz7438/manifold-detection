"""Tests for canonical-runner artifact lifecycle hooks (Task 6).

Covers the explicit runtime-manifest lifecycle:

- ``prepare_experiment`` writes a ``started`` runtime manifest.
- ``finalize_experiment`` moves it to ``completed`` with output hashes.
- ``fail_experiment`` moves it to ``failed`` with a sanitized failure type.
- A formal run on a dirty git tree is rejected before any run-directory
  mutation.
- A declared input without a recorded hash is rejected.
"""

from __future__ import annotations

import json
from dataclasses import replace

import pytest
import yaml

from spectral_detection_posttrain.experiments.artifacts import load_artifact_manifest
from spectral_detection_posttrain.experiments.canonical_runner import (
    fail_experiment,
    finalize_experiment,
    prepare_experiment,
)
from spectral_detection_posttrain.experiments.lifecycle import (
    FormalRunRejectedError,
    LifecycleError,
    MissingInputHashError,
    start_runtime_manifest,
)
from spectral_detection_posttrain.experiments.metadata import sha256_file

_FULL_VAL_SCOPE = {"kind": "full_val", "image_count": 100, "limit_train": None, "limit_val": None}


def _config(scope: dict | None = None) -> dict:
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
        },
        "train": {"batch_size": 1, "epochs": 1},
        "posttrain": {"batch_size": 1, "epochs": 1},
        "matching": {"iou_threshold": 0.5, "score_threshold": 0.05},
        "eval": {"batch_size": 1},
    }
    if scope is not None:
        config["evaluation_scope"] = scope
    return config


def _patch_git(monkeypatch: pytest.MonkeyPatch, *, dirty: bool) -> None:
    monkeypatch.setattr(
        "spectral_detection_posttrain.experiments.metadata.collect_git_metadata",
        lambda: {
            "git_commit": "57b198cfdc8586900785cc534345dfdfeb7681b2",
            "git_branch": "test-branch",
            "git_dirty": dirty,
            "git_status_hash": "0" * 64,
            "git_status_entries": 1 if dirty else 0,
            "git_status_short": " M file.py" if dirty else "",
            "git_status_truncated": False,
        },
    )


def _prepare(
    tmp_path,
    *,
    run_name: str = "lifecycle_smoke",
    config: dict | None = None,
    checkpoint: bool = True,
):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config or _config(), sort_keys=False), encoding="utf-8")
    checkpoint_path = None
    if checkpoint:
        checkpoint_path = tmp_path / "checkpoint.pth"
        checkpoint_path.write_bytes(b"checkpoint-bytes")
    return prepare_experiment(
        config_path=config_path,
        run_name=run_name,
        phase="eval",
        checkpoint_path=checkpoint_path,
        runs_root=tmp_path / "runs",
    )


def test_prepare_writes_started_runtime_manifest(tmp_path) -> None:
    context = _prepare(tmp_path)

    manifest_path = context.run_dir / "manifest.json"
    assert context.manifest_path == manifest_path
    manifest = load_artifact_manifest(manifest_path)

    assert manifest.manifest_kind == "runtime"
    assert manifest.completion == "started"
    assert manifest.run_id == "lifecycle_smoke"
    assert manifest.outputs == ()
    assert manifest.metrics_summary == {}
    assert manifest.gates == {}
    assert manifest.runtime_manifest_sha256 is None
    assert manifest.evaluation_scope.kind == "limited_unknown"

    config_ref = manifest.resolved_config
    assert config_ref.semantic_kind == "resolved_config"
    assert config_ref.logical_path == "runs/lifecycle_smoke/config.yaml"
    assert config_ref.sha256 == sha256_file(context.run_dir / "config.yaml")

    checkpoint_refs = [ref for ref in manifest.inputs if ref.semantic_kind == "initial_checkpoint"]
    assert len(checkpoint_refs) == 1
    assert checkpoint_refs[0].sha256 == context.metadata["checkpoint_hash"]


def test_prepare_formal_run_with_clean_tree_writes_started_manifest(tmp_path, monkeypatch) -> None:
    _patch_git(monkeypatch, dirty=False)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(_config(_FULL_VAL_SCOPE), sort_keys=False), encoding="utf-8")

    context = prepare_experiment(
        config_path=config_path,
        run_name="formal_clean",
        phase="eval",
        runs_root=tmp_path / "runs",
    )

    manifest = load_artifact_manifest(context.run_dir / "manifest.json")
    assert manifest.completion == "started"
    assert manifest.git_dirty is False
    assert manifest.git_commit == "57b198cfdc8586900785cc534345dfdfeb7681b2"
    assert manifest.evaluation_scope.kind == "full_val"
    assert manifest.evaluation_scope.image_count == 100


def test_formal_run_with_dirty_tree_rejected_before_run_dir_mutation(tmp_path, monkeypatch) -> None:
    _patch_git(monkeypatch, dirty=True)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(_config(_FULL_VAL_SCOPE), sort_keys=False), encoding="utf-8")
    runs_root = tmp_path / "runs"

    with pytest.raises(FormalRunRejectedError):
        prepare_experiment(
            config_path=config_path,
            run_name="dirty_formal",
            phase="eval",
            runs_root=runs_root,
        )

    assert not (runs_root / "dirty_formal").exists()


def test_missing_input_hash_rejected(tmp_path) -> None:
    context = _prepare(tmp_path)
    broken = replace(
        context,
        metadata={key: value for key, value in context.metadata.items() if key != "checkpoint_hash"},
    )

    with pytest.raises(MissingInputHashError):
        start_runtime_manifest(broken)


def test_finalize_marks_completed_with_output_hashes(tmp_path) -> None:
    context = _prepare(tmp_path)
    metrics_path = context.run_dir / "eval_metrics.json"
    metrics_path.write_text(json.dumps({"AP50": 0.5}), encoding="utf-8")

    manifest = finalize_experiment(
        context,
        outputs={"eval_metrics": metrics_path},
        metrics={"AP50": 0.5, "num_predictions": 7},
        gates={"clean_eval": True},
    )

    assert manifest.completion == "completed"
    assert len(manifest.outputs) == 1
    ref = manifest.outputs[0]
    assert ref.semantic_kind == "eval_metrics"
    assert ref.logical_path == "runs/lifecycle_smoke/eval_metrics.json"
    assert ref.sha256 == sha256_file(metrics_path)
    assert ref.size_bytes == metrics_path.stat().st_size
    assert manifest.metrics_summary == {"AP50": 0.5, "num_predictions": 7}
    assert manifest.gates == {"clean_eval": True}

    reloaded = load_artifact_manifest(context.run_dir / "manifest.json")
    assert reloaded.completion == "completed"
    assert reloaded.outputs[0].sha256 == ref.sha256


def test_finalize_twice_rejected(tmp_path) -> None:
    context = _prepare(tmp_path)
    finalize_experiment(context, outputs={}, metrics={}, gates={})

    with pytest.raises(LifecycleError):
        finalize_experiment(context, outputs={}, metrics={}, gates={})


def test_finalize_without_prepare_rejected(tmp_path) -> None:
    context = _prepare(tmp_path)
    (context.run_dir / "manifest.json").unlink()

    with pytest.raises(LifecycleError):
        finalize_experiment(context, outputs={}, metrics={}, gates={})


def test_fail_experiment_marks_failed_with_sanitized_failure_type(tmp_path) -> None:
    context = _prepare(tmp_path)
    error = RuntimeError("cuda boom password=hunter2 in C:\\Users\\bob\\model.pth")

    manifest = fail_experiment(context, error)

    assert manifest.completion == "failed"
    assert manifest.unavailable_reason is not None
    assert manifest.unavailable_reason.startswith("RuntimeError:")
    lowered = manifest.unavailable_reason.lower()
    assert "password" not in lowered
    assert "hunter2" not in lowered
    assert "c:\\users\\" not in lowered
    assert "<redacted>" in manifest.unavailable_reason

    reloaded = load_artifact_manifest(context.run_dir / "manifest.json")
    assert reloaded.completion == "failed"
    assert reloaded.unavailable_reason == manifest.unavailable_reason
