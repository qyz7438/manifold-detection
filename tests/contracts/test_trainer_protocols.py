"""Contract tests for the detection trainer protocols (plan Task 15).

These tests pin the shape of ``TrainRequest`` / ``TrainResult`` /
``DetectionTrainer`` and the rule that trainers return data only: scalar
metrics, checkpoint artifact refs, and plain diagnostics — never a
research-status decision (validated / promoted / ...), which belongs to the
reviewed manifest promotion path in ``spectral_detection_posttrain.experiments``.
"""

from __future__ import annotations

import ast
import dataclasses
import hashlib
import inspect
from pathlib import Path

import pytest
import torch

from spectral_detection_posttrain.experiments.artifacts import ArtifactRef as ManifestArtifactRef
from spectral_detection_posttrain.trainers.detection.action_transport import ActionTransportTrainer
from spectral_detection_posttrain.trainers.detection.protocols import (
    RESEARCH_STATUS_TERMS,
    ArtifactRef,
    DetectionTrainer,
    TrainRequest,
    TrainResult,
)
from spectral_detection_posttrain.trainers.detection.standard import StandardDetectionTrainer

REPO_ROOT = Path(__file__).resolve().parents[2]
ADAPTER_FILES = (
    REPO_ROOT / "spectral_detection_posttrain" / "trainers" / "detection" / "standard.py",
    REPO_ROOT / "spectral_detection_posttrain" / "trainers" / "detection" / "action_transport.py",
)


def _artifact_ref_for_file(path: Path, semantic_kind: str = "final_checkpoint") -> ArtifactRef:
    payload = path.read_bytes()
    return ArtifactRef(
        semantic_kind=semantic_kind,
        logical_path=f"runs/contract_smoke/{path.name}",
        sha256=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
    )


# ---------------------------------------------------------------------------
# ArtifactRef reuse
# ---------------------------------------------------------------------------


def test_artifact_ref_is_the_manifest_contract_type() -> None:
    """The protocol reuses the experiments manifest ArtifactRef, not a copy."""
    assert ArtifactRef is ManifestArtifactRef


def test_checkpoint_ref_describes_a_real_file(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint_last.pth"
    checkpoint.write_bytes(b"weights")
    ref = _artifact_ref_for_file(checkpoint)
    result = TrainResult(checkpoints=(ref,), metrics={"train_loss": 1.0})
    assert result.checkpoints[0].sha256 == hashlib.sha256(b"weights").hexdigest()
    assert result.checkpoints[0].size_bytes == len(b"weights")


# ---------------------------------------------------------------------------
# Dataclass shape
# ---------------------------------------------------------------------------


def test_request_and_result_are_frozen_dataclasses() -> None:
    for cls in (TrainRequest, TrainResult):
        assert dataclasses.is_dataclass(cls)
        assert cls.__dataclass_params__.frozen


def test_train_result_fields_are_exactly_the_plan_shape() -> None:
    assert {field.name for field in dataclasses.fields(TrainResult)} == {
        "checkpoints",
        "metrics",
        "diagnostics",
    }


def test_train_request_defaults() -> None:
    request = TrainRequest(config_path="config.yaml", run_name="smoke")
    assert request.runs_root == "runs"
    assert request.seed is None
    assert request.epochs is None
    assert request.limit_train is None
    assert request.limit_val is None


# ---------------------------------------------------------------------------
# TrainRequest validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides",
    [
        {"config_path": ""},
        {"run_name": ""},
        {"run_name": "  "},
        {"runs_root": ""},
        {"seed": 1.5},
        {"epochs": 0},
        {"epochs": -1},
        {"epochs": 1.5},
        {"limit_train": 0},
        {"limit_train": -3},
        {"limit_val": True},
    ],
)
def test_train_request_rejects_invalid_inputs(overrides: dict) -> None:
    kwargs = {"config_path": "config.yaml", "run_name": "smoke", **overrides}
    with pytest.raises(ValueError):
        TrainRequest(**kwargs)


def test_train_request_accepts_full_valid_input() -> None:
    request = TrainRequest(
        config_path="config.yaml",
        run_name="smoke",
        runs_root="runs",
        seed=42,
        epochs=2,
        limit_train=8,
        limit_val=4,
    )
    assert request.epochs == 2


# ---------------------------------------------------------------------------
# TrainResult metrics: scalar union only
# ---------------------------------------------------------------------------


def test_train_result_accepts_scalar_metric_union() -> None:
    result = TrainResult(
        checkpoints=(),
        metrics={
            "ap50": 0.5,
            "num_predictions": 7,
            "note": "smoke run",
            "converged": True,
            "missing": None,
        },
    )
    assert result.metrics["converged"] is True


@pytest.mark.parametrize(
    "bad_value",
    [
        torch.tensor(1.0),
        {"nested": 1.0},
        [1.0, 2.0],
        (1.0,),
    ],
)
def test_train_result_rejects_non_scalar_metrics(bad_value: object) -> None:
    with pytest.raises(ValueError, match="scalar"):
        TrainResult(checkpoints=(), metrics={"bad": bad_value})


def test_train_result_rejects_non_mapping_metrics() -> None:
    with pytest.raises(ValueError, match="metrics"):
        TrainResult(checkpoints=(), metrics=[("ap50", 0.5)])


def test_train_result_rejects_non_string_metric_keys() -> None:
    with pytest.raises(ValueError, match="keys"):
        TrainResult(checkpoints=(), metrics={1: 0.5})


# ---------------------------------------------------------------------------
# TrainResult checkpoints: ArtifactRef only
# ---------------------------------------------------------------------------


def test_train_result_coerces_checkpoint_list_to_tuple(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint_last.pth"
    checkpoint.write_bytes(b"weights")
    ref = _artifact_ref_for_file(checkpoint)
    result = TrainResult(checkpoints=[ref], metrics={})
    assert isinstance(result.checkpoints, tuple)
    assert result.checkpoints == (ref,)


@pytest.mark.parametrize("bad_ref", ["checkpoint_last.pth", 123, {"path": "x"}, None])
def test_train_result_rejects_non_artifact_ref_checkpoints(bad_ref: object) -> None:
    with pytest.raises(ValueError, match="ArtifactRef"):
        TrainResult(checkpoints=(bad_ref,), metrics={})


# ---------------------------------------------------------------------------
# Trainers never decide research status
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("term", sorted(RESEARCH_STATUS_TERMS))
def test_train_result_rejects_status_terms_in_metric_values(term: str) -> None:
    with pytest.raises(ValueError, match="research status"):
        TrainResult(checkpoints=(), metrics={"run_note": f"this run is {term}"})


@pytest.mark.parametrize("term", sorted(RESEARCH_STATUS_TERMS))
def test_train_result_rejects_status_terms_in_metric_keys(term: str) -> None:
    with pytest.raises(ValueError, match="research status"):
        TrainResult(checkpoints=(), metrics={term: 1.0})


@pytest.mark.parametrize("term", sorted(RESEARCH_STATUS_TERMS))
def test_train_result_rejects_status_terms_in_nested_diagnostics(term: str) -> None:
    diagnostics = {"history": [{"note": f"run {term} by trainer"}], "depth": 1}
    with pytest.raises(ValueError, match="research status"):
        TrainResult(checkpoints=(), metrics={}, diagnostics=diagnostics)


def test_train_result_allows_lifecycle_completion_words() -> None:
    """Lifecycle states are data, not research-status decisions."""
    result = TrainResult(
        checkpoints=(),
        metrics={"completion": "completed", "manifest_kind": "runtime"},
        diagnostics={"scope": "smoke", "state": "started"},
    )
    assert result.metrics["completion"] == "completed"


def test_research_status_terms_are_documented_and_non_empty() -> None:
    assert isinstance(RESEARCH_STATUS_TERMS, frozenset)
    assert RESEARCH_STATUS_TERMS
    assert all(isinstance(term, str) and term.islower() for term in RESEARCH_STATUS_TERMS)


def test_train_result_rejects_non_mapping_diagnostics() -> None:
    with pytest.raises(ValueError, match="diagnostics"):
        TrainResult(checkpoints=(), metrics={}, diagnostics=["not", "a", "mapping"])


def test_train_result_rejects_non_string_diagnostics_keys() -> None:
    with pytest.raises(ValueError, match="diagnostics"):
        TrainResult(checkpoints=(), metrics={}, diagnostics={1: "one"})


# ---------------------------------------------------------------------------
# DetectionTrainer protocol and adapter conformance
# ---------------------------------------------------------------------------


def test_detection_trainer_is_a_runtime_checkable_protocol() -> None:
    assert getattr(DetectionTrainer, "_is_protocol", False)
    assert getattr(DetectionTrainer, "_is_runtime_protocol", False)


def test_detection_trainer_train_signature() -> None:
    signature = inspect.signature(DetectionTrainer.train)
    assert list(signature.parameters) == ["self", "request"]
    assert signature.parameters["request"].annotation == "TrainRequest"
    assert signature.return_annotation == "TrainResult"


@pytest.mark.parametrize("trainer", [StandardDetectionTrainer(), ActionTransportTrainer()])
def test_adapters_satisfy_the_detection_trainer_protocol(trainer: object) -> None:
    assert isinstance(trainer, DetectionTrainer)
    signature = inspect.signature(trainer.train)
    assert list(signature.parameters) == ["request"]


# ---------------------------------------------------------------------------
# Adapters stay inside the process and inside the run directory
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("adapter_file", ADAPTER_FILES, ids=lambda path: path.name)
def test_adapters_do_not_shell_out(adapter_file: Path) -> None:
    """Adapters call maintained functions in-process; no subprocess/os.system."""
    tree = ast.parse(adapter_file.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported = {alias.name.split(".")[0] for alias in node.names}
            assert "subprocess" not in imported, f"{adapter_file.name} imports subprocess"
        elif isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".")[0] != "subprocess"
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            assert not (
                isinstance(node.func.value, ast.Name)
                and node.func.value.id == "os"
                and node.func.attr in {"system", "popen"}
            ), f"{adapter_file.name} calls os.{node.func.attr}"


@pytest.mark.parametrize("adapter_file", ADAPTER_FILES, ids=lambda path: path.name)
def test_adapters_write_only_under_the_run_directory(adapter_file: Path) -> None:
    """Adapters must not carry hardcoded repository write paths of their own."""
    tree = ast.parse(adapter_file.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            value = node.value
            assert not (
                value.startswith("runs/") or value.startswith("/") or "\\runs\\" in value
            ), f"hardcoded write path {value!r} in {adapter_file.name}"
