from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from scripts.experiments.awr_weighted_boxhead.analyze_group import (
    _load_group,
    evaluate_group_gates,
)


def _metrics(ap75: float, *, fit_ap75: float | None = None) -> dict:
    return {
        "ap75": ap75,
        "false_positive_rate": 0.20,
        "num_predictions": 100,
        "ece": 0.10,
        "fit_metrics": {"ap75": ap75 + 0.02 if fit_ap75 is None else fit_ap75},
        "budget_counters": {
            "image_exposures": 2500,
            "logical_batches": 320,
            "optimizer_steps": 320,
            "scheduler_steps": 0,
        },
    }


def _passing_group() -> dict[int, dict[str, dict]]:
    group = {}
    for seed in (42, 2024, 999):
        group[seed] = {
            "U": _metrics(0.40),
            "W": _metrics(0.41),
            "F": _metrics(0.405),
            "S": _metrics(0.402),
        }
    return group


def test_group_gates_pass_locked_three_seed_case() -> None:
    result = evaluate_group_gates(_passing_group(), wu_lcb=0.004, ws_lcb=0.003)
    assert result["all_passed"] is True
    assert all(result["gates"].values())
    assert result["deltas"]["W_minus_U_mean"] == pytest.approx(0.01)


def test_group_gate_rejects_training_seed_or_bootstrap_failure() -> None:
    group = _passing_group()
    group[999]["W"]["ap75"] = 0.397
    result = evaluate_group_gates(group, wu_lcb=-0.001, ws_lcb=0.003)
    assert result["gates"]["uniform_gain"] is False
    assert result["all_passed"] is False


def test_group_gate_rejects_budget_safety_calibration_and_gap() -> None:
    group = _passing_group()
    group[42]["F"]["budget_counters"]["image_exposures"] = 2499
    group[2024]["W"]["false_positive_rate"] = 0.211
    group[999]["W"]["ece"] = 0.111
    group[999]["W"]["fit_metrics"]["ap75"] = 0.50
    result = evaluate_group_gates(group, wu_lcb=0.004, ws_lcb=0.003)
    assert result["gates"]["budget_match"] is False
    assert result["gates"]["safety"] is False
    assert result["gates"]["calibration"] is False
    assert result["gates"]["generalization"] is False


def test_group_gate_does_not_mutate_input() -> None:
    group = _passing_group()
    before = deepcopy(group)
    evaluate_group_gates(group, wu_lcb=0.004, ws_lcb=0.003)
    assert group == before


def test_group_loader_rejects_prediction_artifact_hash_mismatch(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    manifest = {
        "seed": 42,
        "arm": "U",
        "version_id": "det.energy.oracle_utility_boxhead.001",
        "git_commit": "abc123",
        "config_sha256": "config-sha",
        "split_manifest_sha256": "split-sha",
        "checkpoint_sha256": "checkpoint-sha",
        "annotation_sha256": "annotation-sha",
        "cache_sha256": "cache-sha",
        "utility_support": {"support_fraction": 0.2, "positive_candidate_count": 600},
        "weight_diagnostics": {
            "normalized_mean": 1.0,
            "saturation_fraction": 0.0,
            "effective_sample_size": 200.0,
        },
    }
    metrics = {
        **_metrics(0.4),
        "seed": 42,
        "arm": "U",
        "prediction_artifact_sha256": "tampered-sha",
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (run_dir / "eval_metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    torch.save({"predictions": {}, "targets": {}}, run_dir / "tune_predictions.pt")

    with pytest.raises(ValueError, match="prediction artifact SHA256"):
        _load_group([run_dir])
