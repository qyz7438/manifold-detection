"""Analyze the locked three-seed U/W/F/S weighted box-head experiment group."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping

import torch


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from spectral_detection_posttrain.eval.paired_bootstrap import (  # noqa: E402
    hierarchical_paired_bootstrap_ap75,
    paired_bootstrap_ap75,
)
from spectral_detection_posttrain.methods.energy_transport.endpoint.awr_weighting import (  # noqa: E402
    BudgetCounters,
    assert_equal_budget,
)
from spectral_detection_posttrain.utils.io import save_json  # noqa: E402


LOCKED_SEEDS = (42, 2024, 999)
LOCKED_ARMS = ("U", "W", "F", "S")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluate_group_gates(
    metrics_by_seed: Mapping[int, Mapping[str, Mapping[str, Any]]],
    *,
    wu_lcb: float,
    ws_lcb: float,
    provenance_passed: bool = True,
    support_passed: bool = True,
    weight_health_passed: bool = True,
) -> dict[str, Any]:
    """Apply locked budget and scientific gates to a complete metric group."""
    if set(metrics_by_seed) != set(LOCKED_SEEDS):
        raise ValueError(f"metrics must contain exactly seeds {LOCKED_SEEDS}")
    for seed in LOCKED_SEEDS:
        if set(metrics_by_seed[seed]) != set(LOCKED_ARMS):
            raise ValueError(f"seed {seed} must contain exactly arms {LOCKED_ARMS}")

    budget_match = True
    for seed in LOCKED_SEEDS:
        counters = {}
        for arm in LOCKED_ARMS:
            payload = metrics_by_seed[seed][arm].get("budget_counters", {})
            try:
                counters[arm] = BudgetCounters(**payload)
            except (TypeError, ValueError):
                budget_match = False
                break
        if not budget_match:
            break
        try:
            assert_equal_budget(counters)
        except AssertionError:
            budget_match = False
            break

    wu_deltas = [
        float(metrics_by_seed[seed]["W"]["ap75"])
        - float(metrics_by_seed[seed]["U"]["ap75"])
        for seed in LOCKED_SEEDS
    ]
    ws_deltas = [
        float(metrics_by_seed[seed]["W"]["ap75"])
        - float(metrics_by_seed[seed]["S"]["ap75"])
        for seed in LOCKED_SEEDS
    ]
    wu_mean = math.fsum(wu_deltas) / len(wu_deltas)
    ws_mean = math.fsum(ws_deltas) / len(ws_deltas)

    safety = True
    calibration = True
    generalization = True
    for seed in LOCKED_SEEDS:
        uniform = metrics_by_seed[seed]["U"]
        weighted = metrics_by_seed[seed]["W"]
        if float(weighted["false_positive_rate"]) - float(uniform["false_positive_rate"]) > 0.01:
            safety = False
        uniform_predictions = int(uniform["num_predictions"])
        weighted_predictions = int(weighted["num_predictions"])
        prediction_increase = (
            0.0
            if uniform_predictions == 0 and weighted_predictions == 0
            else float("inf")
            if uniform_predictions == 0
            else (weighted_predictions - uniform_predictions) / uniform_predictions
        )
        if prediction_increase > 0.10:
            safety = False
        if weighted.get("ece") is None or uniform.get("ece") is None:
            calibration = False
        elif float(weighted["ece"]) - float(uniform["ece"]) > 0.01:
            calibration = False

        weighted_gap = float(weighted["fit_metrics"]["ap75"]) - float(weighted["ap75"])
        uniform_gap = float(uniform["fit_metrics"]["ap75"]) - float(uniform["ap75"])
        if weighted_gap - uniform_gap > 0.05:
            generalization = False

    gates = {
        "provenance": bool(provenance_passed),
        "support": bool(support_passed),
        "weight_health": bool(weight_health_passed),
        "budget_match": budget_match,
        "advantage_causality": ws_lcb > 0.0 and sum(delta > 0.0 for delta in ws_deltas) >= 2,
        "uniform_gain": wu_mean >= 0.002 and wu_lcb > 0.0 and min(wu_deltas) >= -0.002,
        "safety": safety,
        "calibration": calibration,
        "generalization": generalization,
    }
    return {
        "gates": gates,
        "all_passed": all(gates.values()),
        "deltas": {
            "W_minus_U_per_seed": dict(zip(LOCKED_SEEDS, wu_deltas, strict=True)),
            "W_minus_U_mean": wu_mean,
            "W_minus_S_per_seed": dict(zip(LOCKED_SEEDS, ws_deltas, strict=True)),
            "W_minus_S_mean": ws_mean,
        },
    }


def _load_group(run_dirs: list[Path]) -> tuple[dict[int, dict[str, dict]], dict]:
    metrics_by_seed: dict[int, dict[str, dict]] = {seed: {} for seed in LOCKED_SEEDS}
    artifacts: dict[int, dict[str, dict]] = {seed: {} for seed in LOCKED_SEEDS}
    provenance_reference: dict[str, Any] | None = None
    support_passed = True
    weight_health_passed = True
    provenance_keys = (
        "version_id",
        "git_commit",
        "config_sha256",
        "split_manifest_sha256",
        "checkpoint_sha256",
        "annotation_sha256",
        "cache_sha256",
    )
    for run_dir in run_dirs:
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        metrics = json.loads((run_dir / "eval_metrics.json").read_text(encoding="utf-8"))
        seed = int(manifest["seed"])
        arm = str(manifest["arm"])
        if seed not in metrics_by_seed or arm not in LOCKED_ARMS:
            raise ValueError(f"unexpected run identity seed={seed}, arm={arm}")
        if arm in metrics_by_seed[seed]:
            raise ValueError(f"duplicate run identity seed={seed}, arm={arm}")
        if metrics.get("seed") != seed or metrics.get("arm") != arm:
            raise ValueError("manifest and metric run identities disagree")
        current_provenance = {key: manifest.get(key) for key in provenance_keys}
        support = manifest.get("utility_support", {})
        diagnostics = manifest.get("weight_diagnostics", {})
        if (
            float(support.get("support_fraction", -1.0)) < 0.15
            or int(support.get("positive_candidate_count", -1)) < 500
        ):
            support_passed = False
        if (
            not math.isclose(float(diagnostics.get("normalized_mean", float("nan"))), 1.0, abs_tol=1e-7)
            or float(diagnostics.get("saturation_fraction", float("inf"))) > 0.10
            or float(diagnostics.get("effective_sample_size", -1.0)) < 125.0
        ):
            weight_health_passed = False
        if provenance_reference is None:
            provenance_reference = current_provenance
        elif current_provenance != provenance_reference:
            raise ValueError("run provenance hashes do not match across the group")
        prediction_path = run_dir / "tune_predictions.pt"
        expected_prediction_sha = metrics.get("prediction_artifact_sha256")
        if (
            not isinstance(expected_prediction_sha, str)
            or _sha256_file(prediction_path) != expected_prediction_sha
        ):
            raise ValueError("prediction artifact SHA256 mismatch")
        prediction_payload = torch.load(prediction_path, map_location="cpu")
        metrics_by_seed[seed][arm] = metrics
        artifacts[seed][arm] = prediction_payload

    for seed in LOCKED_SEEDS:
        if set(metrics_by_seed[seed]) != set(LOCKED_ARMS):
            raise ValueError(f"incomplete run group for seed {seed}")
    provenance_passed = provenance_reference is not None and all(
        isinstance(value, str) and bool(value) for value in provenance_reference.values()
    )
    return metrics_by_seed, {
        "runs": artifacts,
        "provenance": provenance_reference,
        "preflight_gates": {
            "provenance": provenance_passed,
            "support": support_passed,
            "weight_health": weight_health_passed,
        },
    }


def analyze(run_dirs: list[Path], *, n_resamples: int = 10_000) -> dict[str, Any]:
    if len(run_dirs) != len(LOCKED_SEEDS) * len(LOCKED_ARMS):
        raise ValueError("group analysis requires exactly 12 U/W/F/S run directories")
    metrics, loaded = _load_group(run_dirs)
    artifacts = loaded["runs"]

    per_seed: dict[int, dict[str, Any]] = {}
    for seed in LOCKED_SEEDS:
        wu = paired_bootstrap_ap75(
            artifacts[seed]["W"]["predictions"],
            artifacts[seed]["U"]["predictions"],
            artifacts[seed]["W"]["targets"],
            n_resamples=n_resamples,
            seed=42,
        )
        ws = paired_bootstrap_ap75(
            artifacts[seed]["W"]["predictions"],
            artifacts[seed]["S"]["predictions"],
            artifacts[seed]["W"]["targets"],
            n_resamples=n_resamples,
            seed=42,
        )
        per_seed[seed] = {"W_minus_U": asdict(wu), "W_minus_S": asdict(ws)}

    predictions_w = {seed: artifacts[seed]["W"]["predictions"] for seed in LOCKED_SEEDS}
    targets_w = {seed: artifacts[seed]["W"]["targets"] for seed in LOCKED_SEEDS}
    hierarchical_wu = hierarchical_paired_bootstrap_ap75(
        predictions_w,
        {seed: artifacts[seed]["U"]["predictions"] for seed in LOCKED_SEEDS},
        targets_w,
        n_resamples=n_resamples,
        seed=42,
    )
    hierarchical_ws = hierarchical_paired_bootstrap_ap75(
        predictions_w,
        {seed: artifacts[seed]["S"]["predictions"] for seed in LOCKED_SEEDS},
        targets_w,
        n_resamples=n_resamples,
        seed=42,
    )
    gate_result = evaluate_group_gates(
        metrics,
        wu_lcb=hierarchical_wu.ci_low,
        ws_lcb=hierarchical_ws.ci_low,
        provenance_passed=loaded["preflight_gates"]["provenance"],
        support_passed=loaded["preflight_gates"]["support"],
        weight_health_passed=loaded["preflight_gates"]["weight_health"],
    )
    return {
        "version_id": "det.energy.oracle_utility_boxhead.group.001",
        "provenance": loaded["provenance"],
        "bootstrap_resamples": n_resamples,
        "per_seed_bootstrap": per_seed,
        "hierarchical_bootstrap": {
            "W_minus_U": asdict(hierarchical_wu),
            "W_minus_S": asdict(hierarchical_ws),
        },
        **gate_result,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resamples", type=int, default=10_000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.resamples != 10_000:
        raise ValueError("formal group analysis requires exactly 10,000 resamples")
    result = analyze(args.run_dir, n_resamples=args.resamples)
    save_json(result, args.output)
    print(json.dumps({"all_passed": result["all_passed"], "gates": result["gates"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
