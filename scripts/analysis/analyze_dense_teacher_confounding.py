"""Audit count confounding in the locked dense teacher component artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CONFIG_PATH = (
    ROOT
    / "spectral_detection_posttrain"
    / "configs"
    / "versions"
    / "det.energy.dense_teacher_confound.001.json"
)
CONFIG_SHA256 = "836293cc20379b2d4d0f7a1987e1b63e7cb1c680e97e19a16cbad5f178bea388"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config() -> dict[str, Any]:
    if sha256_file(CONFIG_PATH) != CONFIG_SHA256:
        raise ValueError("canonical dense teacher confounding config SHA256 mismatch")
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if config.get("version_id") != "det.energy.dense_teacher_confound.001":
        raise ValueError("dense teacher confounding version mismatch")
    if config.get("source", {}).get("forbidden_new_data") is not True:
        raise ValueError("confounding audit must forbid new data")
    return config


def _correlation(values: torch.Tensor) -> list[list[float | None]]:
    x = torch.as_tensor(values, dtype=torch.float64)
    if x.ndim != 2 or x.shape[0] < 2:
        raise ValueError("correlation values must have shape (N, D), N >= 2")
    centered = x - x.mean(dim=0, keepdim=True)
    scales = centered.square().sum(dim=0).sqrt()
    output: list[list[float | None]] = []
    for left in range(x.shape[1]):
        row: list[float | None] = []
        for right in range(x.shape[1]):
            denominator = float((scales[left] * scales[right]).item())
            row.append(
                None
                if denominator == 0.0
                else float((centered[:, left] * centered[:, right]).sum().item() / denominator)
            )
        output.append(row)
    return output


def residualized_correlation(
    values: torch.Tensor, controls: torch.Tensor
) -> list[list[float | None]]:
    """Return correlations after closed-form linear residualization."""
    y = torch.as_tensor(values, dtype=torch.float64)
    c = torch.as_tensor(controls, dtype=torch.float64)
    if y.ndim != 2 or c.ndim != 2 or y.shape[0] != c.shape[0]:
        raise ValueError("values and controls must be aligned 2D matrices")
    design = torch.cat((torch.ones((c.shape[0], 1), dtype=c.dtype), c), dim=1)
    fitted = design @ (torch.linalg.pinv(design) @ y)
    return _correlation(y - fitted)


def support_normalized_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, float]]:
    output = []
    for row in rows:
        predictions = max(int(row["prediction_count"]), 1)
        ground_truth = max(int(row["ground_truth_count"]), 1)
        duplicate_edges = max(int(row["duplicate_edge_count"]), 1)
        output.append(
            {
                "coverage_per_gt": float(row["coverage"]) / ground_truth,
                "background_risk_per_prediction": float(row["background_risk"]) / predictions,
                "class_risk_per_prediction": float(row["class_risk"]) / predictions,
                "calibration_error_per_prediction": float(row["calibration_error"]) / predictions,
                "duplicate_risk_per_edge": float(row["duplicate_risk"]) / duplicate_edges,
            }
        )
    return output


def _named_correlation(rows: Sequence[dict[str, Any]], fields: Sequence[str]) -> dict[str, dict[str, float | None]]:
    matrix = _correlation(
        torch.tensor([[float(row[field]) for field in fields] for row in rows], dtype=torch.float64)
    )
    return {
        left: {right: matrix[i][j] for j, right in enumerate(fields)}
        for i, left in enumerate(fields)
    }


def _named_residual_correlation(
    rows: Sequence[dict[str, Any]], fields: Sequence[str]
) -> dict[str, dict[str, float | None]]:
    values = torch.tensor([[float(row[field]) for field in fields] for row in rows], dtype=torch.float64)
    controls = torch.tensor(
        [
            [math.log1p(int(row["prediction_count"])), math.log1p(int(row["ground_truth_count"]))]
            for row in rows
        ],
        dtype=torch.float64,
    )
    matrix = residualized_correlation(values, controls)
    return {
        left: {right: matrix[i][j] for j, right in enumerate(fields)}
        for i, left in enumerate(fields)
    }


def _max_abs_off_diagonal(matrix: dict[str, dict[str, float | None]]) -> float:
    values = [
        abs(float(value))
        for left, row in matrix.items()
        for right, value in row.items()
        if left != right and value is not None
    ]
    return max(values) if values else float("inf")


def _count_only_r2(rows: Sequence[dict[str, Any]], fields: Sequence[str]) -> dict[str, float | None]:
    controls = torch.tensor(
        [
            [1.0, math.log1p(int(row["prediction_count"])), math.log1p(int(row["ground_truth_count"]))]
            for row in rows
        ],
        dtype=torch.float64,
    )
    output: dict[str, float | None] = {}
    for field in fields:
        target = torch.tensor([float(row[field]) for row in rows], dtype=torch.float64).unsqueeze(1)
        prediction = controls @ (torch.linalg.pinv(controls) @ target)
        total = (target - target.mean()).square().sum()
        output[field] = (
            None
            if float(total.item()) == 0.0
            else float((1.0 - (target - prediction).square().sum() / total).item())
        )
    return output


def analyze(rows: Sequence[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    from spectral_detection_posttrain.methods.energy_transport.dense_set_energy import robust_scalar_summary

    penalty_fields = list(config["analysis"]["penalty_components"])
    normalized = support_normalized_rows(rows)
    normalized_fields = list(normalized[0])
    normalized_penalties = [
        "background_risk_per_prediction",
        "class_risk_per_prediction",
        "calibration_error_per_prediction",
    ]
    raw_correlation = _named_correlation(rows, penalty_fields)
    residual_correlation = _named_residual_correlation(rows, penalty_fields)
    normalized_correlation = _named_correlation(normalized, normalized_penalties)
    normalized_summary = {
        field: robust_scalar_summary(
            [float(row[field]) for row in normalized],
            min_iqr=float(config["analysis"]["min_iqr"]),
        )
        for field in normalized_fields
    }
    required_normalized = [
        "coverage_per_gt",
        "background_risk_per_prediction",
        "class_risk_per_prediction",
        "calibration_error_per_prediction",
    ]
    threshold = float(config["analysis"]["max_abs_penalty_correlation"])
    residual_max = _max_abs_off_diagonal(residual_correlation)
    normalized_max = _max_abs_off_diagonal(normalized_correlation)
    gates = {
        "exact_image_count": len(rows) == int(config["gates"]["exact_image_count"]),
        "normalized_required_components_non_degenerate": all(
            not bool(normalized_summary[field]["degenerate"]) for field in required_normalized
        ),
        "residualized_penalty_correlation": residual_max <= threshold,
        "normalized_penalty_correlation": normalized_max <= threshold,
    }
    all_passed = all(gates.values())
    return {
        "image_count": len(rows),
        "raw_penalty_correlation": raw_correlation,
        "residualized_penalty_correlation": residual_correlation,
        "residualized_penalty_max_abs_off_diagonal": residual_max,
        "support_normalized_penalty_correlation": normalized_correlation,
        "support_normalized_penalty_max_abs_off_diagonal": normalized_max,
        "count_only_r2": _count_only_r2(rows, penalty_fields),
        "normalized_components": normalized_summary,
        "gates": gates,
        "all_gates_passed": all_passed,
        "decision": (
            "minimal_fixed_endpoint_probe_allowed"
            if all_passed
            else "reduce_penalty_basis_before_endpoint"
        ),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "runs" / "nwpu_dense_teacher_confound_s42_innerfit" / "eval_metrics.json",
    )
    parser.add_argument("--require-clean-git", action="store_true")
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config()
    source = (args.source or ROOT / config["source"]["result"]).resolve()
    if sha256_file(source) != config["source"]["result_sha256"]:
        raise ValueError("dense teacher source artifact SHA256 mismatch")
    payload = json.loads(source.read_text(encoding="utf-8"))
    expected = config["source"]
    if (
        payload.get("completed") is not True
        or payload.get("version_id") != expected["version_id"]
        or payload.get("git_commit") != expected["git_commit"]
        or payload.get("image_ids_sha256") != expected["image_ids_sha256"]
        or payload.get("forbidden_splits_read") != []
    ):
        raise ValueError("dense teacher source artifact contract mismatch")
    rows = payload.get("rows")
    if not isinstance(rows, list) or len(rows) != int(expected["image_count"]):
        raise ValueError("dense teacher source rows mismatch")
    git_dirty = bool(
        subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
    )
    if args.require_clean_git and git_dirty:
        raise RuntimeError("--require-clean-git requested but repository is dirty")
    summary = analyze(rows, config)
    result = {
        "completed": True,
        "scientific_status": summary["decision"],
        "version_id": config["version_id"],
        "experiment_scope": "locked_inner_fit_artifact_only_no_new_data",
        "config_sha256": CONFIG_SHA256,
        "source_result_sha256": config["source"]["result_sha256"],
        "source_git_commit": config["source"]["git_commit"],
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_dirty": git_dirty,
        "summary": summary,
        "new_data_read": False,
        "decision_rule": config["decision_rule"],
        "claim_boundary": config["claim_boundary"],
        "validated_claim": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    result = run(_parse_args())
    print(json.dumps({"status": result["scientific_status"], "summary": result["summary"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
