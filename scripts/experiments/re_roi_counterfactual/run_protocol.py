"""Run the locked train-only re-ROI counterfactual evidence protocol."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

import torch


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from scripts.experiments.re_roi_counterfactual.build_cache import (  # noqa: E402
    load_split_manifest,
    sha256_text_lf,
)
from scripts.experiments.re_roi_counterfactual.gates import (  # noqa: E402
    GateResult,
    check_support,
    run_protocol_evaluation,
)
from spectral_detection_posttrain.trainers.detection.awr_boxhead import (  # noqa: E402
    validate_re_roi_cache,
)
from spectral_detection_posttrain.utils.seed import resolve_device, set_seed  # noqa: E402


VERSION_ID = "det.energy.re_roi_counterfactual_evidence.001"
LOCKED_SEED = 42
LOCKED_SPLIT_SHA256 = "5c4222e033f7ed333835eab7f7786498b84bba11856c2d362ec8895845942244"
LOCKED_CONFIG = {
    "hidden_dim": 64,
    "epochs": 20,
    "lr": 1e-3,
    "lambda_rank": 0.1,
    "batch_size": 32,
    "bootstrap_resamples": 10_000,
    "bootstrap_seed": 42,
}
EXPECTED_SPLIT_COUNTS = {
    "fit": 250,
    "tune": 68,
    "calibration": 68,
    "outer_heldout": 68,
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _assert_clean_git() -> str:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    if dirty:
        raise RuntimeError("locked re-ROI runs require a clean Git worktree")
    return commit


def _resolve_device(requested: str) -> torch.device:
    device = resolve_device({"device": requested})
    if device.type == "cuda" and os.environ.get("CUDA_VISIBLE_DEVICES") != "2":
        raise RuntimeError("re-ROI GPU runs require CUDA_VISIBLE_DEVICES=2")
    return device


def _locked_split_ids(split: Mapping[str, Any]) -> dict[str, list[int]]:
    result: dict[str, list[int]] = {}
    for name, expected_count in EXPECTED_SPLIT_COUNTS.items():
        try:
            values = [int(value) for value in split["splits"][name]["image_ids"]]
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"split manifest is missing valid {name} image IDs") from error
        if len(values) != expected_count or len(values) != len(set(values)):
            raise ValueError(f"split {name} must contain {expected_count} unique image IDs")
        result[name] = values
    names = tuple(result)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            if set(result[left]) & set(result[right]):
                raise ValueError("re-ROI split image IDs must be pairwise disjoint")
    return result


def load_validated_cache(
    path: Path,
    *,
    split_name: str,
    image_ids: Sequence[int],
    split_manifest_sha256: str,
    checkpoint_sha256: str,
    annotation_sha256: str,
) -> tuple[list[Mapping[str, Any]], dict[str, Any], str]:
    payload = torch.load(path, map_location="cpu")
    records = validate_re_roi_cache(
        payload,
        fit_image_ids=image_ids,
        split_manifest_sha256=split_manifest_sha256,
        checkpoint_sha256=checkpoint_sha256,
        annotation_sha256=annotation_sha256,
        split_name=split_name,
    )
    metadata = dict(payload["metadata"])
    return records, metadata, sha256_file(path)


def _gate_dict(result: GateResult) -> dict[str, Any]:
    return asdict(result)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def _reproduction_command(args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--fit-cache",
        str(args.fit_cache.resolve()),
        "--tune-cache",
        str(args.tune_cache.resolve()),
        "--calibration-cache",
        str(args.calibration_cache.resolve()),
        "--split-manifest",
        str(args.split_manifest.resolve()),
        "--checkpoint",
        str(args.checkpoint.resolve()),
        "--annotation",
        str(args.annotation.resolve()),
        "--output-dir",
        str(args.output_dir.resolve()),
        "--device",
        str(args.device),
        "--seed",
        str(args.seed),
    ]


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.seed != LOCKED_SEED:
        raise ValueError(f"seed must be {LOCKED_SEED}")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to reuse run directory: {args.output_dir}")

    runner_commit = _assert_clean_git()
    device = _resolve_device(args.device)
    set_seed(args.seed)
    split = load_split_manifest(args.split_manifest)
    split_ids = _locked_split_ids(split)
    split_sha = sha256_text_lf(args.split_manifest)
    if split_sha != LOCKED_SPLIT_SHA256:
        raise ValueError("split manifest does not match the locked split SHA256")
    checkpoint_sha = sha256_file(args.checkpoint)
    annotation_sha = sha256_file(args.annotation)

    cache_args = {
        "fit": args.fit_cache,
        "tune": args.tune_cache,
        "calibration": args.calibration_cache,
    }
    records: dict[str, list[Mapping[str, Any]]] = {}
    metadata: dict[str, dict[str, Any]] = {}
    cache_sha: dict[str, str] = {}
    for name, path in cache_args.items():
        loaded_records, loaded_metadata, artifact_sha = load_validated_cache(
            path,
            split_name=name,
            image_ids=split_ids[name],
            split_manifest_sha256=split_sha,
            checkpoint_sha256=checkpoint_sha,
            annotation_sha256=annotation_sha,
        )
        records[name] = loaded_records
        metadata[name] = loaded_metadata
        cache_sha[name] = artifact_sha

    source_commits = {value.get("git_commit") for value in metadata.values()}
    model_configs = {json.dumps(value.get("model_config"), sort_keys=True) for value in metadata.values()}
    cache_configs = {json.dumps(value.get("cache_config"), sort_keys=True) for value in metadata.values()}
    if len(source_commits) != 1 or None in source_commits:
        raise ValueError("fit/tune/calibration caches must share one source Git commit")
    if len(model_configs) != 1 or len(cache_configs) != 1:
        raise ValueError("fit/tune/calibration caches must share model and cache configs")

    support = check_support(
        records["tune"], records["calibration"], fit_records=records["fit"]
    )
    manifest = {
        "version_id": VERSION_ID,
        "runner_git_commit": runner_commit,
        "cache_source_git_commit": next(iter(source_commits)),
        "split_manifest_sha256": split_sha,
        "checkpoint_sha256": checkpoint_sha,
        "annotation_sha256": annotation_sha,
        "cache_sha256": cache_sha,
        "cache_paths": {
            name: str(path.resolve()) for name, path in cache_args.items()
        },
        "split_manifest_path": str(args.split_manifest.resolve()),
        "checkpoint_path": str(args.checkpoint.resolve()),
        "annotation_path": str(args.annotation.resolve()),
        "config": LOCKED_CONFIG,
        "config_sha256": sha256_json(LOCKED_CONFIG),
        "device": str(device),
        "seed": args.seed,
        "evaluation_scope": "nwpu_train_only_fit_tune_calibration",
        "outer_heldout_read": False,
        "detector_validation_read": False,
        "reproduction_command": _reproduction_command(args),
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    _write_json(args.output_dir / "manifest.json", manifest)

    if support.passed:
        evaluation = run_protocol_evaluation(
            records["fit"],
            records["tune"],
            records["calibration"],
            hidden_dim=LOCKED_CONFIG["hidden_dim"],
            epochs=LOCKED_CONFIG["epochs"],
            lr=LOCKED_CONFIG["lr"],
            lambda_rank=LOCKED_CONFIG["lambda_rank"],
            batch_size=LOCKED_CONFIG["batch_size"],
            device=device,
            seed=args.seed,
        )
        gate_results = evaluation.gates
        model_artifacts: dict[str, dict[str, str]] = {}
        for arm, model in evaluation.models.items():
            path = args.output_dir / f"arm_{arm}_final.pt"
            torch.save(
                {
                    "arm": arm,
                    "state_dict": model.state_dict(),
                    "protocol_state": evaluation.protocol_state,
                    "version_id": VERSION_ID,
                },
                path,
            )
            model_artifacts[arm] = {"path": str(path), "sha256": sha256_file(path)}
        training_started = True
    else:
        gate_results = {"support": support}
        evaluation = None
        model_artifacts = {}
        training_started = False

    manifest["protocol_state"] = (
        evaluation.protocol_state if evaluation is not None else {}
    )
    manifest["model_artifacts"] = model_artifacts
    _write_json(args.output_dir / "manifest.json", manifest)

    result = {
        "version_id": VERSION_ID,
        "evaluation_scope": "nwpu_train_only_fit_tune_calibration",
        "training_started": training_started,
        "all_gates_passed": all(gate.passed for gate in gate_results.values()),
        "gates": {name: _gate_dict(gate) for name, gate in gate_results.items()},
        "metrics": evaluation.metrics if evaluation is not None else {},
        "controls": evaluation.controls if evaluation is not None else {},
        "model_artifacts": model_artifacts,
        "outer_heldout_read": False,
        "detector_validation_read": False,
    }
    _write_json(args.output_dir / "eval_metrics.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-cache", required=True, type=Path)
    parser.add_argument("--tune-cache", required=True, type=Path)
    parser.add_argument("--calibration-cache", required=True, type=Path)
    parser.add_argument("--split-manifest", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--annotation", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=LOCKED_SEED)
    return parser.parse_args()


def main() -> int:
    result = run(parse_args())
    print(json.dumps({"all_gates_passed": result["all_gates_passed"], "gates": result["gates"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
