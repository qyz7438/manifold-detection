"""Adaptive train-cache-only audit of step-0.05 NWPU action strata."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path
from typing import Any, Sequence

import torch


ROOT = Path(__file__).resolve().parents[1]
V1_SCRIPT = ROOT / "scripts" / "probe_nwpu_joint_delta_u.py"
V2_SCRIPT = ROOT / "scripts" / "probe_nwpu_joint_delta_u_v2.py"
LOCKED_CONFIG = (
    ROOT
    / "spectral_detection_posttrain"
    / "configs"
    / "versions"
    / "det.energy.step005_audit.001.json"
)
LOCKED_CONFIG_SHA256 = "c3979fe727f8c3a45981b8ca91310e39ae4ad305ccd2064ff64e633e87717e02"
DEFAULT_RUN_DIR = ROOT / "runs" / "nwpu_step005_strata_s42_adaptive"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_locked_config(path: str | Path = LOCKED_CONFIG) -> dict[str, Any]:
    resolved = Path(path).resolve()
    if resolved != LOCKED_CONFIG.resolve():
        raise ValueError(f"config must be the locked file: {LOCKED_CONFIG}")
    if sha256_file(resolved) != LOCKED_CONFIG_SHA256:
        raise ValueError("locked step-0.05 audit config hash mismatch")
    return json.loads(resolved.read_text(encoding="utf-8"))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(LOCKED_CONFIG))
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--device")
    parser.add_argument("--require-clean-git", action="store_true")
    return parser.parse_args(argv)


def _load_script(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load script: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _git_value(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


def _manifest(records: Sequence[tuple[dict[str, Any], dict[str, Any]]]) -> dict[str, Any]:
    image_ids = sorted(int(record[0]["image_id"]) for record in records)
    encoded = json.dumps(image_ids, separators=(",", ":")).encode("ascii")
    return {
        "count": len(image_ids),
        "image_ids_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _image_ids(records: Sequence[tuple[dict[str, Any], dict[str, Any]]]) -> list[int]:
    return [int(record[0]["image_id"]) for record in records]


def _stage_summaries(rows: Any, manifest: Sequence[int], config: dict[str, Any]) -> dict[str, Any]:
    from spectral_detection_posttrain.methods.energy_transport.step_strata import (
        strata_masks,
        summarize_stratum,
    )

    kwargs = {
        "manifest_image_ids": manifest,
        "target_epsilon": float(config["statistics"]["target_epsilon"]),
        "lcb_z": float(config["statistics"]["lcb_z"]),
    }
    result = {
        "all_step005": summarize_stratum(
            rows, torch.ones(rows.target.numel(), dtype=torch.bool), **kwargs
        )
    }
    for key, mask in strata_masks(rows).items():
        result[key] = summarize_stratum(rows, mask, **kwargs)
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = load_locked_config(args.config)
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.require_clean_git and _git_value("status", "--porcelain"):
        raise RuntimeError("--require-clean-git requested but repository is dirty")
    v1 = _load_script(V1_SCRIPT, "step005_v1")
    v2 = _load_script(V2_SCRIPT, "step005_v2")
    source_cache = (ROOT / config["source_train_cache"]["path"]).resolve()
    source_metadata = (ROOT / config["source_train_cache"]["metadata"]).resolve()
    v2._verify_file(
        source_cache, config["source_train_cache"]["sha256"], "source train cache"
    )
    v2._verify_file(
        source_metadata,
        config["source_train_cache"]["metadata_sha256"],
        "source train cache metadata",
    )
    metadata = json.loads(source_metadata.read_text(encoding="utf-8"))
    if metadata.get("pairing") != config["source_train_cache"]["pairing"]:
        raise ValueError("source train cache pairing mismatch")
    records = v1.read_probe_cache(source_cache)
    expected_manifest = {
        "count": int(config["dataset"]["train_images"]),
        "image_ids_sha256": config["dataset"]["train_image_ids_sha256"],
    }
    if _manifest(records) != expected_manifest:
        raise ValueError("source train cache manifest mismatch")
    outer_fit, outer_heldout = v2.split_train_records(
        records,
        calibration_fraction=float(config["split"]["outer_heldout_fraction"]),
        seed=int(config["split"]["outer_seed"]),
    )
    if _manifest(outer_fit) != config["split"]["outer_fit_manifest"]:
        raise ValueError("outer fit manifest mismatch")
    if _manifest(outer_heldout) != config["split"]["outer_heldout_manifest"]:
        raise ValueError("outer heldout manifest mismatch")
    inner_fit, inner_tune = v2.split_train_records(
        outer_fit,
        calibration_fraction=float(config["split"]["inner_tune_fraction"]),
        seed=int(config["split"]["inner_seed"]),
    )

    from spectral_detection_posttrain.methods.energy_transport.step_strata import (
        discover_stable_strata,
        evaluate_step_strata_gates,
        extract_step_action_rows,
        select_primary_stratum,
        strata_masks,
        summarize_stratum,
        target_permutation_control,
    )
    from spectral_detection_posttrain.utils.seed import resolve_device

    device = resolve_device(
        {"device": args.device or ("cuda" if torch.cuda.is_available() else "cpu")}
    )
    fit_rows = extract_step_action_rows(
        inner_fit,
        step=float(config["action"]["step"]),
        small_area_max=float(config["scale_bins"]["small_area_max"]),
        medium_area_max=float(config["scale_bins"]["medium_area_max"]),
    )
    tune_rows = extract_step_action_rows(
        inner_tune,
        step=float(config["action"]["step"]),
        small_area_max=float(config["scale_bins"]["small_area_max"]),
        medium_area_max=float(config["scale_bins"]["medium_area_max"]),
    )
    fit_summaries = _stage_summaries(fit_rows, _image_ids(inner_fit), config)
    tune_summaries = _stage_summaries(tune_rows, _image_ids(inner_tune), config)
    selected = discover_stable_strata(
        fit_summaries,
        tune_summaries,
        min_fit_candidates=int(config["discovery"]["min_fit_candidates"]),
        min_fit_images=int(config["discovery"]["min_fit_images"]),
        min_tune_candidates=int(config["discovery"]["min_tune_candidates"]),
        min_tune_images=int(config["discovery"]["min_tune_images"]),
    )
    primary = select_primary_stratum(selected, tune_summaries)

    outer_rows = extract_step_action_rows(
        outer_heldout,
        step=float(config["action"]["step"]),
        small_area_max=float(config["scale_bins"]["small_area_max"]),
        medium_area_max=float(config["scale_bins"]["medium_area_max"]),
    )
    full_rows = extract_step_action_rows(
        records,
        step=float(config["action"]["step"]),
        small_area_max=float(config["scale_bins"]["small_area_max"]),
        medium_area_max=float(config["scale_bins"]["medium_area_max"]),
    )
    if full_rows.target.numel() != int(config["action"]["expected_full_candidates"]):
        raise ValueError("full step-0.05 candidate support mismatch")
    if torch.unique(full_rows.image_ids).numel() != int(
        config["action"]["expected_full_candidate_images"]
    ):
        raise ValueError("full step-0.05 image support mismatch")
    outer_manifest = _image_ids(outer_heldout)
    outer_all = summarize_stratum(
        outer_rows,
        torch.ones(outer_rows.target.numel(), dtype=torch.bool),
        manifest_image_ids=outer_manifest,
        target_epsilon=float(config["statistics"]["target_epsilon"]),
        lcb_z=float(config["statistics"]["lcb_z"]),
    )
    strata_payload: dict[str, Any] = {}
    if primary is not None:
        outer_masks = strata_masks(outer_rows)
        outer_mask = outer_masks.get(primary)
        if outer_mask is None:
            outer_mask = torch.zeros(outer_rows.target.numel(), dtype=torch.bool)
        outer_summary = summarize_stratum(
            outer_rows,
            outer_mask,
            manifest_image_ids=outer_manifest,
            target_epsilon=float(config["statistics"]["target_epsilon"]),
            lcb_z=float(config["statistics"]["lcb_z"]),
        )
        strata_payload[primary] = {
            "fit": fit_summaries[primary],
            "tune": tune_summaries[primary],
            "outer": outer_summary,
            "target_permutation": target_permutation_control(
                outer_rows,
                outer_mask,
                manifest_image_ids=outer_manifest,
                trials=int(config["statistics"]["target_permutation_trials"]),
                seed=int(config["statistics"]["target_permutation_seed"]),
            ),
        }
    gate_payload = {
        "outer_support": {
            "candidate_count": int(outer_rows.target.numel()),
            "image_count": int(torch.unique(outer_rows.image_ids).numel()),
        },
        "selected_strata": selected,
        "primary_stratum": primary,
        "strata": strata_payload,
    }
    gates = evaluate_step_strata_gates(gate_payload, config)
    git_status = _git_value("status", "--porcelain")
    payload = {
        "completed": True,
        "scientific_status": "audit_passed" if gates["all_passed"] else "audit_failed",
        "version_id": config["version_id"],
        "claim_boundary": config["probe"]["claim_boundary"],
        "inputs": {
            "code_commit": _git_value("rev-parse", "HEAD"),
            "git_dirty": bool(git_status),
            "config_sha256": LOCKED_CONFIG_SHA256,
            "source_train_cache_sha256": config["source_train_cache"]["sha256"],
            "inner_fit_manifest": _manifest(inner_fit),
            "inner_tune_manifest": _manifest(inner_tune),
            "outer_heldout_manifest": _manifest(outer_heldout),
            "device": str(device),
        },
        "protocol": {
            "adaptive": True,
            "fit_discovers_tune_confirms": True,
            "single_primary_stratum": True,
            "outer_evaluated_after_freeze": True,
            "same_image_uniform_control_is_exact": True,
        },
        "support": {
            "full": {
                "candidate_count": int(full_rows.target.numel()),
                "image_count": int(torch.unique(full_rows.image_ids).numel()),
            },
            "fit": fit_summaries["all_step005"],
            "tune": tune_summaries["all_step005"],
            "outer": outer_all,
        },
        "fit_strata": fit_summaries,
        "tune_strata": tune_summaries,
        "selected_strata": selected,
        "primary_stratum": primary,
        "outer_primary": strata_payload.get(primary) if primary is not None else None,
        "gates": gates,
    }
    result_path = run_dir / "eval_metrics.json"
    result_path.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    payload["output"] = str(result_path)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    payload = run(parse_args(argv))
    print(
        json.dumps(
            {
                "completed": payload["completed"],
                "scientific_status": payload["scientific_status"],
                "primary_stratum": payload["primary_stratum"],
                "gates": payload["gates"],
                "output": payload["output"],
            },
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
