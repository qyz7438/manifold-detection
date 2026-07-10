"""Calibrated detector-unseen validation for the unified NWPU Delta-U probe."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

import torch


ROOT = Path(__file__).resolve().parents[1]
LOCKED_CONFIG = (
    ROOT
    / "spectral_detection_posttrain"
    / "configs"
    / "versions"
    / "det.energy.joint_probe.002.json"
)
LOCKED_CONFIG_SHA256 = "79eceeec051245d91d3ccc42969879ea7c37519376a5aa9096ce28f6091cdfdd"
V1_SCRIPT = ROOT / "scripts" / "probe_nwpu_joint_delta_u.py"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_locked_config(path: str | Path = LOCKED_CONFIG) -> dict[str, Any]:
    config_path = Path(path).resolve()
    if config_path != LOCKED_CONFIG.resolve():
        raise ValueError(f"v2 joint probe requires canonical config path {LOCKED_CONFIG}")
    actual_hash = sha256_file(config_path)
    if actual_hash != LOCKED_CONFIG_SHA256:
        raise ValueError(
            f"canonical v2 config SHA256 mismatch: expected {LOCKED_CONFIG_SHA256}, got {actual_hash}"
        )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("version_id") != "det.energy.joint_probe.002":
        raise ValueError("canonical v2 version_id mismatch")
    if config.get("status") != "preregistered":
        raise ValueError("canonical v2 config must remain preregistered")
    return config


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=LOCKED_CONFIG)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit-validation", type=int)
    parser.add_argument("--rebuild-validation-cache", action="store_true")
    parser.add_argument("--require-clean-git", action="store_true")
    parser.add_argument("--require-full-validation", action="store_true")
    return parser.parse_args(argv)


def validate_cli_args(args: argparse.Namespace) -> None:
    if args.require_full_validation and args.limit_validation is not None:
        raise ValueError("full validation cannot be combined with --limit-validation")
    if args.limit_validation is not None and args.limit_validation <= 0:
        raise ValueError("--limit-validation must be positive")


def _load_script(path: Path, module_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load script module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _json_value(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _manifest(image_ids: Sequence[int]) -> dict[str, Any]:
    normalized = sorted(int(image_id) for image_id in image_ids)
    encoded = json.dumps(normalized, separators=(",", ":")).encode("ascii")
    return {
        "count": len(normalized),
        "image_ids_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def split_train_records(
    records: Sequence[tuple[dict[str, Any], dict[str, Any]]],
    *,
    calibration_fraction: float,
    seed: int,
) -> tuple[
    list[tuple[dict[str, Any], dict[str, Any]]],
    list[tuple[dict[str, Any], dict[str, Any]]],
]:
    from spectral_detection_posttrain.methods.energy_transport import group_heldout_split

    by_id: dict[int, tuple[dict[str, Any], dict[str, Any]]] = {}
    for detector_record, trace_record in records:
        image_id = int(detector_record["image_id"])
        if image_id != int(trace_record["image_id"]):
            raise ValueError("train cache pair image IDs differ")
        if image_id in by_id:
            raise ValueError("train cache contains duplicate image IDs")
        by_id[image_id] = (detector_record, trace_record)
    image_ids = torch.tensor(sorted(by_id), dtype=torch.long)
    split = group_heldout_split(image_ids, calibration_fraction, seed)
    fit_ids = image_ids[split.train_mask].tolist()
    calibration_ids = image_ids[split.heldout_mask].tolist()
    return [by_id[int(value)] for value in fit_ids], [
        by_id[int(value)] for value in calibration_ids
    ]


def require_disjoint_image_ids(
    train_image_ids: Sequence[int], validation_image_ids: Sequence[int]
) -> None:
    overlap = set(int(value) for value in train_image_ids) & set(
        int(value) for value in validation_image_ids
    )
    if overlap:
        raise ValueError(f"train/validation image overlap detected: {sorted(overlap)[:10]}")


def evaluate_v2_gates(payload: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    gates_config = config["gates"]
    validation = payload["validation"]
    support = validation["support"]
    raw = validation["raw"]
    calibrated = validation["calibrated"]
    bootstrap = validation["bootstrap"]
    calibration = payload["calibration"]["joint_full"]
    full = calibrated["joint_full"]
    no_edges = calibrated["joint_no_edges"]
    topology = calibrated["joint_edge_topology_shuffle"]
    spatial = calibrated["joint_spatial_shuffle"]

    gates = {
        "G0_validation_support": (
            int(support["candidate_count"])
            >= int(gates_config["min_validation_candidates"])
            and int(support["image_count"])
            >= int(gates_config["min_validation_candidate_images"])
            and float(support["topology_edge_change_fraction"])
            >= float(gates_config["min_topology_edge_change_fraction"])
        ),
        "G1_relational_ranking": (
            float(raw["joint_full"]["pairwise_accuracy"])
            >= float(gates_config["min_pairwise_accuracy"])
            and float(bootstrap["full_vs_no_edges"]["ci_low"])
            > float(gates_config["min_pairwise_ci_low_vs_controls"])
            and float(bootstrap["full_vs_spatial_shuffle"]["ci_low"])
            > float(gates_config["min_pairwise_ci_low_vs_controls"])
            and float(bootstrap["full_vs_edge_topology_shuffle"]["ci_low"])
            > float(gates_config["min_pairwise_ci_low_vs_controls"])
        ),
        "G2_fit_only_calibration": (
            calibration["used_identity_fallback"] is False
            and float(calibration["metrics"]["mean_delta_u_lcb"])
            > float(gates_config["min_calibration_lcb"])
        ),
        "G3_validation_safety": (
            int(full["selected_count"])
            >= int(gates_config["min_validation_selected_actions"])
            and float(full["mean_delta_u_per_image"])
            >= float(gates_config["min_validation_mean_delta_u"])
            and float(full["positive_precision_lift"])
            >= float(gates_config["min_validation_positive_precision_lift"])
            and float(full["action_image_rate"])
            <= float(gates_config["max_validation_action_image_rate"])
        ),
        "G4_context_utility": (
            float(full["mean_delta_u_per_image"])
            - float(no_edges["mean_delta_u_per_image"])
            >= float(gates_config["min_validation_gain_over_controls"])
            and float(full["mean_delta_u_per_image"])
            - float(spatial["mean_delta_u_per_image"])
            >= float(gates_config["min_validation_gain_over_controls"])
            and float(full["mean_delta_u_per_image"])
            - float(topology["mean_delta_u_per_image"])
            >= float(gates_config["min_validation_gain_over_controls"])
        ),
    }
    return {"all_passed": all(gates.values()), "gates": gates}


def _verify_file(path: Path, expected_sha256: str, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"required {label} is missing: {path}")
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ValueError(f"{label} SHA256 mismatch: expected {expected_sha256}, got {actual}")


def _calibration_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    calibration = config["calibration"]
    return {
        "target_epsilon": float(config["loss"]["target_epsilon"]),
        "min_selected_actions": int(calibration["min_selected_actions"]),
        "max_action_image_rate": float(calibration["max_action_image_rate"]),
        "min_positive_precision_lift": float(
            calibration["min_positive_precision_lift"]
        ),
        "min_mean_delta_u_lcb": float(calibration["min_mean_delta_u_lcb"]),
        "lcb_z": float(calibration["lcb_z"]),
    }


def topology_shuffle_coverage(
    records: Sequence[tuple[dict[str, Any], dict[str, Any]]],
    config: dict[str, Any],
) -> dict[str, float | int]:
    from spectral_detection_posttrain.methods.energy_transport import (
        build_proposal_set_edges,
        shuffle_edge_topology,
    )

    total_edges = 0
    changed_edges = 0
    for detector_record, _ in records:
        boxes = torch.as_tensor(detector_record["boxes"], dtype=torch.float32)
        labels = torch.as_tensor(detector_record["predicted_labels"], dtype=torch.long)
        scores = torch.as_tensor(detector_record["scores"], dtype=torch.float32)
        image_indices = torch.zeros(boxes.shape[0], dtype=torch.long)
        image_size = torch.tensor(detector_record["image_size"], dtype=torch.float32)
        edges = build_proposal_set_edges(
            boxes,
            labels,
            scores,
            image_indices,
            image_size,
            max_neighbors=int(config["model"]["max_neighbors"]),
            same_class_only=bool(config["model"]["same_class_edges"]),
        )
        shuffled = shuffle_edge_topology(
            edges,
            labels,
            image_indices,
            seed=31415 + int(detector_record["image_id"]),
        )
        total_edges += int(edges.edge_index.shape[1])
        changed_edges += int(edges.edge_index[1].ne(shuffled.edge_index[1]).sum().item())
    return {
        "topology_total_edges": total_edges,
        "topology_changed_edges": changed_edges,
        "topology_edge_change_fraction": changed_edges / max(1, total_edges),
    }


def _new_joint_model(
    detector_record: dict[str, Any],
    candidate_deltas: torch.Tensor,
    config: dict[str, Any],
    device: torch.device,
) -> torch.nn.Module:
    from spectral_detection_posttrain.methods.energy_transport import JointDeltaUProbe

    return JointDeltaUProbe(
        in_channels=int(detector_record["spatial_features"].shape[1]),
        num_classes=11,
        candidate_deltas=candidate_deltas,
        hidden_dim=int(config["model"]["hidden_dim"]),
    ).to(device)


def _load_arm_model(
    checkpoint: str | Path,
    detector_record: dict[str, Any],
    candidate_deltas: torch.Tensor,
    config: dict[str, Any],
    device: torch.device,
) -> torch.nn.Module:
    model = _new_joint_model(detector_record, candidate_deltas, config, device)
    payload = torch.load(Path(checkpoint), map_location=device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def _arm_rows(
    v1: Any,
    model: torch.nn.Module,
    records: Sequence[tuple[dict[str, Any], dict[str, Any]]],
    candidate_deltas: torch.Tensor,
    config: dict[str, Any],
    device: torch.device,
    arm: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return v1.predict_joint_arm(
        model,
        records,
        candidate_deltas,
        config,
        device,
        arm=arm,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    validate_cli_args(args)
    config_path = Path(args.config).resolve()
    config = load_locked_config(config_path)
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.require_clean_git:
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        if status.stdout.strip():
            raise RuntimeError("--require-clean-git requested but the repository is dirty")

    v1 = _load_script(V1_SCRIPT, "joint_probe_v2_v1")
    m0 = _load_script(ROOT / "scripts" / "eval_nwpu_m0_set_oracle.py", "joint_probe_v2_m0")
    m1 = _load_script(ROOT / "scripts" / "train_nwpu_m1_set_policy.py", "joint_probe_v2_m1")
    source_cache = (ROOT / config["source_train_cache"]["path"]).resolve()
    source_metadata_path = (ROOT / config["source_train_cache"]["metadata"]).resolve()
    m0_config_path = (ROOT / config["source_m0"]["config"]).resolve()
    m0_result_path = (ROOT / config["source_m0"]["validated_result"]).resolve()
    m1_config_path = (ROOT / config["source_m1"]["config"]).resolve()
    checkpoint = (ROOT / config["detector"]["checkpoint"]).resolve()
    annotation = (ROOT / "data" / "NWPU_VHR10_coco.json").resolve()
    data_root = (ROOT / "data" / "NWPU VHR-10 dataset").resolve()
    locked_files = (
        (source_cache, config["source_train_cache"]["sha256"], "source train cache"),
        (
            source_metadata_path,
            config["source_train_cache"]["metadata_sha256"],
            "source train cache metadata",
        ),
        (m0_config_path, config["source_m0"]["config_sha256"], "M0 config"),
        (
            m0_result_path,
            config["source_m0"]["validated_result_sha256"],
            "M0 result",
        ),
        (m1_config_path, config["source_m1"]["config_sha256"], "M1 config"),
        (checkpoint, config["detector"]["checkpoint_sha256"], "detector checkpoint"),
        (annotation, config["dataset"]["annotation_sha256"], "annotation"),
    )
    for path, expected_hash, label in locked_files:
        _verify_file(path, expected_hash, label)

    source_metadata = json.loads(source_metadata_path.read_text(encoding="utf-8"))
    if source_metadata.get("pairing") != "same_detector_forward":
        raise ValueError("source train cache is not atomically paired")
    expected_train_manifest = {
        "count": int(config["dataset"]["train_images"]),
        "image_ids_sha256": config["dataset"]["train_image_ids_sha256"],
    }
    if source_metadata.get("train_manifest") != expected_train_manifest:
        raise ValueError("source train cache manifest mismatch")
    train_records = v1.read_probe_cache(source_cache)
    if _manifest([record[0]["image_id"] for record in train_records]) != expected_train_manifest:
        raise ValueError("source train cache records do not match the locked manifest")
    fit_records, calibration_records = split_train_records(
        train_records,
        calibration_fraction=float(config["split"]["calibration_fraction"]),
        seed=int(config["split"]["seed"]),
    )

    from spectral_detection_posttrain.datasets import build_nwpu_vhr10_loaders
    from spectral_detection_posttrain.experiments.metadata import collect_experiment_metadata
    from spectral_detection_posttrain.methods.energy_transport import (
        calibrate_conservative_threshold,
        calibrated_selection_metrics,
        constant_utility_baselines,
        imagewise_pairwise_accuracy,
        paired_bootstrap_mean_difference,
        build_symmetric_box_candidates,
    )
    from spectral_detection_posttrain.models import build_detector
    from spectral_detection_posttrain.utils.io import load_checkpoint
    from spectral_detection_posttrain.utils.seed import resolve_device, set_seed

    set_seed(int(config["dataset"]["seed"]))
    device = resolve_device({"device": args.device or ("cuda" if torch.cuda.is_available() else "cpu")})
    v1.validate_deterministic_cuda_environment(device)
    torch.use_deterministic_algorithms(True, warn_only=True)
    candidate_deltas = build_symmetric_box_candidates(
        tuple(float(value) for value in config["candidate_pool"]["step_sizes"])
    ).to(device)
    if int(candidate_deltas.shape[0]) != int(config["candidate_pool"]["candidate_count"]):
        raise ValueError("locked candidate grid count mismatch")

    calibration_results: dict[str, Any] = {}
    training_results: dict[str, Any] = {}
    checkpoints: dict[str, str] = {}
    for arm in config["arms"]:
        model, training = v1.train_joint_arm(
            fit_records,
            candidate_deltas,
            config,
            device,
            run_dir,
            arm=str(arm),
            num_classes=11,
        )
        predicted, target, image_ids = _arm_rows(
            v1,
            model,
            calibration_records,
            candidate_deltas,
            config,
            device,
            str(arm),
        )
        calibrated = calibrate_conservative_threshold(
            predicted, target, image_ids, **_calibration_kwargs(config)
        )
        calibration_results[str(arm)] = {
            "threshold": calibrated.threshold,
            "used_identity_fallback": calibrated.used_identity_fallback,
            "candidates_evaluated": calibrated.candidates_evaluated,
            "metrics": calibrated.metrics,
            "raw": v1.probe_metrics(
                predicted,
                target,
                image_ids,
                action_budget=1,
                target_epsilon=float(config["loss"]["target_epsilon"]),
            ),
        }
        training_results[str(arm)] = training
        checkpoints[str(arm)] = str(training["selected_checkpoint"])
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    m0_config = m0.load_locked_config(m0_config_path)
    m1_config = m1.load_locked_config(m1_config_path)
    loader_config = m1._make_detector_config(m1_config, data_root, annotation)
    _, validation_loader = build_nwpu_vhr10_loaders(
        loader_config,
        limit_train=1,
        limit_val=args.limit_validation,
        batch_size=1,
    )
    validation_manifest = m1.split_manifest(validation_loader)
    expected_validation_manifest = {
        "count": int(config["dataset"]["validation_images"]),
        "image_ids_sha256": config["dataset"]["validation_image_ids_sha256"],
    }
    if args.limit_validation is None and validation_manifest != expected_validation_manifest:
        raise ValueError("validation manifest mismatch")
    validation_image_ids = getattr(validation_loader.dataset, "img_ids", None)
    if validation_image_ids is None:
        raise ValueError("validation dataset does not expose image IDs")
    require_disjoint_image_ids(
        [record[0]["image_id"] for record in train_records], validation_image_ids
    )

    code_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()
    validation_cache_path = run_dir / "joint_utility_validation_cache.pt"
    validation_metadata_path = run_dir / "joint_utility_validation_cache_metadata.json"
    expected_validation_metadata = {
        "schema_version": int(config["validation_cache"]["schema_version"]),
        "pairing": config["validation_cache"]["pairing"],
        "config_sha256": sha256_file(config_path),
        "source_train_cache_sha256": config["source_train_cache"]["sha256"],
        "validation_manifest": validation_manifest,
        "code_commit": code_commit,
    }
    if validation_cache_path.exists() and not args.rebuild_validation_cache:
        if not validation_metadata_path.is_file():
            raise FileNotFoundError("validation cache exists without metadata")
        validation_metadata = json.loads(validation_metadata_path.read_text(encoding="utf-8"))
        for key, expected in expected_validation_metadata.items():
            if validation_metadata.get(key) != expected:
                raise ValueError(f"validation cache metadata mismatch for {key}")
        if sha256_file(validation_cache_path) != validation_metadata.get("cache_sha256"):
            raise ValueError("validation cache SHA256 mismatch")
        validation_records = v1.read_probe_cache(validation_cache_path)
    else:
        detector = build_detector(loader_config).to(device)
        load_checkpoint(detector, checkpoint, device)
        detector.eval()
        for parameter in detector.parameters():
            parameter.requires_grad_(False)
        validation_records = v1.build_probe_cache_records(
            detector,
            validation_loader,
            m0,
            m1,
            m0_config,
            config,
            device,
        )
        v1.write_probe_cache(validation_cache_path, validation_records)
        trace_records = [trace for _, trace in validation_records]
        detector_records = [record for record, _ in validation_records]
        validation_metadata = {
            **expected_validation_metadata,
            "cache_sha256": sha256_file(validation_cache_path),
            "trace_manifest": _manifest([record["image_id"] for record in trace_records]),
            **v1._trace_statistics(trace_records),
            **m1.summarize_cache_storage(detector_records),
        }
        validation_metadata_path.write_text(
            json.dumps(validation_metadata, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        del detector
        if device.type == "cuda":
            torch.cuda.empty_cache()

    validation_raw: dict[str, Any] = {}
    validation_calibrated: dict[str, Any] = {}
    imagewise: dict[str, dict[int, float]] = {}
    shared_target: torch.Tensor | None = None
    shared_image_ids: torch.Tensor | None = None
    for arm in config["arms"]:
        model = _load_arm_model(
            checkpoints[str(arm)],
            validation_records[0][0],
            candidate_deltas,
            config,
            device,
        )
        predicted, target, image_ids = _arm_rows(
            v1,
            model,
            validation_records,
            candidate_deltas,
            config,
            device,
            str(arm),
        )
        if shared_target is None:
            shared_target = target
            shared_image_ids = image_ids
        elif not torch.equal(shared_target, target) or not torch.equal(shared_image_ids, image_ids):
            raise ValueError("validation arms do not share identical target rows")
        validation_raw[str(arm)] = v1.probe_metrics(
            predicted,
            target,
            image_ids,
            action_budget=1,
            target_epsilon=float(config["loss"]["target_epsilon"]),
        )
        threshold = calibration_results[str(arm)]["threshold"]
        validation_calibrated[str(arm)] = calibrated_selection_metrics(
            predicted,
            target,
            image_ids,
            threshold=float(threshold),
            target_epsilon=float(config["loss"]["target_epsilon"]),
            lcb_z=float(config["calibration"]["lcb_z"]),
        )
        imagewise[str(arm)] = imagewise_pairwise_accuracy(
            predicted,
            target,
            image_ids,
            target_epsilon=float(config["loss"]["target_epsilon"]),
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if shared_target is None or shared_image_ids is None:
        raise RuntimeError("validation evaluation produced no shared utility rows")
    bootstrap_config = config["bootstrap"]
    bootstrap = {
        "full_vs_no_edges": paired_bootstrap_mean_difference(
            imagewise["joint_full"],
            imagewise["joint_no_edges"],
            resamples=int(bootstrap_config["resamples"]),
            seed=int(bootstrap_config["seed"]),
            confidence=float(bootstrap_config["confidence"]),
        ),
        "full_vs_spatial_shuffle": paired_bootstrap_mean_difference(
            imagewise["joint_full"],
            imagewise["joint_spatial_shuffle"],
            resamples=int(bootstrap_config["resamples"]),
            seed=int(bootstrap_config["seed"]) + 1,
            confidence=float(bootstrap_config["confidence"]),
        ),
        "full_vs_edge_topology_shuffle": paired_bootstrap_mean_difference(
            imagewise["joint_full"],
            imagewise["joint_edge_topology_shuffle"],
            resamples=int(bootstrap_config["resamples"]),
            seed=int(bootstrap_config["seed"]) + 2,
            confidence=float(bootstrap_config["confidence"]),
        ),
    }
    topology_coverage = topology_shuffle_coverage(validation_records, config)
    validation = {
        "scope": "full_validation" if args.limit_validation is None else "smoke_validation",
        "validation_evaluated_once": True,
        "support": {
            "candidate_count": int(validation_raw["joint_full"]["candidate_count"]),
            "image_count": int(validation_raw["joint_full"]["image_count"]),
            "manifest_image_count": int(validation_manifest["count"]),
            **topology_coverage,
        },
        "raw": validation_raw,
        "calibrated": validation_calibrated,
        "baselines": constant_utility_baselines(
            shared_target,
            shared_image_ids,
            target_epsilon=float(config["loss"]["target_epsilon"]),
        ),
        "bootstrap": bootstrap,
    }
    gate_payload = {"validation": validation, "calibration": calibration_results}
    gates = evaluate_v2_gates(gate_payload, config)
    environment = collect_experiment_metadata(config, config_path=config_path)
    environment.update(
        {
            "python_version": sys.version,
            "cuda_runtime": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "deterministic_algorithms": True,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        }
    )
    result = {
        "completed": True,
        "scientific_status": "probe_passed" if gates["all_passed"] else "probe_failed",
        "version_id": config["version_id"],
        "claim_boundary": config["probe"]["claim_boundary"],
        "config": config,
        "inputs": {
            "code_commit": code_commit,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": config["detector"]["checkpoint_sha256"],
            "annotation": str(annotation),
            "annotation_sha256": config["dataset"]["annotation_sha256"],
            "source_train_cache": str(source_cache),
            "source_train_cache_sha256": config["source_train_cache"]["sha256"],
            "fit_manifest": _manifest([record[0]["image_id"] for record in fit_records]),
            "calibration_manifest": _manifest(
                [record[0]["image_id"] for record in calibration_records]
            ),
            "validation_manifest": validation_manifest,
            "environment": environment,
        },
        "training": training_results,
        "calibration": calibration_results,
        "validation_cache": {
            "path": str(validation_cache_path),
            "metadata": validation_metadata,
        },
        "validation": validation,
        "gates": gates,
    }
    output = run_dir / "eval_metrics.json"
    output.write_text(
        json.dumps(_json_value(result), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = run(args)
    print(
        json.dumps(
            {
                "output": str(Path(args.run_dir).resolve() / "eval_metrics.json"),
                "completed": result["completed"],
                "gates": result["gates"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
