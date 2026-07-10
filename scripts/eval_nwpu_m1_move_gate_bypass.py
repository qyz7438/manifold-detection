"""Run the single frozen M1 move-gate bypass post-hoc diagnostic."""

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
LOCKED_CONFIG = ROOT / "spectral_detection_posttrain" / "configs" / "versions" / "det.energy.set_policy.movegate_diag.001.json"
M1_SCRIPT = ROOT / "scripts" / "train_nwpu_m1_set_policy.py"
M1_CONFIG = ROOT / "spectral_detection_posttrain" / "configs" / "versions" / "det.energy.set_policy.m1.001.json"
LOCKED_CONFIG_SHA256 = "cbf9e64042d1d7d73b37171b078aa95affb9fa991917a456dfd556fa1c86020b"
M1_CONFIG_SHA256 = "37238720b1eaf6a0a56ace8644a4f3248be566b301c3e6ad7393764b596daf32"
DIAGNOSTIC_ARM = "move_gate_bypass"
OUTPUT_MODES = ("identity", "learned_set", "move_gate_bypass")
PARITY_TOLERANCE = 1e-9


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_value(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def validate_locked_config(config: dict[str, Any]) -> None:
    if config.get("version_id") != "det.energy.set_policy.movegate_diag.001":
        raise ValueError("move-gate diagnostic config version is locked")
    if config.get("status") != "posthoc_diagnostic":
        raise ValueError("move-gate diagnostic config status is locked")
    if config.get("parent_version_id") != "det.energy.set_policy.m1.001":
        raise ValueError("diagnostic must be attached to canonical M1")
    if config.get("arms") != [DIAGNOSTIC_ARM] or config.get("output_modes") != list(OUTPUT_MODES):
        raise ValueError("diagnostic config must contain exactly one arm")

    dataset = config.get("dataset", {})
    detector = config.get("detector", {})
    policy = config.get("policy", {})
    source = config.get("source_m1", {})
    diagnostic = config.get("diagnostic", {})
    if dataset.get("validation_images") != 196 or dataset.get("val_image_ids_sha256") != "49f05cc9fa82ccaf924ff3be58a8f6376387c225e020ea6138182a4637219684":
        raise ValueError("validation split lock mismatch")
    if dataset.get("annotation_sha256") != "dde27d9362d9c6aa358a1cab160c9e075e57405d3fe876de049973c6e6150f0e":
        raise ValueError("annotation lock mismatch")
    if detector.get("checkpoint_sha256") != "de126708d063a82dd5c721799cd124fc4e52d28866139e103b1cd65427742027":
        raise ValueError("detector checkpoint lock mismatch")
    if (detector.get("score_threshold"), detector.get("nms_threshold"), detector.get("detections_per_image")) != (0.05, 0.5, 100):
        raise ValueError("detector postprocessing is locked")
    if policy.get("checkpoint_sha256") != "54a34c463ea5ed2fea58e7fa9534fc059867d2b7cf1444b32255aadb9055d9cd" or policy.get("energy_weight") != 0.05 or policy.get("frozen") is not True:
        raise ValueError("policy lock mismatch")
    if source.get("config") != "spectral_detection_posttrain/configs/versions/det.energy.set_policy.m1.001.json" or source.get("config_sha256") != M1_CONFIG_SHA256:
        raise ValueError("canonical M1 config lock mismatch")
    if source.get("result_sha256") != "9b089b6625d0dfc9d9fe6a1553be6d5646b78a8179c7de15e506b2e0215a88e5":
        raise ValueError("source M1 result lock mismatch")
    if diagnostic.get("eligibility") != "observable & action_margin > 0" or diagnostic.get("ranking") != "move_logit + action_margin":
        raise ValueError("move-gate bypass rule is locked")
    if diagnostic.get("action_budget") != 4 or diagnostic.get("energy_weight") != 0.05:
        raise ValueError("diagnostic action budget or energy weight is not locked")
    if any(diagnostic.get(key) is not False for key in ("retrain", "threshold_sweep", "clean_gain")):
        raise ValueError("diagnostic must not retrain, sweep thresholds, or claim a clean gain")


def load_locked_config(path: str | Path = LOCKED_CONFIG) -> dict[str, Any]:
    config_path = Path(path).resolve()
    if config_path != LOCKED_CONFIG.resolve():
        raise ValueError(f"diagnostic requires the canonical locked config: {LOCKED_CONFIG}")
    actual_hash = sha256_file(config_path)
    if actual_hash != LOCKED_CONFIG_SHA256:
        raise ValueError(f"diagnostic config SHA256 mismatch: expected {LOCKED_CONFIG_SHA256}, got {actual_hash}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_locked_config(config)
    return config


def _load_m1_module() -> Any:
    spec = importlib.util.spec_from_file_location("train_nwpu_m1_set_policy", M1_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load canonical M1 runner from {M1_SCRIPT}")
    train_nwpu_m1_set_policy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_nwpu_m1_set_policy)
    return train_nwpu_m1_set_policy


def _gate_values(payload: dict[str, Any]) -> dict[str, Any]:
    gates = payload.get("gates", {})
    nested = gates.get("gates") if isinstance(gates, dict) else None
    return nested if isinstance(nested, dict) else gates


def validate_source_m1_payload(payload: dict[str, Any]) -> bool:
    gates = _gate_values(payload)
    required = {
        "completed": payload.get("completed") is True,
        "G0_parity": gates.get("G0_parity") is True,
        "G1_gt_free_eval": gates.get("G1_gt_free_eval") is True,
        "G2_learned_vs_identity": gates.get("G2_learned_vs_identity") is False,
    }
    if not all(required.values()):
        raise ValueError(f"source M1 must have completed/G0/G1 true and G2 false: {required}")
    return True


def _resolved_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (ROOT / path).resolve()


def verify_locked_hashes(
    config: dict[str, Any],
    checkpoint: str | Path,
    policy_checkpoint: str | Path,
    annotation: str | Path,
    source_result: str | Path,
) -> dict[str, str]:
    paths = {
        "checkpoint": Path(checkpoint),
        "policy_checkpoint": Path(policy_checkpoint),
        "annotation": Path(annotation),
        "source_result": Path(source_result),
    }
    if any(not path.is_file() for path in paths.values()):
        missing = ", ".join(key for key, path in paths.items() if not path.is_file())
        raise FileNotFoundError(f"locked diagnostic artifacts are missing: {missing}")
    hashes = {f"{key}_sha256": sha256_file(path) for key, path in paths.items()}
    expected = {
        "checkpoint_sha256": config["detector"]["checkpoint_sha256"],
        "policy_checkpoint_sha256": config["policy"]["checkpoint_sha256"],
        "annotation_sha256": config["dataset"]["annotation_sha256"],
        "source_result_sha256": config["source_m1"]["result_sha256"],
    }
    if hashes != expected:
        raise ValueError(f"locked diagnostic artifact SHA256 mismatch: expected {expected}, got {hashes}")
    return hashes


def _load_source_result(config: dict[str, Any]) -> tuple[dict[str, Any], str]:
    path = _resolved_path(config["source_m1"]["result"])
    actual_hash = sha256_file(path)
    if actual_hash != config["source_m1"]["result_sha256"]:
        raise ValueError("source M1 result SHA256 mismatch")
    payload = json.loads(path.read_text(encoding="utf-8"))
    validate_source_m1_payload(payload)
    return payload, actual_hash


def verify_learned_set_reproduction(evaluation: dict[str, Any], source_payload: dict[str, Any]) -> bool:
    expected = source_payload["metrics"]["learned_set"]
    actual = evaluation["metrics"]["learned_set"]
    for key in ("ap50", "ap75", "false_positive_rate"):
        if abs(float(actual[key]) - float(expected[key])) > PARITY_TOLERANCE:
            raise ValueError(f"learned_set parity mismatch for {key}")
    if int(actual["num_predictions"]) != int(expected["num_predictions"]):
        raise ValueError("learned_set parity mismatch for num_predictions")
    return True


def diagnose_move_gate_bypass(identity: dict[str, Any], bypass: dict[str, Any]) -> dict[str, Any]:
    deltas = {
        "ap75": float(bypass["ap75"]) - float(identity["ap75"]),
        "ap50": float(bypass["ap50"]) - float(identity["ap50"]),
        "false_positive_rate": float(bypass["false_positive_rate"]) - float(identity["false_positive_rate"]),
        "num_predictions_relative": (int(bypass["num_predictions"]) - int(identity["num_predictions"])) / max(1, int(identity["num_predictions"])),
    }
    criteria = {
        "ap75": deltas["ap75"] >= 0.002,
        "ap50": deltas["ap50"] >= -0.002,
        "false_positive_rate": deltas["false_positive_rate"] <= 0.02,
        "num_predictions_relative": deltas["num_predictions_relative"] <= 0.05,
    }
    return {
        "diagnosis": "move_gate_is_primary_bottleneck" if all(criteria.values()) else "move_gate_bypass_not_sufficient",
        "criteria": criteria,
        "deltas": deltas,
        "completed": True,
        "posthoc_diagnostic": True,
        "not_clean_gain": True,
    }


def _policy_from_checkpoint(m1: Any, config: dict[str, Any], checkpoint: Path, device: torch.device) -> Any:
    payload = torch.load(checkpoint, map_location="cpu")
    state_dict = payload.get("state_dict") if isinstance(payload, dict) else None
    if not isinstance(state_dict, dict):
        raise ValueError("M1 policy checkpoint must contain state_dict")
    weight = state_dict.get("spatial_encoder.0.weight")
    if not torch.is_tensor(weight):
        raise ValueError("M1 policy checkpoint has no spatial encoder")
    spatial_size = int(config["policy"]["spatial_size"])
    channels_times_area = int(weight.shape[1])
    area = spatial_size * spatial_size
    if channels_times_area % area:
        raise ValueError("M1 policy spatial encoder shape is incompatible with locked spatial size")
    policy = m1._new_policy(config, channels_times_area // area, device)
    policy.load_state_dict(state_dict)
    policy.eval()
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    return policy


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=LOCKED_CONFIG)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--policy-checkpoint", type=Path)
    parser.add_argument("--annotation", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--source-result", type=Path)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "runs" / "nwpu_m1_movegate_bypass_diag_s42_fullval")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default=None)
    parser.add_argument("--require-full-validation", action="store_true")
    parser.add_argument("--require-clean-git", action="store_true")
    return parser.parse_args(argv)


def validate_cli_args(args: argparse.Namespace) -> None:
    if not args.require_full_validation:
        raise ValueError("the move-gate diagnostic requires full validation")


def run(args: argparse.Namespace) -> dict[str, Any]:
    validate_cli_args(args)
    config = load_locked_config(args.config)
    train_nwpu_m1_set_policy = _load_m1_module()
    m1_config = train_nwpu_m1_set_policy.load_locked_config(M1_CONFIG)
    if sha256_file(M1_CONFIG) != M1_CONFIG_SHA256:
        raise ValueError("canonical M1 config SHA256 mismatch")
    source_payload, source_result_hash = _load_source_result(config)

    checkpoint = _resolved_path(args.checkpoint or config["detector"]["checkpoint"])
    policy_checkpoint = _resolved_path(args.policy_checkpoint or config["policy"]["checkpoint"])
    annotation = _resolved_path(args.annotation or config["dataset"]["annotation"])
    source_result = _resolved_path(args.source_result or config["source_m1"]["result"])
    hashes = verify_locked_hashes(config, checkpoint, policy_checkpoint, annotation, source_result)
    data_root = _resolved_path(args.data_root or "data/NWPU VHR-10 dataset")

    from spectral_detection_posttrain.datasets import build_nwpu_vhr10_loaders
    from spectral_detection_posttrain.models import build_detector
    from spectral_detection_posttrain.utils.git_state import get_git_state
    from spectral_detection_posttrain.utils.io import load_checkpoint
    from spectral_detection_posttrain.utils.seed import resolve_device, set_seed

    if args.require_clean_git:
        status = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, check=True)
        if status.stdout.strip():
            raise RuntimeError("--require-clean-git requested but the repository is dirty")

    set_seed(int(m1_config["dataset"]["seed"]))
    device = resolve_device({"device": args.device or ("cuda" if torch.cuda.is_available() else "cpu")})
    detector_config = train_nwpu_m1_set_policy._make_detector_config(m1_config, data_root, annotation)
    detector = build_detector(detector_config).to(device)
    load_checkpoint(detector, checkpoint, device)
    detector.eval()
    for parameter in detector.parameters():
        parameter.requires_grad_(False)
    policy = _policy_from_checkpoint(train_nwpu_m1_set_policy, m1_config, policy_checkpoint, device)

    _, val_loader = build_nwpu_vhr10_loaders(detector_config, limit_train=None, limit_val=None, batch_size=1)
    val_manifest = train_nwpu_m1_set_policy.split_manifest(val_loader)
    train_nwpu_m1_set_policy.verify_validation_manifest(val_manifest, m1_config)
    if val_manifest != {"count": 196, "image_ids_sha256": config["dataset"]["val_image_ids_sha256"]}:
        raise ValueError(f"validation manifest mismatch: {val_manifest}")

    evaluation, _ = train_nwpu_m1_set_policy.evaluate_validation(
        detector,
        policy,
        val_loader,
        m1_config,
        device,
        include_move_gate_bypass=True,
    )
    verify_learned_set_reproduction(evaluation, source_payload)
    selected_metrics = {mode: evaluation["metrics"][mode] for mode in OUTPUT_MODES}
    diagnosis = diagnose_move_gate_bypass(selected_metrics["identity"], selected_metrics[DIAGNOSTIC_ARM])
    git_state = get_git_state(ROOT)
    result = {
        "completed": True,
        "posthoc_diagnostic": True,
        "not_clean_gain": True,
        "diagnosis": diagnosis["diagnosis"],
        "version_id": config["version_id"],
        "arm": DIAGNOSTIC_ARM,
        "metrics": _json_value(selected_metrics),
        "diagnostics": _json_value({mode: evaluation["diagnostics"][mode] for mode in OUTPUT_MODES}),
        "summary": _json_value(evaluation["summary"]),
        "diagnosis_detail": _json_value(diagnosis),
        "strict_parity": {
            "learned_set_reproduction": True,
            "ap50_tolerance": PARITY_TOLERANCE,
            "ap75_tolerance": PARITY_TOLERANCE,
            "false_positive_rate_tolerance": PARITY_TOLERANCE,
            "num_predictions_exact": True,
        },
        "source_hashes": {
            **hashes,
            "diagnostic_config_sha256": sha256_file(LOCKED_CONFIG),
            "m1_config_sha256": sha256_file(M1_CONFIG),
            "validation_manifest_sha256": config["dataset"]["val_image_ids_sha256"],
            "source_result_sha256": source_result_hash,
        },
        "git_state": git_state,
        "validation": {"scope": "full_val", "images": 196, "batch_size": 1, "manifest": val_manifest},
        "frozen": {"detector": True, "policy": True, "targets": None, "retrained": False, "threshold_sweep": False},
    }
    output = args.output or Path(args.run_dir) / "eval_metrics.json"
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_json_value(result), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = run(args)
    print(json.dumps({"output": str(args.output or Path(args.run_dir) / "eval_metrics.json"), "completed": result["completed"], "diagnosis": result["diagnosis"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
