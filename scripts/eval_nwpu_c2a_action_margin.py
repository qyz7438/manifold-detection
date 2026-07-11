"""Evaluate C2 checkpoints with the move gate removed and action margin only."""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "spectral_detection_posttrain" / "configs" / "versions" / "det.energy.native_budget1.c2a.margin.001.json"
C2_SCRIPT = ROOT / "scripts" / "train_nwpu_c2_native_budget1.py"


def _load_c2():
    spec = importlib.util.spec_from_file_location("c2_margin_source", C2_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError("cannot load C2 runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


c2 = _load_c2()


def load_config() -> dict[str, Any]:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "runs" / "nwpu_c2a_action_margin_smoke_s42")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--annotation", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--device", default=None)
    parser.add_argument("--require-clean-git", action="store_true")
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config()
    c2_config = c2.load_locked_config()
    source = ROOT / config["source"]["result"]
    if c2.m1.sha256_file(source) != config["source"]["result_sha256"]:
        raise RuntimeError("C2 source result hash mismatch")
    if args.require_clean_git and subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise RuntimeError("--require-clean-git requested but repository is dirty")

    from spectral_detection_posttrain.datasets import build_nwpu_vhr10_loaders
    from spectral_detection_posttrain.models import build_detector
    from spectral_detection_posttrain.utils.io import load_checkpoint
    from spectral_detection_posttrain.utils.seed import resolve_device, set_seed

    checkpoint = (args.checkpoint or ROOT / c2_config["detector"]["checkpoint"]).resolve()
    annotation = (args.annotation or ROOT / "data" / "NWPU_VHR10_coco.json").resolve()
    data_root = (args.data_root or ROOT / "data" / "NWPU VHR-10 dataset").resolve()
    c2.m1._verify_input_hashes(c2_config, checkpoint, annotation)
    set_seed(42)
    device = resolve_device({"device": args.device or ("cuda" if torch.cuda.is_available() else "cpu")})
    detector_config = c2.m1._make_detector_config(c2_config, data_root, annotation)
    detector = build_detector(detector_config).to(device)
    load_checkpoint(detector, checkpoint, device)
    detector.eval()
    for parameter in detector.parameters():
        parameter.requires_grad_(False)
    _, val_loader = build_nwpu_vhr10_loaders(detector_config, limit_train=1, limit_val=32, batch_size=1)

    arms = ("local_full", "feature_shuffle", "utility_shuffle")
    evaluations: dict[str, Any] = {}
    verified_checkpoints: dict[str, Any] = {}
    for arm in arms:
        path = source.parent / arm / "policy_best.pt"
        actual = c2.m1.sha256_file(path)
        if actual != config["source"]["checkpoints"][arm]:
            raise RuntimeError(f"{arm} checkpoint hash mismatch")
        policy = c2.m1._new_policy(c2_config, spatial_channels=256, device=device)
        policy.load_state_dict(torch.load(path, map_location=device)["state_dict"])
        evaluation, _ = c2.m1.evaluate_validation(
            detector,
            policy,
            val_loader,
            c2_config,
            device,
            include_action_margin_only=True,
        )
        evaluations[arm] = evaluation
        verified_checkpoints[arm] = {"path": str(path), "sha256": actual}

    full = evaluations["local_full"]
    identity = full["metrics"]["identity"]
    learned = full["metrics"]["action_margin_only"]
    delta75 = float(learned["ap75"] - identity["ap75"])
    delta50 = float(learned["ap50"] - identity["ap50"])
    control_delta = {
        arm: float(learned["ap75"] - evaluations[arm]["metrics"]["action_margin_only"]["ap75"])
        for arm in ("feature_shuffle", "utility_shuffle")
    }
    selected = int(full["diagnostics"]["action_margin_only"]["selected_count"])
    required = config["gates"]
    gates = {
        "nonzero_selection": selected >= int(required["min_selected_count"]),
        "detector_delta": delta75 >= float(required["min_ap75_delta_vs_identity"]) and delta50 >= float(required["min_ap50_delta_vs_identity"]),
        "control_delta": all(value >= float(required["min_ap75_delta_vs_each_control"]) for value in control_delta.values()),
    }
    result = {
        "completed": True,
        "version_id": config["version_id"],
        "experiment_scope": "smoke_val_32_reused_checkpoints",
        "claim_boundary": config["claim_boundary"],
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_result_sha256": config["source"]["result_sha256"],
        "checkpoints": verified_checkpoints,
        "evaluations": evaluations,
        "summary": {"selected_count": selected, "ap50_delta_vs_identity": delta50, "ap75_delta_vs_identity": delta75, "ap75_delta_vs_controls": control_delta},
        "gates": {"all_passed": all(gates.values()), "gates": gates},
    }
    args.run_dir.mkdir(parents=True, exist_ok=True)
    (args.run_dir / "eval_metrics.json").write_text(json.dumps(c2.m1._json_value(result), indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    result = run(parse_args())
    print(json.dumps({"summary": result["summary"], "gates": result["gates"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
