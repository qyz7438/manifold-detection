"""Force one C3 action per image to diagnose conditional action ranking."""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CONFIG = ROOT / "spectral_detection_posttrain" / "configs" / "versions" / "det.energy.global_delta_u.c3a.forced.001.json"
C3_SCRIPT = ROOT / "scripts" / "train_nwpu_c3_global_delta_u.py"


def _load_c3():
    spec = importlib.util.spec_from_file_location("c3a_source", C3_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError("cannot load C3 runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


c3 = _load_c3()


def load_config() -> dict[str, Any]:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "runs" / "nwpu_c3a_forced_top1_smoke_s42")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--annotation", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--device", default=None)
    parser.add_argument("--require-clean-git", action="store_true")
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config()
    base = c3.load_locked_config()
    source = ROOT / config["source"]["result"]
    if c3.m1.sha256_file(source) != config["source"]["result_sha256"]:
        raise RuntimeError("C3 source result hash mismatch")
    cache = source.parent / "global_delta_u_cache.pt"
    if c3.m1.sha256_file(cache) != config["source"]["cache_sha256"]:
        raise RuntimeError("C3 cache hash mismatch")
    if args.require_clean_git and subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise RuntimeError("--require-clean-git requested but repository is dirty")

    from spectral_detection_posttrain.datasets import build_nwpu_vhr10_loaders
    from spectral_detection_posttrain.methods.energy_transport.global_top1 import GlobalTop1PolicyHead
    from spectral_detection_posttrain.methods.energy_transport.native_contract import build_native_c1_deltas
    from spectral_detection_posttrain.models import build_detector
    from spectral_detection_posttrain.utils.io import load_checkpoint
    from spectral_detection_posttrain.utils.seed import resolve_device, set_seed

    checkpoint = (args.checkpoint or ROOT / base["detector"]["checkpoint"]).resolve()
    annotation = (args.annotation or ROOT / "data" / "NWPU_VHR10_coco.json").resolve()
    data_root = (args.data_root or ROOT / "data" / "NWPU VHR-10 dataset").resolve()
    c3.m1._verify_input_hashes(base, checkpoint, annotation)
    set_seed(42)
    device = resolve_device({"device": args.device or ("cuda" if torch.cuda.is_available() else "cpu")})
    detector_config = c3.m1._make_detector_config(base, data_root, annotation)
    detector = build_detector(detector_config).to(device)
    load_checkpoint(detector, checkpoint, device)
    detector.eval()
    for parameter in detector.parameters():
        parameter.requires_grad_(False)
    _, val_loader = build_nwpu_vhr10_loaders(detector_config, limit_train=1, limit_val=32, batch_size=1)
    c3.verify_locked_manifest(base, c3.m1.split_manifest(val_loader), "val", 32)

    deltas = build_native_c1_deltas(float(base["candidate_pool"]["step"])).to(device)
    policies: dict[str, torch.nn.Module] = {}
    verified: dict[str, Any] = {}
    for arm in ("local_full", "feature_shuffle", "utility_shuffle"):
        path = source.parent / arm / "policy_best.pt"
        actual = c3.m1.sha256_file(path)
        if actual != config["source"]["checkpoints"][arm]:
            raise RuntimeError(f"{arm} checkpoint hash mismatch")
        policy = GlobalTop1PolicyHead(
            in_channels=256,
            num_classes=11,
            candidate_deltas=deltas,
            hidden_dim=int(base["policy"]["hidden_dim"]),
            spatial_size=int(base["policy"]["spatial_size"]),
            energy_weight=float(base["policy"]["energy_weight"]),
        ).to(device)
        policy.load_state_dict(torch.load(path, map_location=device)["state_dict"])
        policy.eval()
        policies[arm] = policy
        verified[arm] = {"path": str(path), "sha256": actual}
    evaluation = c3.evaluate_policies(detector, policies, val_loader, base, device, allow_noop=False)
    identity = evaluation["metrics"]["identity"]
    learned = evaluation["metrics"]["local_full"]
    delta50 = float(learned["ap50"] - identity["ap50"])
    delta75 = float(learned["ap75"] - identity["ap75"])
    controls = {
        arm: float(learned["ap75"] - evaluation["metrics"][arm]["ap75"])
        for arm in ("feature_shuffle", "utility_shuffle")
    }
    required = config["gates"]
    gates = {
        "nonzero_selection": evaluation["diagnostics"]["local_full"]["selected_count"] >= required["min_selected_count"],
        "detector_delta": delta75 >= required["min_ap75_delta_vs_identity"] and delta50 >= required["min_ap50_delta_vs_identity"],
        "control_delta": all(value >= required["min_ap75_delta_vs_each_control"] for value in controls.values()),
        "native_parity": evaluation["parity"]["passed"] and evaluation["parity"]["mismatched_images"] == 0,
    }
    result = {
        "completed": True,
        "scientific_status": "conditional_rank_signal" if all(gates.values()) else "conditional_rank_no_signal",
        "version_id": config["version_id"],
        "experiment_scope": "reused_checkpoint_smoke_val_32",
        "claim_boundary": config["claim_boundary"],
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_result_sha256": config["source"]["result_sha256"],
        "checkpoints": verified,
        "evaluation": evaluation,
        "summary": {"ap50_delta_vs_identity": delta50, "ap75_delta_vs_identity": delta75, "ap75_delta_vs_controls": controls},
        "gates": {"all_passed": all(gates.values()), "gates": gates},
        "validated_claim": False,
    }
    args.run_dir.mkdir(parents=True, exist_ok=True)
    (args.run_dir / "eval_metrics.json").write_text(json.dumps(c3.m1._json_value(result), indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    result = run(parse_args())
    print(json.dumps({"status": result["scientific_status"], "summary": result["summary"], "gates": result["gates"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
