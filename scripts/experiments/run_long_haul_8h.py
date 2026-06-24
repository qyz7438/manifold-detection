r"""8-hour autonomous long-haul experiment controller.

Drives the experiments described in:
    docs/research_directions/LONG_HAUL_8H_PLAN.md

Phases:
    P0  Research-first documentation (already captured in the plan).
    P1  Adversarial patch attack smoke test on Penn-Fudan.
    P2  SpectralChordDefense smoke test on attacked images.
    P3  Full Plan B experiment on validation set + ablations.
    P4  Plan C semantic segmentation smoke test (placeholder).
    P5  Plan D classification smoke test (placeholder).
    P6  Result analysis and next-step recommendation.

State is persisted so the run can be resumed after interruption.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from spectral_detection_posttrain.utils.io import save_json


PHASES = ["P1", "P2", "P3", "P4", "P5", "P6"]
MAX_RETRIES = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="8-hour long-haul experiment controller")
    parser.add_argument("--output-dir", default="runs/experiments/long_haul_8h")
    parser.add_argument("--max-hours", type=float, default=8.0)
    parser.add_argument("--checkpoint", default="runs/canonical_baseline_10ep_gpu_20260616_bg/checkpoint_last.pth")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-plans", default="")
    return parser.parse_args()


def now_str() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def load_state(state_path: Path) -> dict[str, Any]:
    if state_path.exists():
        with open(state_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"phase": "P1", "status": "pending", "history": [], "retries": {}}


def save_state(state: dict[str, Any], state_path: Path) -> None:
    save_json(state, state_path)


def append_log(log_path: Path, message: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}] {message}\n")


def run_command_and_parse_metrics(cmd: list[str], metrics_path: Path, log_path: Path | None = None) -> dict[str, Any]:
    """Run a command and parse the produced metrics JSON."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if log_path is not None:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"\n=== {' '.join(cmd[-4:])} ===\n")
            f.write(result.stdout)
    if result.returncode != 0:
        return {"error": result.stdout, "returncode": result.returncode, "phase_failed": True}
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"error": "metrics.json not found", "stdout": result.stdout, "phase_failed": True}


def mutate_attack_config(retry: int) -> dict[str, Any]:
    """Return progressively stronger attack parameters."""
    return {
        "patch_size": 80 + retry * 16,
        "attack_steps": 300 + retry * 100,
        "attack_lr": 0.5,
        "attack_momentum": 0.9,
    }


def mutate_defense_config(retry: int) -> dict[str, Any]:
    """Return progressively gentler defense parameters."""
    return {
        "defense_size": 128,
        "anomaly_threshold": 2.5 + retry * 0.5,
        "latent_dim": 256,
    }


def run_phase(
    phase: str,
    args: argparse.Namespace,
    state: dict[str, Any],
) -> dict[str, Any]:
    """Execute one phase and return a result dict."""
    output_dir = Path(args.output_dir)
    log_path = output_dir / "LONG_HAUL_LOG.txt"
    python = "E:/anaconda/01/envs/RLimage/python.exe"

    retries = state.setdefault("retries", {})
    retry = int(retries.get(phase, 0))

    append_log(log_path, f"{phase} started (retry={retry})")

    if phase == "P1":
        cfg = mutate_attack_config(retry)
        out = output_dir / f"P1_attack_smoke_r{retry}"
        cmd = [
            python, "-u", str(PROJECT_ROOT / "scripts/experiments/plan_b_smoke_test.py"),
            "--checkpoint", args.checkpoint,
            "--output-dir", str(out),
            "--n-images", "10",
            "--defense-size", "128",
            "--latent-dim", "256",
            "--anomaly-threshold", "2.5",
            "--patch-size", str(cfg["patch_size"]),
            "--attack-steps", str(cfg["attack_steps"]),
            "--attack-lr", str(cfg["attack_lr"]),
            "--attack-momentum", str(cfg["attack_momentum"]),
            "--device", args.device,
            "--seed", str(args.seed),
        ]
        metrics = run_command_and_parse_metrics(cmd, out / "metrics.json", output_dir / "subprocess.log")
        if metrics.get("phase_failed", False):
            return {"status": "error", "metrics": metrics, "success": False, "detail": "subprocess crashed"}
        ap50_drop = metrics.get("AP50_clean", 1.0) - metrics.get("AP50_adv", 1.0)
        success = ap50_drop >= 0.10
        return {
            "status": "completed" if success else "failed",
            "metrics": metrics,
            "success": success,
            "detail": f"AP50_drop={ap50_drop:.4f} cfg={cfg}",
        }

    if phase == "P2":
        cfg = mutate_defense_config(retry)
        out = output_dir / f"P2_defense_smoke_r{retry}"
        cmd = [
            python, "-u", str(PROJECT_ROOT / "scripts/experiments/plan_b_smoke_test.py"),
            "--checkpoint", args.checkpoint,
            "--output-dir", str(out),
            "--n-images", "10",
            "--defense-size", str(cfg["defense_size"]),
            "--latent-dim", str(cfg["latent_dim"]),
            "--anomaly-threshold", str(cfg["anomaly_threshold"]),
            "--patch-size", "80",
            "--attack-steps", "300",
            "--attack-lr", "0.5",
            "--attack-momentum", "0.9",
            "--device", args.device,
            "--seed", str(args.seed),
        ]
        metrics = run_command_and_parse_metrics(cmd, out / "metrics.json", output_dir / "subprocess.log")
        if metrics.get("phase_failed", False):
            return {"status": "error", "metrics": metrics, "success": False, "detail": "subprocess crashed"}
        recovery = metrics.get("recovery_rate", 0.0)
        clean_drop = metrics.get("clean_drop", 1.0)
        success = recovery >= 0.5 and clean_drop <= 0.15
        return {
            "status": "completed" if success else "failed",
            "metrics": metrics,
            "success": success,
            "detail": f"recovery={recovery:.4f}, clean_drop={clean_drop:.4f} cfg={cfg}",
        }

    if phase == "P3":
        # Use a fixed 20-image subset to avoid OOM/crashes on the full val set.
        out = output_dir / "P3_full_defense"
        cmd = [
            python, "-u", str(PROJECT_ROOT / "scripts/experiments/plan_b_full_experiment.py"),
            "--checkpoint", args.checkpoint,
            "--output-dir", str(out),
            "--n-images", "20",
            "--defense-size", "128",
            "--latent-dim", "256",
            "--anomaly-threshold", "3.0",
            "--patch-size", "80",
            "--attack-steps", "300",
            "--attack-lr", "0.5",
            "--attack-momentum", "0.9",
            "--device", args.device,
            "--seed", str(args.seed),
        ]
        metrics = run_command_and_parse_metrics(cmd, out / "metrics.json", output_dir / "subprocess.log")
        if metrics.get("phase_failed", False):
            return {"status": "error", "metrics": metrics, "success": False, "detail": "subprocess crashed"}
        recovery = metrics.get("recovery_rate", 0.0)
        clean_drop = metrics.get("clean_drop", 1.0)
        success = recovery >= 0.5 and clean_drop <= 0.15
        return {
            "status": "completed" if success else "failed",
            "metrics": metrics,
            "success": success,
            "detail": f"full recovery={recovery:.4f}, clean_drop={clean_drop:.4f}",
        }

    if phase == "P4":
        return {
            "status": "skipped",
            "metrics": {},
            "success": True,
            "detail": "Plan C smoke test script not yet implemented",
        }

    if phase == "P5":
        return {
            "status": "skipped",
            "metrics": {},
            "success": True,
            "detail": "Plan D smoke test script not yet implemented",
        }

    if phase == "P6":
        return run_analysis_phase(output_dir)

    raise ValueError(f"Unknown phase: {phase}")


def run_analysis_phase(output_dir: Path) -> dict[str, Any]:
    """Gather all metrics.json files and write a summary report."""
    summary: dict[str, Any] = {"phases": {}, "recommendation": "", "timestamp": now_str()}
    for phase in PHASES[:-1]:
        metrics_path = output_dir / phase / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path, "r", encoding="utf-8") as f:
                summary["phases"][phase] = json.load(f)
        else:
            summary["phases"][phase] = {"error": "metrics.json missing"}

    p3 = summary["phases"].get("P3", {})
    p1 = summary["phases"].get("P1", {})
    p2 = summary["phases"].get("P2", {})

    p3_ok = p3.get("recovery_rate", 0.0) >= 0.5 and p3.get("clean_drop", 1.0) <= 0.15
    p1_drop = p1.get("AP50_clean", 1.0) - p1.get("AP50_adv", 1.0)
    p1_ok = p1_drop >= 0.10
    p2_ok = p2.get("recovery_rate", 0.0) >= 0.5 and p2.get("clean_drop", 1.0) <= 0.15

    if p3_ok:
        summary["recommendation"] = "Plan B verified on 20-image subset. Next: scale to full val set and NWPU/VisDrone."
    elif p2_ok and not p3_ok:
        summary["recommendation"] = "Plan B smoke OK but full experiment failed. Next: tune anomaly_threshold / lambda_step on full set."
    elif p1_ok and not p2_ok:
        summary["recommendation"] = "Attack works but defense too destructive. Next: reduce lambda_step, preserve more low frequencies."
    else:
        summary["recommendation"] = "Plan B not yet viable with current patch+defense. Next: integrate stronger defense (SAR/JPEG) or stronger attack."

    summary_path = output_dir / "SUMMARY_8H.md"
    write_summary_markdown(summary_path, summary)
    return {
        "status": "completed",
        "metrics": summary,
        "success": True,
        "detail": f"Summary written to {summary_path}",
    }


def write_summary_markdown(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# 8-Hour Long-Haul Experiment Summary\n",
        f"Generated: {summary['timestamp']}\n\n",
        "## Per-Phase Results\n",
    ]
    for phase, metrics in summary["phases"].items():
        lines.append(f"### {phase}\n")
        if "error" in metrics:
            lines.append(f"- error: {metrics['error']}\n")
        else:
            for k, v in metrics.items():
                if isinstance(v, (int, float, str)):
                    lines.append(f"- {k}: {v}\n")
        lines.append("\n")
    lines.append("## Recommendation\n")
    lines.append(summary["recommendation"] + "\n")
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "LONG_HAUL_STATE.json"
    log_path = output_dir / "LONG_HAUL_LOG.txt"

    skipped = {p.strip() for p in args.skip_plans.split(",") if p.strip()}

    if args.resume and state_path.exists():
        state = load_state(state_path)
    else:
        state = {"phase": "P1", "status": "pending", "history": [], "retries": {}, "started_at": now_str()}

    phase_index = PHASES.index(state["phase"])
    max_end = time.time() + args.max_hours * 3600

    append_log(log_path, f"Long-haul started (max_hours={args.max_hours}, skip={skipped})")

    while phase_index < len(PHASES):
        if time.time() >= max_end:
            append_log(log_path, "Time budget exhausted; stopping.")
            state["status"] = "timeout"
            save_state(state, state_path)
            break

        phase = PHASES[phase_index]

        if phase in skipped:
            append_log(log_path, f"{phase} skipped by user request")
            state["history"].append({"phase": phase, "status": "skipped", "timestamp": now_str()})
            phase_index += 1
            state["phase"] = PHASES[phase_index] if phase_index < len(PHASES) else "DONE"
            save_state(state, state_path)
            continue

        result = run_phase(phase, args, state)
        retries = state.setdefault("retries", {})
        retry = int(retries.get(phase, 0))

        state["history"].append({
            "phase": phase,
            "status": result["status"],
            "detail": result.get("detail", ""),
            "timestamp": now_str(),
        })
        append_log(log_path, f"{phase} retry={retry} {result['status']}: {result.get('detail', '')}")

        if result.get("status") == "error":
            append_log(log_path, f"{phase} subprocess error; stopping long-haul.")
            state["status"] = "error"
            state["phase"] = phase
            save_state(state, state_path)
            break

        if not result.get("success", False) and phase in ("P1", "P2"):
            if retry < MAX_RETRIES:
                retries[phase] = retry + 1
                state["phase"] = phase
                append_log(log_path, f"{phase} retrying with mutated parameters (retry {retry + 1}/{MAX_RETRIES})")
                # phase_index stays the same to retry current phase.
            else:
                append_log(log_path, f"{phase} exhausted {MAX_RETRIES} retries; moving on.")
                phase_index += 1
                state["phase"] = PHASES[phase_index] if phase_index < len(PHASES) else "DONE"
        else:
            phase_index += 1
            state["phase"] = PHASES[phase_index] if phase_index < len(PHASES) else "DONE"

        state["status"] = "running" if state["phase"] != "DONE" else "completed"
        save_state(state, state_path)

        if state["phase"] == "DONE":
            break

    append_log(log_path, f"Long-haul finished with status={state['status']}")
    print(f"Long-haul finished with status={state['status']}")


if __name__ == "__main__":
    main()
