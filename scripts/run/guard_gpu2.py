"""Dry-run-first launch guard for physical GPU2 on the remote manifold host.

The guard enforces the locked execution rule: physical GPU2 only,
``memory.free > 8192 MiB`` after the new process reaches its peak, a clean
Git tree, and an experiment definition that is either active or marked for
exact frozen reproduction.  It queries, checks, prints, and (only when
``--execute`` is passed) launches.  It never kills, stops, reprioritizes, or
signals another process.

Usage:
    python scripts/run/guard_gpu2.py query [--nvidia-smi PATH]
    python scripts/run/guard_gpu2.py check --experiment ID [--json]
    python scripts/run/guard_gpu2.py dry-run --experiment ID [--execute] [--json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_REGISTRY = (
    ROOT / "spectral_detection_posttrain" / "configs" / "registry" / "experiments.json"
)
RESEARCH_LINES_REGISTRY = (
    ROOT / "spectral_detection_posttrain" / "configs" / "registry" / "research_lines.json"
)

GPU2_INDEX = 2
GPU2_FREE_MIB_THRESHOLD = 8192
CUDA_VISIBLE_DEVICES = "2"

# Paths that may legitimately be untracked (user-owned documents or
# generated scratch directories).  These are ignored by the clean-tree gate.
CLEAN_TREE_IGNORE_PREFIXES = (
    "runs/",
    "data/",
    ".agent_reports/",
    ".repochan/",
    "tmp/",
    "output/",
    ".omc/",
    "pipeline_state/",
)
EXPLICIT_UNTRACKED_ALLOWLIST = frozenset(
    {
        "docs/energy_transport_vs_fpn_sm_analysis.md",
    }
)


def _run(
    cmd: list[str],
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    check: bool = False,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=check,
    )


def query_gpu2(nvidia_smi: str = "nvidia-smi") -> dict[str, Any]:
    """Query physical GPU2 and return {index, name, memory_total_mib, memory_free_mib}.

    Raises ``RuntimeError`` if the query fails, the wrong GPU is returned, or
    the output cannot be parsed.
    """
    result = _run(
        [
            nvidia_smi,
            "--query-gpu=index,name,memory.total,memory.free",
            "--format=csv,noheader,nounits",
            "-i",
            str(GPU2_INDEX),
        ]
    )
    if result.returncode != 0:
        raise RuntimeError(f"nvidia-smi failed: {result.stderr.strip()}")
    line = result.stdout.strip()
    if not line:
        raise RuntimeError("nvidia-smi returned empty output")
    parts = [p.strip() for p in line.split(",")]
    if len(parts) != 4:
        raise RuntimeError(f"unexpected nvidia-smi output: {line!r}")
    try:
        index = int(parts[0])
    except ValueError as exc:
        raise RuntimeError(f"non-integer GPU index: {parts[0]!r}") from exc
    if index != GPU2_INDEX:
        raise RuntimeError(f"nvidia-smi returned GPU {index}, expected {GPU2_INDEX}")
    try:
        memory_total_mib = float(parts[2])
        memory_free_mib = float(parts[3])
    except ValueError as exc:
        raise RuntimeError(f"non-numeric memory values: {parts[2]!r}, {parts[3]!r}") from exc
    return {
        "index": index,
        "name": parts[1],
        "memory_total_mib": memory_total_mib,
        "memory_free_mib": memory_free_mib,
    }


def is_git_clean(root: Path = ROOT) -> tuple[bool, str]:
    """Return (clean, reason).  Ignores runtime directories and known user files."""
    diff = _run(["git", "diff", "--quiet"], cwd=root)
    if diff.returncode != 0:
        return False, "working tree has uncommitted changes"
    cached = _run(["git", "diff", "--cached", "--quiet"], cwd=root)
    if cached.returncode != 0:
        return False, "index has staged changes"
    status = _run(["git", "status", "--porcelain", "--untracked-files=all"], cwd=root)
    for line in status.stdout.splitlines():
        if not line:
            continue
        # porcelain format: XY path or XY path -> orig? for renames; ignore.
        path = line[3:].split(" -> ")[-1].strip()
        normalized = path.replace("\\", "/")
        if normalized in EXPLICIT_UNTRACKED_ALLOWLIST:
            continue
        if any(normalized.startswith(prefix) for prefix in CLEAN_TREE_IGNORE_PREFIXES):
            continue
        return False, f"untracked or modified file outside runtime dirs: {path}"
    return True, ""


def load_experiment(
    exp_id: str,
    experiments_path: Path | None = None,
    lines_path: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load experiment record and its research line; validate GPU2 policy."""
    if experiments_path is None:
        experiments_path = EXPERIMENTS_REGISTRY
    if lines_path is None:
        lines_path = RESEARCH_LINES_REGISTRY
    experiments = json.loads(experiments_path.read_text(encoding="utf-8"))
    lines = json.loads(lines_path.read_text(encoding="utf-8"))
    record = next(
        (r for r in experiments.get("records", []) if r.get("id") == exp_id), None
    )
    if record is None:
        raise ValueError(f"experiment {exp_id!r} not found in registry")
    line = next(
        (ln for ln in lines.get("lines", []) if ln.get("id") == record.get("research_line")),
        None,
    )
    if line is None:
        raise ValueError(f"research line {record.get('research_line')!r} not found")
    if record.get("gpu_policy") != "remote_gpu2_guarded":
        raise ValueError(
            f"experiment {exp_id!r} has gpu_policy {record.get('gpu_policy')!r}, "
            "expected remote_gpu2_guarded"
        )
    runnable = record.get("runnable", "")
    line_status = line.get("status", "")
    allowed = (
        (runnable == "active" and line_status == "active")
        or (runnable == "frozen_reproduction_only" and line_status == "frozen")
    )
    if not allowed:
        raise ValueError(
            f"experiment {exp_id!r} is runnable={runnable!r} with line status "
            f"{line_status!r}; only active/active or frozen_reproduction_only/frozen "
            "are allowed for GPU2 launch"
        )
    return record, line


def build_command(
    experiment: dict[str, Any],
    python_path: str,
    workspace_path: str,
) -> str:
    """Return the launch command string for an authorized experiment."""
    entrypoint = experiment.get("legacy_entrypoint", "")
    config_path = experiment.get("config_path", "")
    parts = [f"CUDA_VISIBLE_DEVICES={CUDA_VISIBLE_DEVICES}"]
    parts.append(f"PYTHONPATH={workspace_path}:$PYTHONPATH")
    parts.append(python_path)
    if entrypoint:
        parts.append(entrypoint)
    if config_path:
        parts.append(f"--config {config_path}")
    return " ".join(parts)


def command_hash(command: str) -> str:
    return hashlib.sha256(command.encode("utf-8")).hexdigest()


def record_launch(
    experiment_id: str,
    command: str,
    pid: int,
    manifest_path: str,
    record_dir: Path,
) -> Path:
    """Write a non-secret launch record.  The directory is local/runtime only."""
    record_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    filename = f"{experiment_id.replace('.', '_')}_{timestamp}_{pid}.json"
    record_path = record_dir / filename
    payload = {
        "experiment_id": experiment_id,
        "command": command,
        "command_hash": command_hash(command),
        "pid": pid,
        "manifest_path": manifest_path,
        "recorded_at_utc": timestamp,
    }
    record_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return record_path


def check_all(
    exp_id: str,
    nvidia_smi: str = "nvidia-smi",
    root: Path = ROOT,
) -> dict[str, Any]:
    """Run every gate and return a structured result dict."""
    result: dict[str, Any] = {
        "experiment_id": exp_id,
        "passed": False,
        "gpu2": {},
        "git_clean": {},
        "experiment": {},
        "reason": "",
    }
    try:
        experiment, line = load_experiment(exp_id)
        result["experiment"] = {
            "id": experiment["id"],
            "research_line": line["id"],
            "line_status": line["status"],
            "runnable": experiment["runnable"],
        }
    except ValueError as exc:
        result["reason"] = str(exc)
        return result

    clean, clean_reason = is_git_clean(root)
    result["git_clean"] = {"clean": clean, "reason": clean_reason}
    if not clean:
        result["reason"] = f"git not clean: {clean_reason}"
        return result

    try:
        gpu_info = query_gpu2(nvidia_smi)
    except RuntimeError as exc:
        result["gpu2"] = {"error": str(exc)}
        result["reason"] = f"GPU2 query failed: {exc}"
        return result

    result["gpu2"] = gpu_info
    if gpu_info["memory_free_mib"] <= GPU2_FREE_MIB_THRESHOLD:
        result["reason"] = (
            f"GPU2 free memory {gpu_info['memory_free_mib']:.0f} MiB is not greater "
            f"than the {GPU2_FREE_MIB_THRESHOLD} MiB threshold"
        )
        return result

    config_path = root / experiment.get("config_path", "")
    if experiment.get("config_path") and not config_path.exists():
        result["reason"] = f"config path does not exist: {config_path}"
        return result

    result["passed"] = True
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    query_parser = subparsers.add_parser("query", help="query GPU2 state")
    query_parser.add_argument("--nvidia-smi", default="nvidia-smi")

    check_parser = subparsers.add_parser("check", help="check all launch gates")
    check_parser.add_argument("--experiment", required=True)
    check_parser.add_argument("--json", action="store_true")
    check_parser.add_argument("--nvidia-smi", default="nvidia-smi")

    dry_parser = subparsers.add_parser("dry-run", help="print the launch command")
    dry_parser.add_argument("--experiment", required=True)
    dry_parser.add_argument("--execute", action="store_true")
    dry_parser.add_argument("--json", action="store_true")
    dry_parser.add_argument("--nvidia-smi", default="nvidia-smi")
    dry_parser.add_argument(
        "--record-dir",
        type=Path,
        default=ROOT / "runs" / "gpu2_launches",
    )
    dry_parser.add_argument("--manifest-path", default="")

    args = parser.parse_args(argv)

    if args.command == "query":
        try:
            info = query_gpu2(args.nvidia_smi)
        except RuntimeError as exc:
            print(f"GPU2 query failed: {exc}", file=sys.stderr)
            return 1
        print(json.dumps(info, indent=2, sort_keys=True))
        return 0

    if args.command == "check":
        result = check_all(args.experiment, args.nvidia_smi)
        if args.json:
            print(json.dumps(result, indent=2, sort_keys=True))
        else:
            if result["passed"]:
                print(f"GPU2 launch gates passed for {args.experiment}")
            else:
                print(f"GPU2 launch gates failed: {result['reason']}")
        return 0 if result["passed"] else 1

    if args.command == "dry-run":
        result = check_all(args.experiment, args.nvidia_smi)
        if not result["passed"]:
            if args.json:
                print(json.dumps(result, indent=2, sort_keys=True))
            else:
                print(f"GPU2 launch gates failed: {result['reason']}")
            return 1
        experiment, _ = load_experiment(args.experiment)
        python_path = os.environ.get(
            "MANIFOLD_REMOTE_PYTHON",
            "/home/ps/anaconda3/envs/RLimage/bin/python",
        )
        workspace_path = os.environ.get(
            "MANIFOLD_REMOTE_WORKSPACE",
            "/home/ps/lzz/manifold-detection-energy-transport",
        )
        command = build_command(experiment, python_path, workspace_path)
        if args.json:
            payload = {
                "experiment_id": args.experiment,
                "passed": True,
                "command": command,
                "command_hash": command_hash(command),
            }
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            print(command)
        if args.execute:
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
            env["PYTHONPATH"] = f"{workspace_path}:{env.get('PYTHONPATH', '')}"
            proc = subprocess.Popen(
                command.split(),
                cwd=ROOT,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            manifest_path = args.manifest_path or ""
            record_path = record_launch(
                args.experiment,
                command,
                proc.pid,
                manifest_path,
                args.record_dir,
            )
            if not args.json:
                print(f"launched pid={proc.pid} record={record_path}")
        return 0

    return 2


if __name__ == "__main__":
    sys.exit(main())
