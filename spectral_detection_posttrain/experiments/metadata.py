from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import torch
import torchvision


def sha256_file(path: str | Path) -> str:
    file_path = Path(path)
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_config(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _git(args: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    except Exception:
        return None
    return completed.stdout.strip()


def collect_git_metadata() -> dict[str, Any]:
    status = _git(["status", "--short"])
    status_hash = hashlib.sha256((status or "").encode("utf-8", errors="replace")).hexdigest()
    status_lines = (status or "").splitlines()
    return {
        "git_commit": _git(["rev-parse", "HEAD"]),
        "git_branch": _git(["rev-parse", "--abbrev-ref", "HEAD"]),
        "git_dirty": bool(status),
        "git_status_hash": status_hash,
        "git_status_entries": len(status_lines),
        "git_status_short": "\n".join(status_lines[:50]),
        "git_status_truncated": len(status_lines) > 50,
    }


def collect_experiment_metadata(
    config: dict[str, Any],
    config_path: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "config_hash": hash_config(config),
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
    }
    metadata.update(collect_git_metadata())
    if config_path is not None:
        metadata["config_path"] = str(config_path)
        metadata["config_file_hash"] = sha256_file(config_path)
    if checkpoint_path is not None:
        metadata["checkpoint_path"] = str(checkpoint_path)
        metadata["checkpoint_hash"] = sha256_file(checkpoint_path)
    return metadata
