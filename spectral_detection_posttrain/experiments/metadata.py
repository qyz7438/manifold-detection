from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping

import torch
import torchvision

from spectral_detection_posttrain.experiments.contracts import EvaluationScope


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


def evaluation_scope_provenance(config: Mapping[str, Any]) -> dict[str, Any] | None:
    """Extract the additive evaluation-scope provenance block from a config.

    Returns ``None`` when the config carries no ``evaluation_scope``. The
    ``formal`` marker comes from ``evaluation_scope_formal`` written by
    config normalization; it defaults to ``False`` (non-formal) when absent,
    so unmarked scopes can never masquerade as validated runs.
    """
    scope = config.get("evaluation_scope")
    if scope is None:
        return None
    if isinstance(scope, EvaluationScope):
        kind = scope.kind
        image_count = scope.image_count
        limit_train = scope.limit_train
        limit_val = scope.limit_val
    elif isinstance(scope, Mapping):
        kind = scope.get("kind")
        image_count = scope.get("image_count")
        limit_train = scope.get("limit_train")
        limit_val = scope.get("limit_val")
    else:
        raise ValueError(
            "evaluation_scope must be a mapping or EvaluationScope, "
            f"got {type(scope).__name__}"
        )
    return {
        "kind": kind,
        "image_count": image_count,
        "limit_train": limit_train,
        "limit_val": limit_val,
        "formal": bool(config.get("evaluation_scope_formal", False)),
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
    provenance = evaluation_scope_provenance(config)
    if provenance is not None:
        metadata["evaluation_scope"] = provenance
        metadata["normalization_warnings"] = list(config.get("normalization_warnings") or [])
    metadata.update(collect_git_metadata())
    if config_path is not None:
        metadata["config_path"] = str(config_path)
        metadata["config_file_hash"] = sha256_file(config_path)
    if checkpoint_path is not None:
        metadata["checkpoint_path"] = str(checkpoint_path)
        metadata["checkpoint_hash"] = sha256_file(checkpoint_path)
    return metadata
