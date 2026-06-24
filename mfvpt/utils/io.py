from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


def ensure_run_dir(run_name: str) -> Path:
    run_dir = Path("runs") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_json(data: dict[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def append_jsonl(data: dict[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


def save_checkpoint(model: torch.nn.Module, path: str | Path, metadata: dict[str, Any] | None = None) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "metadata": metadata or {}}, output_path)


def load_checkpoint(model: torch.nn.Module, path: str | Path, device: torch.device) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=device)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    if isinstance(checkpoint, dict):
        return checkpoint.get("metadata", {})
    return {}
