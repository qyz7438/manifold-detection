from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from spectral_detection_posttrain.experiments.metadata import collect_experiment_metadata
from spectral_detection_posttrain.experiments.schema import validate_experiment_config
from spectral_detection_posttrain.core.models import build_detector
from spectral_detection_posttrain.utils.config import load_config, save_config
from spectral_detection_posttrain.utils.io import load_checkpoint, save_json


@dataclass(frozen=True)
class ExperimentContext:
    config: dict[str, Any]
    config_path: Path
    run_name: str
    run_dir: Path
    phase: str
    metadata: dict[str, Any]
    checkpoint_path: Path | None = None


def validate_checkpoint_path(path: str | Path | None, required: bool = True) -> Path | None:
    if path is None:
        if required:
            raise FileNotFoundError("Checkpoint path is required")
        return None
    checkpoint = Path(path)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
    if not checkpoint.is_file():
        raise ValueError(f"Checkpoint path is not a file: {checkpoint}")
    if checkpoint.stat().st_size <= 0:
        raise ValueError(f"Checkpoint file is empty: {checkpoint}")
    return checkpoint


def prepare_experiment(
    config_path: str | Path,
    run_name: str,
    phase: str,
    checkpoint_path: str | Path | None = None,
    runs_root: str | Path = "runs",
    formal: bool = True,
) -> ExperimentContext:
    config_file = Path(config_path)
    config = load_config(config_file)
    return prepare_experiment_from_config(
        config,
        config_path=config_file,
        run_name=run_name,
        phase=phase,
        checkpoint_path=checkpoint_path,
        runs_root=runs_root,
        formal=formal,
    )


def prepare_experiment_from_config(
    config: dict[str, Any],
    config_path: str | Path,
    run_name: str,
    phase: str,
    checkpoint_path: str | Path | None = None,
    runs_root: str | Path = "runs",
    formal: bool = True,
) -> ExperimentContext:
    config_file = Path(config_path)
    config = validate_experiment_config(config, formal=formal)
    checkpoint = validate_checkpoint_path(checkpoint_path, required=checkpoint_path is not None)

    run_dir = Path(runs_root) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, run_dir / "config.yaml")

    metadata = collect_experiment_metadata(config, config_path=config_file, checkpoint_path=checkpoint)
    metadata["phase"] = phase
    metadata["run_name"] = run_name
    save_json(metadata, run_dir / "metadata.json")

    return ExperimentContext(
        config=config,
        config_path=config_file,
        run_name=run_name,
        run_dir=run_dir,
        phase=phase,
        metadata=metadata,
        checkpoint_path=checkpoint,
    )


def build_experiment_model(
    context: ExperimentContext,
    checkpoint_path: str | Path | None = None,
    device: torch.device | None = None,
    pretrained: bool | None = None,
) -> torch.nn.Module:
    model_config = dict(context.config)
    model_config["model"] = dict(context.config["model"])
    if pretrained is not None:
        model_config["model"]["pretrained"] = pretrained
    model = build_detector(model_config)
    if device is not None:
        model = model.to(device)
    checkpoint = validate_checkpoint_path(checkpoint_path, required=False)
    if checkpoint is not None:
        load_checkpoint(model, checkpoint, device or torch.device("cpu"))
    return model


def checkpoint_metadata(context: ExperimentContext, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    metadata = dict(context.metadata)
    if extra:
        metadata.update(extra)
    return metadata


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a canonical experiment run directory.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--non-formal", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    context = prepare_experiment(
        config_path=args.config,
        run_name=args.run_name,
        phase=args.phase,
        checkpoint_path=args.checkpoint,
        runs_root=args.runs_root,
        formal=not args.non_formal,
    )
    print(context.run_dir)


if __name__ == "__main__":
    main()
