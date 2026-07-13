"""Standard detection trainer adapter (plan Task 15).

Composes the maintained baseline training functions — the same ones the
``train_baseline.py`` CLI wires together — and returns a structured
:class:`TrainResult`:

- ``prepare_experiment_from_config`` / ``build_experiment_model`` /
  ``checkpoint_metadata`` (canonical runner)
- ``build_penn_fudan_loaders`` (datasets; default loader builder)
- ``train_one_epoch`` (``trainers.detection.train_baseline``)
- ``finalize_experiment`` / ``fail_experiment`` (artifact lifecycle)

The CLI remains the operational entry point. This adapter exists so callers
get structured results plus an explicit completed/failed runtime manifest;
like every trainer it returns data only and never decides research status.

The default loader builder is the Penn-Fudan baseline loader, matching
``train_baseline.py``. Tests inject a fake loader builder to run the adapter
without the dataset.
"""

from __future__ import annotations

from typing import Any, Callable

import torch

from spectral_detection_posttrain.datasets import build_penn_fudan_loaders
from spectral_detection_posttrain.experiments.canonical_runner import (
    build_experiment_model,
    checkpoint_metadata,
    prepare_experiment_from_config,
)
from spectral_detection_posttrain.experiments.lifecycle import (
    RUNTIME_MANIFEST_NAME,
    fail_experiment,
    finalize_experiment,
)
from spectral_detection_posttrain.trainers.detection.protocols import (
    TrainRequest,
    TrainResult,
    filter_scalar_metrics,
)
from spectral_detection_posttrain.trainers.detection.train_baseline import train_one_epoch
from spectral_detection_posttrain.utils.config import load_config, override_epochs
from spectral_detection_posttrain.utils.io import append_jsonl, save_checkpoint
from spectral_detection_posttrain.utils.seed import resolve_device, set_seed

LoaderBuilder = Callable[..., tuple[Any, Any]]
ModelBuilder = Callable[..., torch.nn.Module]


class StandardDetectionTrainer:
    """Protocol adapter over the maintained baseline training functions.

    ``loader_builder`` and ``model_builder`` are dependency-injection seams
    defaulting to the maintained functions; tests use them to run the adapter
    on tiny fake data without datasets or weight downloads.
    """

    def __init__(
        self,
        *,
        loader_builder: LoaderBuilder | None = None,
        model_builder: ModelBuilder | None = None,
    ) -> None:
        self._loader_builder = loader_builder or build_penn_fudan_loaders
        self._model_builder = model_builder or build_experiment_model

    def train(self, request: TrainRequest) -> TrainResult:
        config = load_config(request.config_path)
        override_epochs(config, "train", request.epochs)
        seed = request.seed if request.seed is not None else int(config.get("seed", 42))
        set_seed(seed)
        context = prepare_experiment_from_config(
            config,
            request.config_path,
            request.run_name,
            phase="train",
            runs_root=request.runs_root,
        )
        config = context.config
        run_dir = context.run_dir
        metrics_path = run_dir / "metrics_train.jsonl"
        checkpoint_path = run_dir / "checkpoint_last.pth"

        history: list[dict[str, Any]] = []
        try:
            epochs = int(config["train"].get("epochs", 1))
            if epochs < 1:
                raise ValueError(f"epochs must be >= 1, got {epochs}")
            train_loader, _ = self._loader_builder(
                config,
                limit_train=request.limit_train,
                limit_val=request.limit_val,
                batch_size=int(config["train"].get("batch_size", 2)),
            )
            device = resolve_device(config)
            model = self._model_builder(context, device=device)
            optimizer = torch.optim.SGD(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                lr=float(config["train"].get("lr", 0.005)),
                momentum=float(config["train"].get("momentum", 0.9)),
                weight_decay=float(config["train"].get("weight_decay", 0.0005)),
            )
            for epoch in range(1, epochs + 1):
                row = train_one_epoch(
                    model, train_loader, optimizer, device, request.run_name, epoch, epochs
                )
                row["epoch"] = epoch
                append_jsonl(row, metrics_path)
                save_checkpoint(
                    model, checkpoint_path, checkpoint_metadata(context, {"epoch": epoch})
                )
                history.append(row)
        except Exception as exc:
            fail_experiment(context, exc)
            raise

        final_row = history[-1]
        metrics = filter_scalar_metrics(final_row)
        manifest = finalize_experiment(
            context,
            outputs={"final_checkpoint": checkpoint_path, "train_metrics": metrics_path},
            metrics=metrics,
            gates={},
        )
        return TrainResult(
            checkpoints=tuple(
                ref for ref in manifest.outputs if "checkpoint" in ref.semantic_kind
            ),
            metrics=metrics,
            diagnostics={
                "run_dir": str(run_dir),
                "epochs": epochs,
                "history": history,
                "manifest_path": str(run_dir / RUNTIME_MANIFEST_NAME),
            },
        )
