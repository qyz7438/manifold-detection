"""Action-local ROI transport trainer adapter (plan Task 15).

Composes the maintained action-transport primitives and returns a structured
:class:`TrainResult`:

- ``extract_proposal_action_batch`` / ``supervised_action_transport_loss`` /
  ``action_batch_to_predictions`` / ``SupervisedActionLossConfig``
  (``trainers.detection.action_local_transport``)
- ``ActionLocalTransportHead`` (``methods.energy_transport``)
- ``build_detection_loaders`` (datasets; default loader builder)
- ``evaluate_detection_predictions`` (eval)
- canonical-runner prepare plus lifecycle finalize/fail

The full-featured CLI ``scripts/train_energy_transport_action.py`` remains the
operational entry point (high-water-mark teacher, zero-action parity
diagnostics, per-class/per-size metrics). This adapter provides the minimal
protocol-shaped train/eval loop over the same package-level primitives and,
like every trainer, returns data only and never decides research status.

The config file is a standard experiment config (``model``/``data``/``train``/
``matching``/``eval`` sections) plus an optional ``action`` section::

    action:
      hidden_dim: 64            # ActionLocalTransportHead hidden dim
      max_score_delta: 0.05
      max_box_delta: 0.20
      residual_scale: 0.0
      match_mode: class_agnostic
      box_base: decoded
      action_score_threshold: 0.05
      nms_threshold: 0.50
      detections_per_img: 100
      loss: {}                  # SupervisedActionLossConfig field overrides
"""

from __future__ import annotations

from typing import Any, Callable

import torch

from spectral_detection_posttrain.datasets import build_detection_loaders
from spectral_detection_posttrain.eval.detection_metrics import evaluate_detection_predictions
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
from spectral_detection_posttrain.methods.energy_transport import ActionLocalTransportHead
from spectral_detection_posttrain.trainers.detection.action_local_transport import (
    SupervisedActionLossConfig,
    action_batch_to_predictions,
    extract_proposal_action_batch,
    supervised_action_transport_loss,
)
from spectral_detection_posttrain.trainers.detection.protocols import (
    TrainRequest,
    TrainResult,
    filter_scalar_metrics,
)
from spectral_detection_posttrain.utils.config import load_config, override_epochs
from spectral_detection_posttrain.utils.io import save_checkpoint, save_json
from spectral_detection_posttrain.utils.seed import resolve_device, set_seed

LoaderBuilder = Callable[..., tuple[Any, Any]]
ModelBuilder = Callable[..., torch.nn.Module]


def _freeze_detector(model: torch.nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.eval()


def _to_device(targets: list[dict[str, Any]], device: torch.device) -> list[dict[str, Any]]:
    return [
        {key: value.to(device) if torch.is_tensor(value) else value for key, value in target.items()}
        for target in targets
    ]


def _train_one_epoch(
    model: torch.nn.Module,
    action_head: ActionLocalTransportHead,
    loader: Any,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_config: SupervisedActionLossConfig,
    match_mode: str,
    box_base: str,
) -> dict[str, float]:
    """One action-head epoch over the maintained extract/loss primitives.

    Minimal mirror of ``scripts/train_energy_transport_action.train_one_epoch``
    without the high-water-mark teacher; the CLI keeps the full feature set.
    """
    model.eval()
    action_head.train()
    totals: dict[str, float] = {}
    seen_batches = 0
    for images, targets in loader:
        images = [image.to(device) for image in images]
        batch = extract_proposal_action_batch(
            model,
            images,
            _to_device(targets, device),
            match_mode=match_mode,
            box_base=box_base,
        )
        actions = action_head(batch.state.features)
        loss_dict = supervised_action_transport_loss(
            batch, actions, config=loss_config, teacher_actions=None
        )
        loss = loss_dict["loss_total"]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        seen_batches += 1
        for key, value in loss_dict.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach().item())
    return {key: value / max(1, seen_batches) for key, value in totals.items()}


@torch.no_grad()
def _evaluate_action_head(
    model: torch.nn.Module,
    action_head: ActionLocalTransportHead,
    loader: Any,
    device: torch.device,
    loss_config: SupervisedActionLossConfig,
    metric_kwargs: dict[str, Any],
    match_mode: str,
    box_base: str,
    action_score_threshold: float,
    nms_threshold: float,
    detections_per_img: int,
) -> dict[str, Any]:
    """Evaluate the action head with native detector postprocessing."""
    if box_base != "decoded":
        raise ValueError("native postprocessing requires box_base='decoded'")
    model.eval()
    action_head.eval()
    predictions: list[dict[str, torch.Tensor]] = []
    targets_out: list[dict[str, Any]] = []
    for images, targets in loader:
        images = [image.to(device) for image in images]
        batch = extract_proposal_action_batch(
            model,
            images,
            _to_device(targets, device),
            match_mode=match_mode,
            box_base=box_base,
        )
        actions = action_head(batch.state.features)
        predictions.extend(
            action_batch_to_predictions(
                batch,
                actions,
                gate_actions=loss_config.gate_actions,
                score_threshold=action_score_threshold,
                nms_threshold=nms_threshold,
                detections_per_img=detections_per_img,
                native_model=model,
            )
        )
        targets_out.extend(
            {key: value.detach().cpu() if torch.is_tensor(value) else value for key, value in target.items()}
            for target in targets
        )
    return evaluate_detection_predictions(predictions, targets_out, **metric_kwargs)


class ActionTransportTrainer:
    """Protocol adapter over the maintained action-transport primitives.

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
        self._loader_builder = loader_builder or build_detection_loaders
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
        action_config = dict(config.get("action") or {})
        checkpoint_path = run_dir / "action_head_last.pth"
        metrics_path = run_dir / "eval_metrics.json"

        history: list[dict[str, Any]] = []
        final_eval: dict[str, Any] = {}
        try:
            epochs = int(config["train"].get("epochs", 1))
            if epochs < 1:
                raise ValueError(f"epochs must be >= 1, got {epochs}")
            train_loader, val_loader = self._loader_builder(
                config,
                limit_train=request.limit_train,
                limit_val=request.limit_val,
                batch_size=int(config["train"].get("batch_size", 2)),
            )
            device = resolve_device(config)
            model = self._model_builder(context, device=device)
            _freeze_detector(model)
            feature_dim = int(model.roi_heads.box_predictor.cls_score.in_features)
            action_head = ActionLocalTransportHead(
                feature_dim=feature_dim,
                hidden_dim=action_config.get("hidden_dim"),
                max_score_delta=float(action_config.get("max_score_delta", 0.05)),
                max_box_delta=float(action_config.get("max_box_delta", 0.20)),
                residual_scale=float(action_config.get("residual_scale", 0.0)),
            ).to(device)
            loss_config = SupervisedActionLossConfig(**dict(action_config.get("loss") or {}))
            optimizer = torch.optim.AdamW(
                action_head.parameters(),
                lr=float(config["train"].get("lr", 1e-3)),
                weight_decay=float(config["train"].get("weight_decay", 1e-4)),
            )
            match_mode = str(action_config.get("match_mode", "class_agnostic"))
            box_base = str(action_config.get("box_base", "decoded"))
            matching_config = config.get("matching") or {}
            action_score_threshold = float(
                action_config.get("action_score_threshold", matching_config.get("score_threshold", 0.05))
            )
            nms_threshold = float(action_config.get("nms_threshold", 0.50))
            detections_per_img = int(action_config.get("detections_per_img", 100))
            metric_kwargs = {
                "iou_threshold": float(matching_config.get("iou_threshold", 0.5)),
                "score_threshold": float(matching_config.get("score_threshold", 0.05)),
                "high_conf_threshold": float((config.get("eval") or {}).get("high_conf_threshold", 0.7)),
                "per_class": False,
                "num_classes": int(config["model"]["num_classes"]),
                "per_size": False,
            }
            for epoch in range(1, epochs + 1):
                train_row = _train_one_epoch(
                    model,
                    action_head,
                    train_loader,
                    optimizer,
                    device,
                    loss_config,
                    match_mode,
                    box_base,
                )
                final_eval = _evaluate_action_head(
                    model,
                    action_head,
                    val_loader,
                    device,
                    loss_config,
                    metric_kwargs,
                    match_mode,
                    box_base,
                    action_score_threshold,
                    nms_threshold,
                    detections_per_img,
                )
                row = {
                    "epoch": epoch,
                    **train_row,
                    **{f"val_{key}": value for key, value in filter_scalar_metrics(final_eval).items()},
                }
                history.append(row)
                save_checkpoint(
                    action_head, checkpoint_path, checkpoint_metadata(context, {"epoch": epoch})
                )
        except Exception as exc:
            fail_experiment(context, exc)
            raise

        save_json(
            {"run_name": request.run_name, "history": history, "final_metrics": final_eval},
            metrics_path,
        )
        metrics = filter_scalar_metrics(final_eval)
        metrics["train_loss_final"] = float(history[-1].get("loss_total", 0.0))
        metrics["epochs"] = int(history[-1]["epoch"])
        manifest = finalize_experiment(
            context,
            outputs={"action_head_checkpoint": checkpoint_path, "eval_metrics": metrics_path},
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
                "epochs": int(history[-1]["epoch"]),
                "feature_dim": feature_dim,
                "history": history,
                "manifest_path": str(run_dir / RUNTIME_MANIFEST_NAME),
            },
        )
