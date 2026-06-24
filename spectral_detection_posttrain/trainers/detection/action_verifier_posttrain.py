from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from tqdm import tqdm

from spectral_detection_posttrain.datasets import build_penn_fudan_loaders
from spectral_detection_posttrain.eval.detection_metrics import evaluate_detection_predictions
from spectral_detection_posttrain.experiments.canonical_runner import (
    build_experiment_model,
    checkpoint_metadata,
    prepare_experiment_from_config,
)
from spectral_detection_posttrain.core.matching.box_iou import box_iou
from spectral_detection_posttrain.core.models.bbox_adapter import (
    freeze_bbox_adapter_only,
    freeze_selected_adapters,
    install_residual_bbox_adapter,
)
from spectral_detection_posttrain.core.models.build_detector import set_detector_eval_except_trainable
from spectral_detection_posttrain.methods.dpo.action_verifier import (
    ActionVerifierConfig,
    build_action_batch,
    build_dpo_pairs,
    build_rlvr_rewards,
    compute_fft_action_quality,
    compute_manifold_action_quality,
    dpo_loss_from_log_probs,
    normalize_group_advantage,
)
from spectral_detection_posttrain.methods.rlvr.roi_policy_loss import (
    baseline_kl_loss,
    extract_roi_head_outputs_for_boxes,
)
from spectral_detection_posttrain.utils.config import load_config
from spectral_detection_posttrain.utils.io import append_jsonl, save_checkpoint, save_json
from spectral_detection_posttrain.utils.seed import resolve_device, set_seed


OBJECTIVES = {"rlvr", "dpo"}
VERIFIERS = {"fft", "iou_oracle", "manifold"}


@dataclass(frozen=True)
class SafetyGuardResult:
    triggered: bool
    reason: str = ""
    observed: float = 0.0
    threshold: float = 0.0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Round 2.120-2.123 action-conditioned verifier post-training.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--objective", required=True, choices=sorted(OBJECTIVES))
    parser.add_argument("--verifier", required=True, choices=sorted(VERIFIERS))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument("--max-proposals", type=int, default=16)
    parser.add_argument("--num-samples", type=int, default=2)
    parser.add_argument("--sigma", type=float, default=0.1)
    parser.add_argument("--include-identity-action", action="store_true")
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--verifier-weight", type=float, default=0.1)
    parser.add_argument("--dpo-beta", type=float, default=0.5)
    parser.add_argument("--pair-margin", type=float, default=0.05)
    parser.add_argument("--policy-loss-weight", type=float, default=1e-4)
    parser.add_argument("--baseline-kl-weight", type=float, default=1.0)
    parser.add_argument("--det-loss-weight", type=float, default=1.0)
    parser.add_argument("--max-pred-multiplier", type=float, default=2.0)
    parser.add_argument("--max-fp-rate", type=float, default=0.6)
    parser.add_argument("--bbox-adapter", action="store_true")
    parser.add_argument("--bbox-adapter-hidden-dim", type=int, default=128)
    parser.add_argument("--bbox-adapter-scale", type=float, default=1.0)
    parser.add_argument("--bbox-adapter-delta-weight", type=float, default=0.0)
    parser.add_argument("--cls-adapter", action="store_true")
    parser.add_argument("--cls-adapter-scale", type=float, default=1.0)
    parser.add_argument("--cls-confidence-loss-weight", type=float, default=0.0)
    parser.add_argument("--cls-confidence-score-max", type=float, default=0.7)
    parser.add_argument("--cls-confidence-iou-min", type=float, default=0.75)
    parser.add_argument("--score-threshold", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args(argv)


def freeze_bbox_pred_only(model: torch.nn.Module) -> list[str]:
    for parameter in model.parameters():
        parameter.requires_grad = False
    try:
        bbox_pred = model.roi_heads.box_predictor.bbox_pred
    except AttributeError as exc:
        raise AttributeError("Model must expose roi_heads.box_predictor.bbox_pred") from exc

    trainable_names = []
    for name, parameter in bbox_pred.named_parameters():
        parameter.requires_grad = True
        trainable_names.append(f"roi_heads.box_predictor.bbox_pred.{name}")
    if not trainable_names:
        raise RuntimeError("No bbox_pred parameters were found to train.")
    return trainable_names


def compute_weighted_objective(
    policy_loss: torch.Tensor,
    kl_loss: torch.Tensor,
    det_loss: torch.Tensor,
    *,
    policy_loss_weight: float,
    baseline_kl_weight: float,
    det_loss_weight: float,
) -> torch.Tensor:
    return (
        float(policy_loss_weight) * policy_loss
        + float(baseline_kl_weight) * kl_loss
        + float(det_loss_weight) * det_loss
    )


def evaluate_safety_guard(
    metrics: dict,
    baseline_metrics: dict,
    *,
    max_pred_multiplier: float,
    max_fp_rate: float,
) -> SafetyGuardResult:
    baseline_predictions = float(baseline_metrics.get("num_predictions", 0.0) or 0.0)
    max_predictions = baseline_predictions * float(max_pred_multiplier)
    num_predictions = float(metrics.get("num_predictions", 0.0) or 0.0)
    if num_predictions > max_predictions:
        return SafetyGuardResult(
            triggered=True,
            reason="prediction_count_exceeded",
            observed=num_predictions,
            threshold=max_predictions,
        )

    false_positive_rate = float(metrics.get("false_positive_rate", 0.0) or 0.0)
    if false_positive_rate > float(max_fp_rate):
        return SafetyGuardResult(
            triggered=True,
            reason="false_positive_rate_exceeded",
            observed=false_positive_rate,
            threshold=float(max_fp_rate),
        )
    return SafetyGuardResult(triggered=False)


def should_save_best_checkpoint(metrics: dict, *, best_ap75: float, safety_guard: SafetyGuardResult) -> bool:
    if safety_guard.triggered:
        return False
    return float(metrics.get("ap75", 0.0) or 0.0) > float(best_ap75) + 1e-9


def bbox_adapter_delta_penalty(model: torch.nn.Module) -> torch.Tensor | None:
    predictor = getattr(model.roi_heads, "box_predictor", None)
    adapter = getattr(predictor, "bbox_adapter", None)
    if adapter is None:
        return None
    penalty = None
    for parameter in adapter.parameters():
        term = parameter.pow(2).mean()
        penalty = term if penalty is None else penalty + term
    return penalty


def confidence_correction_loss(
    class_logits: torch.Tensor,
    scores: torch.Tensor,
    best_iou: torch.Tensor,
    *,
    score_max: float,
    iou_min: float,
) -> torch.Tensor:
    if class_logits.numel() == 0:
        return class_logits.sum() * 0.0
    mask = (scores.to(class_logits.device) < float(score_max)) & (best_iou.to(class_logits.device) >= float(iou_min))
    if not mask.any():
        return class_logits.sum() * 0.0
    target = torch.ones(int(mask.sum().item()), dtype=torch.long, device=class_logits.device)
    return F.cross_entropy(class_logits[mask], target)


def _person_box_deltas(box_regression: torch.Tensor, num_classes: int) -> torch.Tensor:
    box_regression_4d = box_regression.reshape(box_regression.shape[0], num_classes, 4)
    class_index = 1 if num_classes > 1 else 0
    return box_regression_4d[:, class_index]


def _match_action_iou(decoded_boxes: torch.Tensor, gt_boxes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if decoded_boxes.numel() == 0:
        empty = decoded_boxes.new_empty(decoded_boxes.shape[:2])
        return empty, empty.bool()
    if gt_boxes.numel() == 0:
        zeros = decoded_boxes.new_zeros(decoded_boxes.shape[:2])
        return zeros, zeros.bool()
    flat = decoded_boxes.reshape(-1, 4)
    ious = box_iou(flat, gt_boxes.to(flat.device)).max(dim=1).values.reshape(decoded_boxes.shape[:2])
    return ious, ious >= 0.5


def _box_geometry_features(decoded_boxes: torch.Tensor, image_size: tuple[int, int]) -> torch.Tensor:
    height, width = image_size
    boxes = decoded_boxes.float()
    w = (boxes[..., 2] - boxes[..., 0]).clamp_min(0.0) / max(float(width), 1.0)
    h = (boxes[..., 3] - boxes[..., 1]).clamp_min(0.0) / max(float(height), 1.0)
    cx = ((boxes[..., 0] + boxes[..., 2]) * 0.5) / max(float(width), 1.0)
    cy = ((boxes[..., 1] + boxes[..., 3]) * 0.5) / max(float(height), 1.0)
    area = w * h
    aspect = torch.log((w + 1e-6) / (h + 1e-6))
    return torch.stack([cx, cy, w, h, area, aspect], dim=-1)


def _verifier_quality(verifier: str, image: torch.Tensor, decoded_boxes: torch.Tensor, reference_features: torch.Tensor) -> torch.Tensor:
    if verifier == "fft":
        return compute_fft_action_quality(image, decoded_boxes, crop_size=32)
    if verifier == "manifold":
        features = _box_geometry_features(decoded_boxes, tuple(image.shape[-2:]))
        return compute_manifold_action_quality(features, reference_features)
    raise ValueError(f"Unknown verifier: {verifier}")


def _reference_features_from_targets(targets: list[dict], device: torch.device) -> torch.Tensor:
    features = []
    for target in targets:
        boxes = target.get("boxes", torch.empty((0, 4))).to(device)
        if boxes.numel() == 0:
            continue
        image_id_features = torch.stack(
            [
                (boxes[:, 0] + boxes[:, 2]) * 0.5 / 320.0,
                (boxes[:, 1] + boxes[:, 3]) * 0.5 / 320.0,
                (boxes[:, 2] - boxes[:, 0]).clamp_min(0) / 320.0,
                (boxes[:, 3] - boxes[:, 1]).clamp_min(0) / 320.0,
                ((boxes[:, 2] - boxes[:, 0]).clamp_min(0) * (boxes[:, 3] - boxes[:, 1]).clamp_min(0)) / (320.0 * 320.0),
                torch.log(((boxes[:, 2] - boxes[:, 0]).clamp_min(1e-6)) / ((boxes[:, 3] - boxes[:, 1]).clamp_min(1e-6))),
            ],
            dim=1,
        )
        features.append(image_id_features)
    if not features:
        return torch.zeros((1, 6), device=device)
    return torch.cat(features, dim=0).float()


def _targets_to_device(targets: list[dict], device: torch.device) -> list[dict]:
    return [
        {key: value.to(device) if torch.is_tensor(value) else value for key, value in target.items()}
        for target in targets
    ]


def _set_detector_train_for_loss(model: torch.nn.Module) -> None:
    model.train()
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()


def build_action_verifier_loaders(
    config: dict,
    *,
    limit_train: int | None,
    limit_val: int | None,
):
    train_batch_size = int(config.get("posttrain", {}).get("batch_size", config.get("train", {}).get("batch_size", 1)))
    eval_batch_size = int(config.get("eval", {}).get("batch_size", train_batch_size))
    train_loader, _ = build_penn_fudan_loaders(
        config,
        limit_train=limit_train,
        limit_val=1,
        batch_size=train_batch_size,
    )
    _, val_loader = build_penn_fudan_loaders(
        config,
        limit_train=1,
        limit_val=limit_val,
        batch_size=eval_batch_size,
    )
    return train_loader, val_loader


@torch.no_grad()
def _evaluate(model: torch.nn.Module, loader, device: torch.device, config: dict) -> dict:
    model.eval()
    predictions = []
    targets = []
    for images, batch_targets in loader:
        outputs = model([image.to(device) for image in images])
        predictions.extend([{k: v.detach().cpu() for k, v in output.items()} for output in outputs])
        targets.extend([{k: v.detach().cpu() if torch.is_tensor(v) else v for k, v in target.items()} for target in batch_targets])
    return evaluate_detection_predictions(
        predictions,
        targets,
        iou_threshold=float(config["matching"].get("iou_threshold", 0.5)),
        score_threshold=float(config["matching"].get("score_threshold", 0.05)),
        high_conf_threshold=float(config["eval"].get("high_conf_threshold", 0.7)),
    )


def main_for_experiment(
    objective: str,
    verifier: str,
    argv: list[str] | None = None,
    default_run_name: str | None = None,
) -> None:
    args = parse_args(argv)
    if args.objective != objective or args.verifier != verifier:
        raise ValueError(f"Entrypoint expects objective={objective} verifier={verifier}")

    config = load_config(args.config)
    if default_run_name and args.run_name == "auto":
        args.run_name = default_run_name
    if args.seed is not None:
        config["seed"] = int(args.seed)
    set_seed(int(config.get("seed", 42)))
    context = prepare_experiment_from_config(
        config,
        args.config,
        args.run_name,
        phase=f"round2120_2123_{objective}_{verifier}",
        checkpoint_path=args.checkpoint,
    )
    config = context.config
    run_dir = context.run_dir
    save_json(
        {
            "objective": objective,
            "verifier": verifier,
            "epochs": args.epochs,
            "max_proposals": args.max_proposals,
            "num_samples": args.num_samples,
            "sigma": args.sigma,
            "include_identity_action": args.include_identity_action,
            "policy_loss_weight": args.policy_loss_weight,
            "baseline_kl_weight": args.baseline_kl_weight,
            "det_loss_weight": args.det_loss_weight,
            "max_pred_multiplier": args.max_pred_multiplier,
            "max_fp_rate": args.max_fp_rate,
            "bbox_adapter": args.bbox_adapter,
            "bbox_adapter_hidden_dim": args.bbox_adapter_hidden_dim,
            "bbox_adapter_scale": args.bbox_adapter_scale,
            "bbox_adapter_delta_weight": args.bbox_adapter_delta_weight,
            "cls_adapter": args.cls_adapter,
            "cls_adapter_scale": args.cls_adapter_scale,
            "cls_confidence_loss_weight": args.cls_confidence_loss_weight,
            "cls_confidence_score_max": args.cls_confidence_score_max,
            "cls_confidence_iou_min": args.cls_confidence_iou_min,
        },
        run_dir / "round_config.json",
    )
    if args.epochs <= 0:
        print({"run_name": args.run_name, "objective": objective, "verifier": verifier, "epochs": 0})
        return

    train_loader, val_loader = build_action_verifier_loaders(
        config,
        limit_train=args.limit_train,
        limit_val=args.limit_val,
    )
    device = resolve_device(config)
    model = build_experiment_model(context, checkpoint_path=args.checkpoint, device=device, pretrained=False)
    baseline_model = build_experiment_model(context, checkpoint_path=args.checkpoint, device=device, pretrained=False)
    baseline_model.eval()
    for parameter in baseline_model.parameters():
        parameter.requires_grad = False

    baseline_metrics = _evaluate(baseline_model, val_loader, device, config)
    save_json(baseline_metrics, run_dir / "baseline_eval_metrics.json")

    if args.bbox_adapter:
        install_residual_bbox_adapter(
            model,
            hidden_dim=int(args.bbox_adapter_hidden_dim),
            scale=float(args.bbox_adapter_scale),
            enable_cls_adapter=bool(args.cls_adapter),
            cls_scale=float(args.cls_adapter_scale),
        )
        if args.cls_adapter:
            trainable_names = freeze_selected_adapters(
                model,
                train_bbox_adapter=True,
                train_cls_adapter=True,
            )
        else:
            trainable_names = freeze_bbox_adapter_only(model)
    else:
        trainable_names = freeze_bbox_pred_only(model)
    save_json({"trainable_parameters": trainable_names}, run_dir / "trainable_parameters.json")
    set_detector_eval_except_trainable(model)
    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    if not trainable_parameters:
        raise RuntimeError("No trainable parameters.")
    optimizer = torch.optim.AdamW(trainable_parameters, lr=float(args.lr))
    num_classes = int(config["model"].get("num_classes", 2))
    action_cfg = ActionVerifierConfig(
        num_samples=int(args.num_samples),
        sigma=float(args.sigma),
        seed=int(config.get("seed", 42)),
        include_identity_action=bool(args.include_identity_action),
    )
    reference_features = _reference_features_from_targets([target for _, targets in train_loader for target in targets], device)
    best_ap75 = -1.0

    for epoch in range(1, int(args.epochs) + 1):
        set_detector_eval_except_trainable(model)
        total_loss = 0.0
        total_valid = 0
        total_actions = 0
        total_batches = 0
        total_policy_loss = 0.0
        total_kl_loss = 0.0
        total_det_loss = 0.0
        progress = tqdm(train_loader, desc=f"{args.run_name} e{epoch}/{args.epochs}")
        for images, targets in progress:
            device_images = [image.to(device) for image in images]
            with torch.no_grad():
                rollout_outputs = baseline_model(device_images)

            if float(args.det_loss_weight) > 0.0:
                _set_detector_train_for_loss(model)
                det_loss_dict = model(device_images, _targets_to_device(targets, device))
                det_loss = sum(det_loss_dict.values())
                set_detector_eval_except_trainable(model)
            else:
                det_loss = torch.tensor(0.0, device=device)

            policy_batch_loss = None
            kl_batch_loss = None
            confidence_batch_loss = None
            batch_valid = 0
            batch_actions = 0
            for image, target, rollout_output in zip(images, targets, rollout_outputs):
                keep = rollout_output["scores"].detach().cpu() >= float(args.score_threshold)
                proposals = rollout_output["boxes"].detach().cpu()[keep][: int(args.max_proposals)]
                proposal_scores = rollout_output["scores"].detach().cpu()[keep][: int(args.max_proposals)]
                if proposals.numel() == 0:
                    continue
                class_logits, box_regression, _, _ = extract_roi_head_outputs_for_boxes(model, [image.to(device)], [proposals])
                baseline_logits, baseline_box_regression, _, _ = extract_roi_head_outputs_for_boxes(
                    baseline_model, [image.to(device)], [proposals]
                )
                mu = _person_box_deltas(box_regression, num_classes)
                baseline_mu = _person_box_deltas(baseline_box_regression, num_classes)
                action_batch = build_action_batch(proposals.to(device), mu, tuple(image.shape[-2:]), action_cfg)
                baseline_actions = build_action_batch(proposals.to(device), baseline_mu, tuple(image.shape[-2:]), action_cfg)
                gt_boxes = target.get("boxes", torch.empty((0, 4))).to(device)
                iou, matched = _match_action_iou(action_batch.decoded_boxes, gt_boxes)
                proposal_iou, _ = _match_action_iou(proposals.to(device).unsqueeze(1), gt_boxes)
                if verifier == "iou_oracle":
                    verifier_quality = iou.detach()
                else:
                    verifier_quality = _verifier_quality(verifier, image, action_batch.decoded_boxes, reference_features)
                quality = iou + float(args.verifier_weight) * verifier_quality

                if objective == "rlvr":
                    rewards = build_rlvr_rewards(iou, verifier_quality, matched, verifier_weight=float(args.verifier_weight))
                    advantages = normalize_group_advantage(rewards)
                    policy_loss = -(advantages.detach() * action_batch.log_probs).mean()
                    valid_count = int(matched.sum().item())
                else:
                    pairs = build_dpo_pairs(quality, margin=float(args.pair_margin))
                    policy_loss = dpo_loss_from_log_probs(
                        action_batch.log_probs,
                        baseline_actions.log_probs.detach(),
                        pairs,
                        beta=float(args.dpo_beta),
                    )
                    valid_count = int(pairs.valid.sum().item())

                cls_kl_loss = baseline_kl_loss(class_logits, baseline_logits)
                bbox_keep_loss = F.smooth_l1_loss(box_regression, baseline_box_regression.to(box_regression.device))
                kl_loss = cls_kl_loss + bbox_keep_loss
                confidence_loss = confidence_correction_loss(
                    class_logits,
                    proposal_scores,
                    proposal_iou.squeeze(1),
                    score_max=float(args.cls_confidence_score_max),
                    iou_min=float(args.cls_confidence_iou_min),
                )
                policy_batch_loss = policy_loss if policy_batch_loss is None else policy_batch_loss + policy_loss
                kl_batch_loss = kl_loss if kl_batch_loss is None else kl_batch_loss + kl_loss
                confidence_batch_loss = (
                    confidence_loss
                    if confidence_batch_loss is None
                    else confidence_batch_loss + confidence_loss
                )
                batch_valid += valid_count
                batch_actions += int(action_batch.log_probs.numel())

            if policy_batch_loss is None:
                if float(args.det_loss_weight) <= 0.0:
                    continue
                policy_batch_loss = det_loss * 0.0
                kl_batch_loss = det_loss * 0.0
                confidence_batch_loss = det_loss * 0.0
            elif kl_batch_loss is None:
                kl_batch_loss = policy_batch_loss * 0.0
            if confidence_batch_loss is None:
                confidence_batch_loss = policy_batch_loss * 0.0

            batch_loss = compute_weighted_objective(
                policy_batch_loss,
                kl_batch_loss,
                det_loss,
                policy_loss_weight=float(args.policy_loss_weight),
                baseline_kl_weight=float(args.baseline_kl_weight),
                det_loss_weight=float(args.det_loss_weight),
            )
            if float(args.cls_confidence_loss_weight) > 0.0:
                batch_loss = batch_loss + float(args.cls_confidence_loss_weight) * confidence_batch_loss
            adapter_penalty = bbox_adapter_delta_penalty(model)
            if adapter_penalty is not None and float(args.bbox_adapter_delta_weight) > 0.0:
                batch_loss = batch_loss + float(args.bbox_adapter_delta_weight) * adapter_penalty
            if not torch.isfinite(batch_loss):
                save_json(
                    {
                        "epoch": epoch,
                        "reason": "non_finite_loss",
                        "loss": float(batch_loss.detach().cpu().item()),
                    },
                    run_dir / "safety_stop.json",
                )
                print({"epoch": epoch, "safety_triggered": True, "safety_reason": "non_finite_loss"})
                return
            if not batch_loss.requires_grad:
                continue
            optimizer.zero_grad(set_to_none=True)
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_parameters, max_norm=2.0)
            optimizer.step()

            total_loss += float(batch_loss.item())
            total_policy_loss += float(policy_batch_loss.item())
            total_kl_loss += float(kl_batch_loss.item())
            total_det_loss += float(det_loss.item())
            total_valid += batch_valid
            total_actions += batch_actions
            total_batches += 1
            progress.set_postfix(loss=total_loss / max(1, total_batches), valid=total_valid)

        metrics = _evaluate(model, val_loader, device, config)
        safety_guard = evaluate_safety_guard(
            metrics,
            baseline_metrics,
            max_pred_multiplier=float(args.max_pred_multiplier),
            max_fp_rate=float(args.max_fp_rate),
        )
        row = {
            "epoch": epoch,
            "loss": total_loss / max(1, total_batches),
            "policy_loss": total_policy_loss / max(1, total_batches),
            "kl_loss": total_kl_loss / max(1, total_batches),
            "det_loss": total_det_loss / max(1, total_batches),
            "valid_count": total_valid,
            "action_count": total_actions,
            "valid_rate": total_valid / max(1, total_actions),
            "ap50": metrics["ap50"],
            "ap75": metrics["ap75"],
            "ece": metrics.get("ece", 0.0),
            "num_predictions": metrics.get("num_predictions", 0),
            "false_positive_rate": metrics.get("false_positive_rate", 0.0),
            "safety_triggered": safety_guard.triggered,
            "safety_reason": safety_guard.reason,
        }
        append_jsonl(row, run_dir / "metrics_train.jsonl")
        save_json(metrics, run_dir / "eval_metrics.json")
        if safety_guard.triggered:
            save_json(
                {
                    "epoch": epoch,
                    "guard": asdict(safety_guard),
                    "metrics": metrics,
                    "baseline_metrics": baseline_metrics,
                },
                run_dir / "safety_stop.json",
            )
            print(row)
            print({"epoch": epoch, "safety_triggered": True, "safety_reason": safety_guard.reason})
            break

        save_checkpoint(model, run_dir / "checkpoint_last.pth", checkpoint_metadata(context, row))
        if should_save_best_checkpoint(metrics, best_ap75=best_ap75, safety_guard=safety_guard):
            best_ap75 = float(metrics["ap75"])
            save_checkpoint(model, run_dir / "checkpoint_best.pth", checkpoint_metadata(context, row))
        print(row)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    main_for_experiment(args.objective, args.verifier, argv=argv)


if __name__ == "__main__":
    main()
