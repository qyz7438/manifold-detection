from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from spectral_detection_posttrain.datasets import build_penn_fudan_loaders
from spectral_detection_posttrain.datasets.patch_transform import add_detection_patch
from spectral_detection_posttrain.eval.detection_metrics import evaluate_detection_predictions
from spectral_detection_posttrain.experiments.canonical_runner import (
    build_experiment_model,
    checkpoint_metadata,
    prepare_experiment_from_config,
)
from spectral_detection_posttrain.core.matching.pred_gt_matcher import match_predictions_to_gt
from spectral_detection_posttrain.core.models.build_detector import set_detector_eval_except_trainable, set_rlvr_trainable_params
from spectral_detection_posttrain.methods.rlvr.detection_verifier import (
    DetectionVerifierConfig,
    build_reward_component_summary,
    build_rewarded_roi_actions,
    signal_uses_amp,
    signal_uses_structure,
)
from spectral_detection_posttrain.methods.rlvr.roi_policy_loss import (
    baseline_kl_loss,
    extract_roi_head_outputs_for_boxes,
    roi_logit_max_abs_diff,
    signed_roi_policy_loss,
    weighted_fastrcnn_policy_loss,
)
from spectral_detection_posttrain.signals.fft.rlvr_reward import (
    compute_per_box_ramp,
    compute_per_box_structure,
    normalize_ramp,
)
from spectral_detection_posttrain.utils.config import load_config
from spectral_detection_posttrain.utils.io import append_jsonl, save_checkpoint, save_json
from spectral_detection_posttrain.utils.seed import resolve_device, set_seed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RLVR KL-stabilized signed policy post-training.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", "--baseline", required=True, dest="baseline")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--signal", required=True, choices=[
        "none", "ramp", "shuffled_ramp", "shuffled_amp",
        "structure", "shuffled_structure",
        "amp_structure", "shuffled_amp_structure",
    ])
    parser.add_argument("--unfreeze", required=True, choices=["cls", "box", "roi"])
    parser.add_argument("--optimizer", required=True, choices=["adamw", "sgd"])
    parser.add_argument("--reward-lambda", type=float, required=True)
    parser.add_argument("--struct-weight", type=float, default=0.0)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--beta", type=float, default=0.05)
    parser.add_argument("--policy-loss-weight", type=float, default=0.0)
    parser.add_argument("--box-loss-weight", type=float, default=0.0)
    parser.add_argument("--det-loss-weight", type=float, default=0.0)
    parser.add_argument("--baseline-kl-weight", type=float, default=1.0)
    parser.add_argument("--recovery-loss-weight", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--policy-objective", default="signed", choices=["signed", "weighted_ce"])
    parser.add_argument("--rollout-source", default="baseline", choices=["baseline", "current"])
    parser.add_argument("--r-amp-stats", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--max-candidates", type=int, default=40)
    parser.add_argument("--reward-score-threshold", type=float, default=0.2)
    parser.add_argument("--eval-patch-mode", default="none",
                        choices=["none", "clean", "random", "checkerboard", "object_edge", "object_inside", "near_object"])
    return parser.parse_args(argv)


def load_r_amp_stats_for_signal(signal: str, stats_path: str | None) -> dict | None:
    if not signal_uses_amp(signal):
        return None
    if not stats_path:
        raise ValueError(f"--r-amp-stats is required when --signal uses amplitude reward: {signal}")
    path = Path(stats_path)
    if not path.is_file():
        raise FileNotFoundError(f"R_amp stats file not found: {path}")
    stats = json.loads(path.read_text(encoding="utf-8"))
    missing = [key for key in ("p05", "p95", "count") if key not in stats]
    if missing:
        raise ValueError(f"R_amp stats missing required keys {missing}: {path}")
    if int(stats.get("count", 0)) <= 0:
        raise ValueError(f"R_amp stats has no samples: {path}")
    if float(stats["p95"]) <= float(stats["p05"]):
        raise ValueError(f"R_amp stats has invalid percentile range: {path}")
    return stats


def validate_policy_objective_args(policy_objective: str, box_loss_weight: float) -> None:
    if policy_objective == "weighted_ce" and box_loss_weight != 0:
        raise ValueError(
            "weighted_ce with box_loss_weight requires encoded Fast R-CNN bbox targets; "
            "use --box-loss-weight 0 or the signed objective."
        )


@torch.no_grad()
def _eval_ap50(model, val_loader, device, config, patch_mode="none") -> float:
    model.eval()
    patch_cfg = config.get("patch", {})
    predictions = []
    targets = []
    patch_type = "checkerboard" if "checkerboard" in patch_mode else "random"
    for images, batch_targets in val_loader:
        if patch_mode not in ("none", "clean"):
            pm = "random"
            if patch_mode in ("object_edge", "object_inside", "near_object"):
                pm = patch_mode
            images = [
                add_detection_patch(img, tgt, placement=pm, patch_type=patch_type,
                                    patch_size=int(patch_cfg.get("patch_size", 48)))
                for img, tgt in zip(images, batch_targets)
            ]
        outputs = model([img.to(device) for img in images])
        predictions.extend([{k: v.detach().cpu() for k, v in output.items()} for output in outputs])
        targets.extend([{k: v.detach().cpu() if torch.is_tensor(v) else v for k, v in tgt.items()}
                        for tgt in batch_targets])
    metrics = evaluate_detection_predictions(
        predictions, targets,
        iou_threshold=float(config["matching"].get("iou_threshold", 0.5)),
        score_threshold=float(config["matching"].get("score_threshold", 0.05)),
        high_conf_threshold=float(config["eval"].get("high_conf_threshold", 0.7)),
    )
    return float(metrics["ap50"])


def main() -> None:
    args = parse_args()
    validate_policy_objective_args(args.policy_objective, args.box_loss_weight)
    config = load_config(args.config)
    rlvr_cfg = config.setdefault("rlvr", {})
    run_seed = int(args.seed if args.seed is not None else config.get("seed", 42))
    config["seed"] = run_seed
    config.setdefault("rlvr", {})["epochs"] = args.epochs
    set_seed(run_seed)
    context = prepare_experiment_from_config(
        config,
        args.config,
        args.run_name,
        phase="rlvr",
        checkpoint_path=args.baseline,
    )
    config = context.config
    run_dir = context.run_dir
    rlvr_cfg = config.setdefault("rlvr", {})

    train_loader, val_loader = build_penn_fudan_loaders(
        config, limit_train=args.limit_train, limit_val=args.limit_val,
        batch_size=int(rlvr_cfg.get("batch_size", 1)),
    )
    device = resolve_device(config)

    model = build_experiment_model(context, checkpoint_path=args.baseline, device=device, pretrained=False)
    set_rlvr_trainable_params(model, mode=args.unfreeze)
    set_detector_eval_except_trainable(model)

    # frozen baseline for rollout + KL
    baseline_model = build_experiment_model(context, checkpoint_path=args.baseline, device=device, pretrained=False)
    baseline_model.eval()
    for p in baseline_model.parameters():
        p.requires_grad = False

    r_amp_stats = load_r_amp_stats_for_signal(args.signal, args.r_amp_stats)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters.")
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(trainable_params, lr=float(rlvr_cfg.get("lr_adamw", 0.0005)))
    else:
        optimizer = torch.optim.SGD(
            trainable_params, lr=float(rlvr_cfg.get("lr_sgd", 0.005)),
            momentum=float(config["train"].get("momentum", 0.9)),
            weight_decay=float(config["train"].get("weight_decay", 0.0005)),
        )

    matching_cfg = config["matching"]
    iou_threshold = float(matching_cfg.get("iou_threshold", 0.5))
    score_threshold = float(matching_cfg.get("score_threshold", 0.05))
    high_conf_threshold = float(config["eval"].get("high_conf_threshold", 0.7))
    amp_bins = int(config.get("quality_head", {}).get("amp_bins", 32))

    verifier_cfg = DetectionVerifierConfig(
        signal=args.signal, temperature=args.temperature,
        w_iou=1.0, w_cls=0.2, w_amp=args.reward_lambda, w_struct=args.struct_weight,
        w_hconf_fp=float(args.alpha), high_conf_threshold=high_conf_threshold,
    )

    # no-update fast path: skip training entirely, just save checkpoint
    is_no_update = (
        args.det_loss_weight == 0
        and args.policy_loss_weight == 0
        and args.baseline_kl_weight == 0
        and args.box_loss_weight == 0
    )

    # initial KL sanity: compute ROI logit parity before any training
    set_detector_eval_except_trainable(model)
    baseline_model.eval()
    sample_img = next(iter(train_loader))[0][0].to(device)
    dev_img = [sample_img]  # TorchVision expects list of 3D tensors [C, H, W]
    with torch.no_grad():
        preds = baseline_model(dev_img)
        boxes_for_sanity = preds[0]["boxes"][:20].detach().cpu()
        cur_logits, _, _, _ = extract_roi_head_outputs_for_boxes(model, dev_img, [boxes_for_sanity])
        bl_logits, _, _, _ = extract_roi_head_outputs_for_boxes(baseline_model, dev_img, [boxes_for_sanity])
    initial_kl = float(baseline_kl_loss(cur_logits, bl_logits).item())
    initial_max_diff = roi_logit_max_abs_diff(cur_logits, bl_logits)
    save_json(
        {"initial_roi_kl": initial_kl, "initial_logit_max_abs_diff": initial_max_diff,
         "rollout_source": args.rollout_source},
        run_dir / "initial_sanity.json",
    )

    if is_no_update:
        save_checkpoint(model, run_dir / "checkpoint_best.pth",
                        {"epoch": 0, "run_name": args.run_name, "val_ap50": None})
        save_checkpoint(model, run_dir / "checkpoint_last.pth",
                        {"epoch": 0, "run_name": args.run_name, "val_ap50": None})
        result = {"best_epoch": 0, "best_val_ap50": None, "total_epochs": 0,
                  "no_update": True, "initial_roi_kl": initial_kl,
                  "initial_logit_max_abs_diff": initial_max_diff,
                  "signal": args.signal, "unfreeze": args.unfreeze, "optimizer": args.optimizer,
                  "reward_lambda": args.reward_lambda, "policy_loss_weight": args.policy_loss_weight,
                  "baseline_kl_weight": args.baseline_kl_weight, "det_loss_weight": args.det_loss_weight}
        save_json(result, run_dir / "rlvr_result.json")
        print(result)
        return

    if args.rollout_source == "baseline" and args.det_loss_weight == 0:
        if initial_kl > 1e-4 or initial_max_diff > 1e-3:
            raise RuntimeError(
                f"Initial baseline/current ROI logits differ: "
                f"kl={initial_kl:.6f} max_abs={initial_max_diff:.6f}"
            )

    best_ap50 = -1.0
    best_epoch = 0
    stale_epochs = 0

    for epoch in range(1, args.epochs + 1):
        set_detector_eval_except_trainable(model)
        total_loss_det = 0.0
        total_loss_policy = 0.0
        total_loss_kl = 0.0
        total_loss = 0.0
        total_seen = 0

        progress = tqdm(train_loader, desc=f"{args.run_name} epoch {epoch}/{args.epochs}")
        for images, targets in progress:
            device_images = [img.to(device) for img in images]

            # Phase 1: optional supervised detection loss (needs full train mode)
            if args.det_loss_weight > 0:
                model.train()
                device_targets = [{k: v.to(device) if torch.is_tensor(v) else v for k, v in t.items()} for t in targets]
                loss_dict = model(device_images, device_targets)
                loss_det = sum(v for v in loss_dict.values())
                set_detector_eval_except_trainable(model)
            else:
                loss_det = torch.tensor(0.0, device=device)

            # Phase 2: generate rollouts from frozen baseline (or current)
            rollout_model = baseline_model if args.rollout_source == "baseline" else model
            rollout_model.eval()
            with torch.no_grad():
                predictions = rollout_model(device_images)

            # Phase 3: R_amp + build actions
            set_detector_eval_except_trainable(model)
            s_amp_list: list[torch.Tensor] = []
            s_struct_list: list[torch.Tensor] = []
            for image, prediction, target in zip(images, predictions, targets):
                pred_cpu = {k: v.detach().cpu() if torch.is_tensor(v) else v for k, v in prediction.items()}
                tgt_cpu = {k: v.detach().cpu() if torch.is_tensor(v) else v for k, v in target.items()}
                gt_boxes = tgt_cpu.get("boxes", torch.empty((0, 4)))
                pred_boxes = pred_cpu.get("boxes", torch.empty((0, 4)))
                matched_ramp = match_predictions_to_gt(pred_cpu, tgt_cpu, iou_threshold=iou_threshold, score_threshold=score_threshold)
                best_gt_indices = torch.full((len(pred_boxes),), -1, dtype=torch.long)
                for m in matched_ramp["matches"]:
                    best_gt_indices[m["pred_index"]] = m["gt_index"]
                if signal_uses_amp(args.signal) and r_amp_stats is not None and len(pred_boxes) > 0:
                    raw_ramp = compute_per_box_ramp(image, pred_boxes, gt_boxes, best_gt_indices, amp_bins=amp_bins)
                    norm_ramp = normalize_ramp(raw_ramp, r_amp_stats)
                    s_amp_list.append(norm_ramp)
                else:
                    s_amp_list.append(torch.zeros(len(pred_boxes)))
                if signal_uses_structure(args.signal) and len(pred_boxes) > 0:
                    s_struct_list.append(compute_per_box_structure(image, pred_boxes, gt_boxes, best_gt_indices))
                else:
                    s_struct_list.append(torch.zeros(len(pred_boxes)))

            actions = [
                build_rewarded_roi_actions(
                    pred, tgt, num_classes=int(config["model"]["num_classes"]),
                    verifier_cfg=verifier_cfg, max_candidates=args.max_candidates,
                    reward_score_threshold=args.reward_score_threshold, s_amp=s_amp, s_struct=s_struct,
                )
                for pred, tgt, s_amp, s_struct in zip(predictions, targets, s_amp_list, s_struct_list)
            ]

            proposal_boxes = [a["boxes"] for a in actions]
            policy_labels = torch.cat([a["policy_labels"] for a in actions], dim=0).to(device)
            advantages = torch.cat([a["advantages"] for a in actions], dim=0).to(device)
            supervised_labels = torch.cat([a["labels"] for a in actions], dim=0).to(device)
            ce_weights = torch.cat([a["weights"] for a in actions], dim=0).to(device)

            # Phase 4: differentiable ROI outputs from current model + baseline
            class_logits, box_regression, scaled_boxes, transformed_image_sizes = \
                extract_roi_head_outputs_for_boxes(model, device_images, proposal_boxes)

            with torch.no_grad():
                baseline_logits, _, _, _ = extract_roi_head_outputs_for_boxes(
                    baseline_model, device_images, proposal_boxes,
                )

            # Phase 5: policy + KL loss
            if args.policy_objective == "signed":
                loss_policy = signed_roi_policy_loss(class_logits, policy_labels, advantages)
            else:
                regression_targets_zero = torch.zeros((supervised_labels.shape[0], 4), device=device)
                pl = weighted_fastrcnn_policy_loss(
                    class_logits, box_regression, supervised_labels, regression_targets_zero, ce_weights,
                    box_loss_weight=args.box_loss_weight,
                )
                loss_policy = pl["loss_roi_policy_cls"] + pl["loss_roi_policy_box"]

            loss_kl = baseline_kl_loss(class_logits, baseline_logits)

            loss = (
                args.det_loss_weight * loss_det
                + args.policy_loss_weight * loss_policy
                + args.baseline_kl_weight * loss_kl
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss_det += float(loss_det.item()) * len(images)
            total_loss_policy += float(loss_policy.item()) * len(images)
            total_loss_kl += float(loss_kl.item()) * len(images)
            total_loss += float(loss.item()) * len(images)
            total_seen += len(images)

            # diagnostics
            candidate_count = sum(len(a["boxes"]) for a in actions) / max(1, len(actions))
            matched_count = sum((a["matched"]).sum().item() for a in actions) / max(1, len(actions))
            fp_count = candidate_count - matched_count
            person_rate = float((policy_labels > 0).float().mean().item())
            progress.set_postfix(
                loss_policy=total_loss_policy / max(1, total_seen),
                loss_kl=total_loss_kl / max(1, total_seen),
            )

        avg_loss = total_loss / max(1, total_seen)
        val_ap50 = _eval_ap50(model, val_loader, device, config, patch_mode=args.eval_patch_mode)
        print(f"epoch {epoch}: loss={avg_loss:.4f} val_ap50={val_ap50:.4f}")

        row = {
            "epoch": epoch,
            "loss": avg_loss,
            "loss_det": total_loss_det / max(1, total_seen),
            "loss_policy": total_loss_policy / max(1, total_seen),
            "loss_kl": total_loss_kl / max(1, total_seen),
            "val_ap50": val_ap50,
            "candidate_count": candidate_count,
            "matched_tp_count": matched_count,
            "fp_count": fp_count,
            "advantage_mean": float(advantages.mean().item()),
            "advantage_std": float(advantages.std(unbiased=False).item()),
            "policy_label_person_rate": person_rate,
        }
        append_jsonl(row, run_dir / "metrics_train.jsonl")
        save_checkpoint(model, run_dir / "checkpoint_last.pth",
                        checkpoint_metadata(context, {"epoch": epoch, "val_ap50": val_ap50}))

        if val_ap50 > best_ap50 + 1e-6:
            best_ap50 = val_ap50
            best_epoch = epoch
            stale_epochs = 0
            save_checkpoint(model, run_dir / "checkpoint_best.pth",
                            checkpoint_metadata(context, {"epoch": epoch, "val_ap50": val_ap50}))
        else:
            stale_epochs += 1
        if stale_epochs >= args.early_stopping_patience:
            print(f"Early stopping at epoch {epoch}, best_ap50={best_ap50:.4f} at epoch {best_epoch}")
            break

    result = {"best_epoch": best_epoch, "best_val_ap50": best_ap50, "total_epochs": epoch,
              "signal": args.signal, "unfreeze": args.unfreeze, "optimizer": args.optimizer,
              "reward_lambda": args.reward_lambda, "policy_loss_weight": args.policy_loss_weight,
              "temperature": args.temperature, "policy_objective": args.policy_objective,
              "baseline_kl_weight": args.baseline_kl_weight, "det_loss_weight": args.det_loss_weight}
    save_json(result, run_dir / "rlvr_result.json")
    print(result)


if __name__ == "__main__":
    main()
