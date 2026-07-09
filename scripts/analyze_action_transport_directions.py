"""Evaluate whether learned ROI box actions contain proposal-specific direction signal."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spectral_detection_posttrain.datasets import build_detection_loaders
from spectral_detection_posttrain.eval.action_transport_diagnostics import (
    box_only_actions,
    concatenate_transition_batches,
    permute_box_actions_within_images,
    proposal_transition_tensors,
    summarize_proposal_transitions,
)
from spectral_detection_posttrain.eval.detection_metrics import evaluate_detection_predictions
from spectral_detection_posttrain.methods.energy_transport import (
    ActionLocalTransportHead,
    ROITransportActions,
)
from spectral_detection_posttrain.models import build_detector
from spectral_detection_posttrain.trainers.detection.action_local_transport import (
    action_batch_to_predictions,
    encode_box_delta,
    extract_proposal_action_batch,
)
from spectral_detection_posttrain.utils.checkpoint_hash import sha256_file
from spectral_detection_posttrain.utils.git_state import get_git_state
from spectral_detection_posttrain.utils.io import ensure_run_dir, load_checkpoint, save_json
from spectral_detection_posttrain.utils.seed import resolve_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--detector-checkpoint", required=True)
    parser.add_argument(
        "--action-run",
        action="append",
        required=True,
        help="Repeat LABEL=RUN_DIR for each trained action head",
    )
    parser.add_argument("--model-name", default="fasterrcnn_mobilenet_v3_large_320_fpn")
    parser.add_argument("--nwpu-root", default="./data/NWPU VHR-10 dataset")
    parser.add_argument("--nwpu-annotation", default="./data/NWPU_VHR10_coco.json")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--scales", default="0,0.05,0.1,0.25,0.5,1.0")
    parser.add_argument("--permutation-seed", type=int, default=1701)
    parser.add_argument("--oracle-min-iou", type=float, default=0.5)
    parser.add_argument("--score-threshold", type=float, default=0.05)
    parser.add_argument("--nms-threshold", type=float, default=0.5)
    parser.add_argument("--detections-per-img", type=int, default=100)
    parser.add_argument("--require-clean-git", action="store_true", default=False)
    return parser.parse_args()


def parse_scales(value: str) -> list[float]:
    scales = sorted({float(item.strip()) for item in value.split(",") if item.strip()})
    if not scales or scales[0] < 0.0:
        raise ValueError("scales must contain non-negative values")
    if 0.0 not in scales or 1.0 not in scales:
        raise ValueError("scales must include 0 and 1")
    return scales


def parse_action_run(value: str) -> tuple[str, Path]:
    label, separator, path = value.partition("=")
    label = label.strip()
    path = path.strip()
    if not separator or not label or not path:
        raise ValueError("action run must use LABEL=RUN_DIR")
    return label, Path(path)


def build_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "seed": int(args.seed),
        "data_seed": int(args.data_seed),
        "device": "cuda" if args.device == "auto" else args.device,
        "data": {
            "dataset": "nwpu",
            "root": args.nwpu_root,
            "annotation": args.nwpu_annotation,
            "train_fraction": 0.7,
            "max_size": 480,
            "num_workers": int(args.num_workers),
        },
        "model": {
            "name": args.model_name,
            "model_name": args.model_name,
            "pretrained": False,
            "num_classes": 11,
            "min_size": 480,
            "max_size": 480,
        },
        "train": {"batch_size": int(args.batch_size), "lr": 0.0},
        "matching": {"iou_threshold": 0.5, "score_threshold": args.score_threshold},
        "eval": {"batch_size": int(args.batch_size), "high_conf_threshold": 0.7},
    }


def load_action_run(
    label: str,
    run_dir: Path,
    *,
    device: torch.device,
    detector_hash: str,
) -> tuple[ActionLocalTransportHead, dict[str, Any]]:
    metrics_path = run_dir / "eval_metrics.json"
    checkpoint_path = run_dir / "action_head_best_ap75.pth"
    if not metrics_path.exists() or not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing action metrics/checkpoint under {run_dir}")
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    source_hash = (payload.get("source_checkpoint") or {}).get("sha256")
    if source_hash != detector_hash:
        raise ValueError(
            f"{label} source checkpoint hash {source_hash!r} does not match detector {detector_hash}"
        )
    action_config = payload["action_config"]
    head = ActionLocalTransportHead(
        feature_dim=int(action_config["feature_dim"]),
        hidden_dim=action_config.get("hidden_dim"),
        max_score_delta=float(action_config["max_score_delta"]),
        max_box_delta=float(action_config["max_box_delta"]),
        residual_scale=0.0,
    ).to(device)
    load_checkpoint(head, checkpoint_path, device)
    head.eval()
    metadata = {
        "label": label,
        "run_dir": str(run_dir.resolve()),
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "best_epoch": payload.get("best_epoch"),
        "best_ap75": payload.get("best_ap75"),
        "action_config": action_config,
    }
    return head, metadata


def oracle_box_actions(
    batch,
    *,
    max_box_delta: float,
    min_iou: float,
    preserve_high_iou: bool,
) -> ROITransportActions:
    state = batch.state
    if state.ious is None or state.matched_gt_indices is None:
        raise ValueError("oracle diagnostics require matched proposal state")
    target = encode_box_delta(state.boxes, batch.matched_gt_boxes).clamp(
        -float(max_box_delta),
        float(max_box_delta),
    )
    valid = (
        state.matched_gt_indices.ge(0)
        & state.labels.eq(batch.matched_gt_labels)
        & state.ious.ge(float(min_iou))
    )
    if preserve_high_iou:
        valid = valid & state.ious.lt(0.75)
    target = torch.where(valid[:, None], target, torch.zeros_like(target))
    return ROITransportActions(
        feature_delta=torch.zeros_like(state.features),
        score_delta=torch.zeros_like(state.scores),
        box_delta=target,
        keep_logit=torch.zeros_like(state.scores),
    )


def _scale_name(scale: float) -> str:
    return f"{scale:g}".replace(".", "p")


def _mode_actions(
    learned: ROITransportActions,
    permuted: ROITransportActions,
    batch,
    *,
    scales: list[float],
    max_box_delta: float,
    oracle_min_iou: float,
) -> dict[str, ROITransportActions]:
    modes: dict[str, ROITransportActions] = {}
    for scale in scales:
        suffix = _scale_name(scale)
        modes[f"learned_s{suffix}"] = box_only_actions(learned, scale=scale)
        if scale > 0.0:
            modes[f"permuted_s{suffix}"] = box_only_actions(permuted, scale=scale)
    modes["oracle_local"] = oracle_box_actions(
        batch,
        max_box_delta=max_box_delta,
        min_iou=oracle_min_iou,
        preserve_high_iou=False,
    )
    modes["oracle_mid_preserve"] = oracle_box_actions(
        batch,
        max_box_delta=max_box_delta,
        min_iou=oracle_min_iou,
        preserve_high_iou=True,
    )
    return modes


@torch.no_grad()
def run_diagnostics(args: argparse.Namespace) -> dict[str, Any]:
    set_seed(int(args.seed))
    scales = parse_scales(args.scales)
    config = build_config(args)
    device = resolve_device(config)
    detector_path = Path(args.detector_checkpoint).resolve()
    detector_hash = sha256_file(detector_path)
    _, val_loader = build_detection_loaders(
        config,
        limit_val=args.limit_val,
        batch_size=int(args.batch_size),
    )
    model = build_detector(config).to(device)
    load_checkpoint(model, detector_path, device)
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.eval()

    heads: dict[str, ActionLocalTransportHead] = {}
    head_metadata: dict[str, dict[str, Any]] = {}
    for raw_spec in args.action_run:
        label, run_dir = parse_action_run(raw_spec)
        if label in heads:
            raise ValueError(f"Duplicate action label: {label}")
        head, metadata = load_action_run(
            label,
            run_dir,
            device=device,
            detector_hash=detector_hash,
        )
        heads[label] = head
        head_metadata[label] = metadata

    predictions: dict[str, dict[str, list[dict[str, torch.Tensor]]]] = {
        label: {} for label in heads
    }
    transition_batches: dict[str, dict[str, list[dict[str, torch.Tensor]]]] = {
        label: {} for label in heads
    }
    targets_out: list[dict[str, torch.Tensor]] = []
    generators = {
        label: torch.Generator(device="cpu").manual_seed(
            int(args.permutation_seed) + index * 1009
        )
        for index, label in enumerate(heads)
    }

    for images, targets in tqdm(val_loader, desc="action direction diagnostics"):
        images_device = [image.to(device) for image in images]
        targets_device = [
            {key: value.to(device) if torch.is_tensor(value) else value for key, value in target.items()}
            for target in targets
        ]
        batch = extract_proposal_action_batch(
            model,
            images_device,
            targets_device,
            match_mode="class_agnostic",
            box_base="decoded",
        )
        targets_out.extend(
            [
                {key: value.detach().cpu() if torch.is_tensor(value) else value for key, value in target.items()}
                for target in targets
            ]
        )

        for label, head in heads.items():
            learned = head(batch.state.features)
            permuted = permute_box_actions_within_images(
                learned,
                batch.state.image_indices,
                generator=generators[label],
            )
            max_box_delta = float(head_metadata[label]["action_config"]["max_box_delta"])
            modes = _mode_actions(
                learned,
                permuted,
                batch,
                scales=scales,
                max_box_delta=max_box_delta,
                oracle_min_iou=float(args.oracle_min_iou),
            )
            for mode, actions in modes.items():
                predictions[label].setdefault(mode, []).extend(
                    action_batch_to_predictions(
                        batch,
                        actions,
                        gate_actions=False,
                        score_threshold=float(args.score_threshold),
                        nms_threshold=float(args.nms_threshold),
                        detections_per_img=int(args.detections_per_img),
                        native_model=model,
                    )
                )
                transition_batches[label].setdefault(mode, []).append(
                    proposal_transition_tensors(
                        batch.state,
                        matched_gt_boxes=batch.matched_gt_boxes,
                        matched_gt_labels=batch.matched_gt_labels,
                        box_delta=actions.box_delta,
                        image_sizes=batch.image_sizes,
                    )
                )

    metric_kwargs = {
        "iou_threshold": 0.5,
        "score_threshold": float(args.score_threshold),
        "high_conf_threshold": 0.7,
        "per_class": True,
        "num_classes": 11,
        "per_size": True,
    }
    results: dict[str, Any] = {}
    for label in heads:
        mode_results: dict[str, Any] = {}
        for mode, outputs in predictions[label].items():
            metrics = evaluate_detection_predictions(outputs, targets_out, **metric_kwargs)
            transitions = concatenate_transition_batches(transition_batches[label][mode])
            mode_results[mode] = {
                "metrics": metrics,
                "proposal_transitions": summarize_proposal_transitions(
                    transitions,
                    threshold=0.75,
                ),
            }
        results[label] = {
            "metadata": head_metadata[label],
            "modes": mode_results,
            "decision": summarize_direction_decision(mode_results, scales=scales),
        }

    return {
        "run_name": args.run_name,
        "completed": True,
        "config": {
            **config,
            "git_state": get_git_state(),
            "detector_checkpoint": str(detector_path),
            "detector_checkpoint_sha256": detector_hash,
            "scales": scales,
            "permutation_seed": int(args.permutation_seed),
            "oracle_min_iou": float(args.oracle_min_iou),
            "limit_val": args.limit_val,
        },
        "results": results,
    }


def summarize_direction_decision(
    modes: dict[str, Any],
    *,
    scales: list[float],
) -> dict[str, Any]:
    zero_ap75 = float(modes["learned_s0"]["metrics"]["ap75"])
    candidates: list[dict[str, float | str]] = []
    for scale in scales:
        if scale <= 0.0:
            continue
        suffix = _scale_name(scale)
        learned = float(modes[f"learned_s{suffix}"]["metrics"]["ap75"])
        permuted = float(modes[f"permuted_s{suffix}"]["metrics"]["ap75"])
        candidates.append(
            {
                "scale": float(scale),
                "learned_ap75": learned,
                "permuted_ap75": permuted,
                "gain_vs_zero": learned - zero_ap75,
                "alignment_gap": learned - permuted,
            }
        )
    best = max(candidates, key=lambda row: float(row["learned_ap75"]))
    best_gap = max(candidates, key=lambda row: float(row["alignment_gap"]))
    oracle_mid = float(modes["oracle_mid_preserve"]["metrics"]["ap75"])
    has_signal = any(
        float(row["gain_vs_zero"]) >= 0.003
        and float(row["alignment_gap"]) >= 0.003
        for row in candidates
    )
    scale_one = next(row for row in candidates if float(row["scale"]) == 1.0)
    step_too_large = bool(
        float(best["scale"]) < 1.0
        and float(best["learned_ap75"]) - float(scale_one["learned_ap75"]) >= 0.01
        and float(best["alignment_gap"]) >= 0.003
    )
    return {
        "zero_ap75": zero_ap75,
        "scale_rows": candidates,
        "best_learned_scale": best,
        "best_alignment_gap": best_gap,
        "oracle_mid_ap75": oracle_mid,
        "oracle_mid_headroom": oracle_mid - zero_ap75,
        "directional_signal": has_signal,
        "step_too_large_candidate": step_too_large,
        "verdict": "directional_signal" if has_signal else "no_detectable_directional_signal",
    }


def main() -> None:
    args = parse_args()
    git_state = get_git_state()
    if args.require_clean_git and git_state["dirty"]:
        raise RuntimeError(f"Git working tree is dirty at {git_state['commit']}")
    result = run_diagnostics(args)
    run_dir = ensure_run_dir(args.run_name)
    save_json(result, run_dir / "diagnostics.json")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
