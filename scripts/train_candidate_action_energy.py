"""Train a class-conditioned energy model over discrete local box actions."""

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
    concatenate_transition_batches,
    proposal_transition_tensors,
    summarize_proposal_transitions,
)
from spectral_detection_posttrain.eval.detection_metrics import evaluate_detection_predictions
from spectral_detection_posttrain.methods.energy_transport import (
    ActionBenefitEnergyHead,
    CandidateEnergyLossConfig,
    ROITransportActions,
    build_candidate_quality_targets,
    build_symmetric_box_candidates,
    candidate_action_energy_loss,
    select_min_energy_box_actions,
)
from spectral_detection_posttrain.models import build_detector
from spectral_detection_posttrain.trainers.detection.action_local_transport import (
    action_batch_to_predictions,
    extract_proposal_action_batch,
)
from spectral_detection_posttrain.utils.checkpoint_hash import sha256_file
from spectral_detection_posttrain.utils.git_state import get_git_state
from spectral_detection_posttrain.utils.io import ensure_run_dir, load_checkpoint, save_checkpoint, save_json
from spectral_detection_posttrain.utils.seed import resolve_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--detector-checkpoint", required=True)
    parser.add_argument("--model-name", default="fasterrcnn_mobilenet_v3_large_320_fpn")
    parser.add_argument("--nwpu-root", default="./data/NWPU VHR-10 dataset")
    parser.add_argument("--nwpu-annotation", default="./data/NWPU_VHR10_coco.json")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--step-sizes", default="0.05,0.1,0.2")
    parser.add_argument("--min-iou-gain", type=float, default=0.002)
    parser.add_argument("--temperature", type=float, default=0.10)
    parser.add_argument("--energy-weight", type=float, default=0.001)
    parser.add_argument("--min-energy-drop", type=float, default=0.0)
    parser.add_argument("--min-score", type=float, default=0.05)
    parser.add_argument("--require-foreground-dominant", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-actions-per-image", type=int, default=32)
    parser.add_argument("--min-oracle-ap75", type=float, default=0.36)
    parser.add_argument("--score-threshold", type=float, default=0.05)
    parser.add_argument("--nms-threshold", type=float, default=0.5)
    parser.add_argument("--detections-per-img", type=int, default=100)
    parser.add_argument("--shuffle-seed", type=int, default=31415)
    parser.add_argument("--require-clean-git", action="store_true", default=False)
    return parser.parse_args()


def parse_step_sizes(value: str) -> tuple[float, ...]:
    steps = tuple(sorted({float(item.strip()) for item in value.split(",") if item.strip()}))
    if not steps or any(step <= 0.0 for step in steps):
        raise ValueError("step sizes must be positive")
    return steps


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
        "train": {
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
        },
        "matching": {"iou_threshold": 0.5, "score_threshold": args.score_threshold},
        "eval": {"batch_size": int(args.batch_size), "high_conf_threshold": 0.7},
    }


def observable_action_mask(state, *, min_score: float, require_foreground: bool) -> torch.Tensor:
    mask = state.scores >= float(min_score)
    if require_foreground and state.logits is not None and state.logits.shape[1] > 1:
        probabilities = torch.softmax(state.logits, dim=-1)
        mask = mask & (probabilities[:, 1:].amax(dim=1) > probabilities[:, 0])
    return mask


def candidate_energies(
    head: ActionBenefitEnergyHead,
    state,
    candidates: torch.Tensor,
    *,
    rows: torch.Tensor | None = None,
) -> torch.Tensor:
    if state.logits is None:
        raise ValueError("candidate energy requires class logits")
    if rows is None:
        rows = torch.arange(state.batch_size, device=state.features.device)
    rows = rows.long()
    count = int(rows.numel())
    candidate_count = int(candidates.shape[0])
    if count == 0:
        return state.features.new_empty((0, candidate_count))
    candidate_values = candidates.to(device=state.features.device, dtype=state.features.dtype)
    features = state.features[rows][:, None, :].expand(count, candidate_count, -1).reshape(
        count * candidate_count, -1
    )
    logits = state.logits[rows][:, None, :].expand(count, candidate_count, -1).reshape(
        count * candidate_count, -1
    )
    labels = state.labels[rows][:, None].expand(count, candidate_count).reshape(-1)
    scores = state.scores[rows][:, None].expand(count, candidate_count).reshape(-1)
    deltas = candidate_values[None, :, :].expand(count, candidate_count, 4).reshape(-1, 4)
    return head(features, logits, labels, scores, deltas).reshape(count, candidate_count)


def full_candidate_energies(
    head: ActionBenefitEnergyHead,
    state,
    candidates: torch.Tensor,
    eligible: torch.Tensor,
) -> torch.Tensor:
    energies = state.scores.new_full((state.batch_size, candidates.shape[0]), 1.0)
    energies[:, 0] = 0.0
    rows = torch.nonzero(eligible, as_tuple=False).flatten()
    if rows.numel() > 0:
        energies[rows] = candidate_energies(head, state, candidates, rows=rows)
    return energies


def train_one_epoch(
    model: torch.nn.Module,
    energy_head: ActionBenefitEnergyHead,
    candidates: torch.Tensor,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_config: CandidateEnergyLossConfig,
    *,
    epoch: int,
    min_iou_gain: float,
    min_score: float,
    require_foreground: bool,
) -> dict[str, float]:
    model.eval()
    energy_head.train()
    totals: dict[str, float] = {}
    batches = 0
    for images, raw_targets in tqdm(loader, desc=f"candidate energy epoch {epoch}"):
        images = [image.to(device) for image in images]
        batch = extract_proposal_action_batch(
            model,
            images,
            _to_device(raw_targets, device),
            match_mode="class_agnostic",
            box_base="decoded",
        )
        eligible = observable_action_mask(
            batch.state,
            min_score=min_score,
            require_foreground=require_foreground,
        )
        rows = torch.nonzero(eligible, as_tuple=False).flatten()
        if rows.numel() == 0:
            continue
        with torch.no_grad():
            target = build_candidate_quality_targets(
                batch.state,
                candidates,
                matched_gt_boxes=batch.matched_gt_boxes,
                matched_gt_labels=batch.matched_gt_labels,
                image_sizes=batch.image_sizes,
                min_iou_gain=min_iou_gain,
            )
        energies = candidate_energies(energy_head, batch.state, candidates, rows=rows)
        losses = candidate_action_energy_loss(
            energies,
            candidate_quality=target.candidate_quality[rows],
            target_indices=target.target_indices[rows],
            scores=batch.state.scores[rows],
            config=loss_config,
        )
        optimizer.zero_grad(set_to_none=True)
        losses["loss_total"].backward()
        optimizer.step()
        batches += 1
        for key, value in losses.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach().item())
    return {key: value / max(1, batches) for key, value in totals.items()}


@torch.no_grad()
def evaluate_candidate_energy(
    model: torch.nn.Module,
    energy_head: ActionBenefitEnergyHead,
    candidates: torch.Tensor,
    loader,
    device: torch.device,
    *,
    min_iou_gain: float,
    min_energy_drop: float,
    min_score: float,
    require_foreground: bool,
    max_actions_per_image: int,
    score_threshold: float,
    nms_threshold: float,
    detections_per_img: int,
    shuffle_seed: int,
    desc: str,
) -> dict[str, Any]:
    model.eval()
    energy_head.eval()
    predictions = {name: [] for name in ("identity", "learned", "shuffled_energy", "oracle_budget", "oracle_unlimited")}
    targets_out: list[dict[str, torch.Tensor]] = []
    selected_rows: list[torch.Tensor] = []
    target_rows: list[torch.Tensor] = []
    eligible_rows: list[torch.Tensor] = []
    oracle_gain_rows: list[torch.Tensor] = []
    realized_gain_rows: list[torch.Tensor] = []
    selected_energy_drop_rows: list[torch.Tensor] = []
    transition_batches: list[dict[str, torch.Tensor]] = []
    generator = torch.Generator(device="cpu").manual_seed(int(shuffle_seed))

    for images, raw_targets in tqdm(loader, desc=desc):
        images_device = [image.to(device) for image in images]
        batch = extract_proposal_action_batch(
            model,
            images_device,
            _to_device(raw_targets, device),
            match_mode="class_agnostic",
            box_base="decoded",
        )
        target = build_candidate_quality_targets(
            batch.state,
            candidates,
            matched_gt_boxes=batch.matched_gt_boxes,
            matched_gt_labels=batch.matched_gt_labels,
            image_sizes=batch.image_sizes,
            min_iou_gain=min_iou_gain,
        )
        eligible = observable_action_mask(
            batch.state,
            min_score=min_score,
            require_foreground=require_foreground,
        )
        energies = full_candidate_energies(energy_head, batch.state, candidates, eligible)
        learned, selected, _ = select_min_energy_box_actions(
            batch.state,
            candidates,
            energies,
            min_energy_drop=min_energy_drop,
            min_score=min_score,
            require_foreground_dominant=require_foreground,
            max_actions_per_image=max_actions_per_image,
        )
        shuffled = _permute_energy_rows(
            energies,
            batch.state.image_indices,
            eligible,
            generator=generator,
        )
        shuffled_actions, _, _ = select_min_energy_box_actions(
            batch.state,
            candidates,
            shuffled,
            min_energy_drop=min_energy_drop,
            min_score=min_score,
            require_foreground_dominant=require_foreground,
            max_actions_per_image=max_actions_per_image,
        )
        oracle_energy = -target.candidate_quality
        oracle_budget, _, _ = select_min_energy_box_actions(
            batch.state,
            candidates,
            oracle_energy,
            min_energy_drop=min_iou_gain,
            min_score=min_score,
            require_foreground_dominant=require_foreground,
            max_actions_per_image=max_actions_per_image,
        )
        oracle_unlimited, _, _ = select_min_energy_box_actions(
            batch.state,
            candidates,
            oracle_energy,
            min_energy_drop=min_iou_gain,
            min_score=min_score,
            require_foreground_dominant=require_foreground,
            max_actions_per_image=None,
        )
        identity = ROITransportActions(
            feature_delta=torch.zeros_like(batch.state.features),
            score_delta=torch.zeros_like(batch.state.scores),
            box_delta=torch.zeros_like(batch.state.boxes),
            keep_logit=torch.zeros_like(batch.state.scores),
        )
        actions_by_mode = {
            "identity": identity,
            "learned": learned,
            "shuffled_energy": shuffled_actions,
            "oracle_budget": oracle_budget,
            "oracle_unlimited": oracle_unlimited,
        }
        for mode, actions in actions_by_mode.items():
            predictions[mode].extend(
                action_batch_to_predictions(
                    batch,
                    actions,
                    gate_actions=False,
                    score_threshold=score_threshold,
                    nms_threshold=nms_threshold,
                    detections_per_img=detections_per_img,
                    native_model=model,
                )
            )
        transition_batches.append(
            proposal_transition_tensors(
                batch.state,
                matched_gt_boxes=batch.matched_gt_boxes,
                matched_gt_labels=batch.matched_gt_labels,
                box_delta=learned.box_delta,
                image_sizes=batch.image_sizes,
            )
        )
        selected_rows.append(selected.detach().cpu())
        target_rows.append(target.target_indices.detach().cpu())
        eligible_rows.append(eligible.detach().cpu())
        oracle_gain_rows.append(target.oracle_gain.detach().cpu())
        selected_quality = target.candidate_quality.gather(1, selected[:, None]).squeeze(1)
        realized_gain_rows.append((selected_quality - target.base_iou).detach().cpu())
        selected_energy = energies.gather(1, selected[:, None]).squeeze(1)
        selected_energy_drop_rows.append((energies[:, 0] - selected_energy).detach().cpu())
        targets_out.extend(
            [
                {key: value.detach().cpu() if torch.is_tensor(value) else value for key, value in target_item.items()}
                for target_item in raw_targets
            ]
        )

    metric_kwargs = {
        "iou_threshold": 0.5,
        "score_threshold": score_threshold,
        "high_conf_threshold": 0.7,
        "per_class": True,
        "num_classes": 11,
        "per_size": True,
    }
    selected = torch.cat(selected_rows)
    target_indices = torch.cat(target_rows)
    eligible = torch.cat(eligible_rows).bool()
    oracle_gain = torch.cat(oracle_gain_rows)
    realized_gain = torch.cat(realized_gain_rows)
    selected_energy_drop = torch.cat(selected_energy_drop_rows)
    predicted_move = selected.ne(0) & eligible
    oracle_move = target_indices.ne(0) & eligible
    correct_action = selected.eq(target_indices) & eligible
    beneficial_move = predicted_move & realized_gain.gt(0.0)
    harmful_move = predicted_move & realized_gain.lt(0.0)
    return {
        "modes": {
            mode: evaluate_detection_predictions(outputs, targets_out, **metric_kwargs)
            for mode, outputs in predictions.items()
        },
        "selection": {
            "eligible": int(eligible.sum().item()),
            "predicted_moves": int(predicted_move.sum().item()),
            "oracle_moves": int(oracle_move.sum().item()),
            "top1_accuracy_eligible": float((selected[eligible] == target_indices[eligible]).float().mean().item()),
            "top1_accuracy_oracle_moves": float(correct_action[oracle_move].float().mean().item())
            if oracle_move.any()
            else None,
            "move_precision": float(oracle_move[predicted_move].float().mean().item())
            if predicted_move.any()
            else None,
            "exact_action_precision": float(correct_action[predicted_move].float().mean().item())
            if predicted_move.any()
            else None,
            "beneficial_move_rate": float(beneficial_move[predicted_move].float().mean().item())
            if predicted_move.any()
            else None,
            "harmful_move_rate": float(harmful_move[predicted_move].float().mean().item())
            if predicted_move.any()
            else None,
            "realized_gain_mean": float(realized_gain[predicted_move].mean().item())
            if predicted_move.any()
            else None,
            "oracle_gain_mean_eligible": float(oracle_gain[eligible].mean().item()),
            "energy_drop_mean": float(selected_energy_drop[predicted_move].mean().item())
            if predicted_move.any()
            else None,
        },
        "learned_transitions": summarize_proposal_transitions(
            concatenate_transition_batches(transition_batches),
            threshold=0.75,
        ),
    }


def _permute_energy_rows(
    energies: torch.Tensor,
    image_indices: torch.Tensor,
    eligible: torch.Tensor,
    *,
    generator: torch.Generator,
) -> torch.Tensor:
    if eligible.shape != image_indices.shape:
        raise ValueError("eligible must contain one value per proposal")
    output = energies.clone()
    for image_idx in torch.unique(image_indices.detach().cpu(), sorted=True).tolist():
        rows = torch.nonzero(
            (image_indices == int(image_idx)) & eligible,
            as_tuple=False,
        ).flatten()
        if rows.numel() <= 1:
            continue
        order = torch.randperm(int(rows.numel()), generator=generator, device="cpu").to(rows.device)
        output[rows] = energies[rows[order]]
    return output


def _to_device(targets: list[dict[str, Any]], device: torch.device) -> list[dict[str, Any]]:
    return [
        {key: value.to(device) if torch.is_tensor(value) else value for key, value in target.items()}
        for target in targets
    ]


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))
    git_state = get_git_state()
    if args.require_clean_git and git_state["dirty"]:
        raise RuntimeError(f"Git working tree is dirty at {git_state['commit']}")
    config = build_config(args)
    device = resolve_device(config)
    run_dir = ensure_run_dir(args.run_name)
    detector_path = Path(args.detector_checkpoint).resolve()
    detector_hash = sha256_file(detector_path)
    train_loader, val_loader = build_detection_loaders(
        config,
        limit_train=args.limit_train,
        limit_val=args.limit_val,
        batch_size=int(args.batch_size),
    )
    model = build_detector(config).to(device)
    load_checkpoint(model, detector_path, device)
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.eval()
    candidates = build_symmetric_box_candidates(parse_step_sizes(args.step_sizes)).to(device)
    feature_dim = int(model.roi_heads.box_predictor.cls_score.in_features)
    energy_head = ActionBenefitEnergyHead(
        feature_dim=feature_dim,
        num_classes=11,
        hidden_dim=int(args.hidden_dim),
    ).to(device)
    loss_config = CandidateEnergyLossConfig(
        temperature=float(args.temperature),
        energy_weight=float(args.energy_weight),
    )
    optimizer = torch.optim.AdamW(
        energy_head.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    eval_kwargs = {
        "min_iou_gain": float(args.min_iou_gain),
        "min_energy_drop": float(args.min_energy_drop),
        "min_score": float(args.min_score),
        "require_foreground": bool(args.require_foreground_dominant),
        "max_actions_per_image": int(args.max_actions_per_image),
        "score_threshold": float(args.score_threshold),
        "nms_threshold": float(args.nms_threshold),
        "detections_per_img": int(args.detections_per_img),
        "shuffle_seed": int(args.shuffle_seed),
    }
    initial = evaluate_candidate_energy(
        model,
        energy_head,
        candidates,
        val_loader,
        device,
        desc="initial candidate energy eval",
        **eval_kwargs,
    )
    oracle_ap75 = float(initial["modes"]["oracle_budget"]["ap75"])
    base_result = {
        "run_name": args.run_name,
        "config": {**config, "git_state": git_state, "args": vars(args)},
        "detector_checkpoint": {"path": str(detector_path), "sha256": detector_hash},
        "candidates": candidates.detach().cpu().tolist(),
        "candidate_count": int(candidates.shape[0]),
        "loss_config": loss_config.__dict__,
        "initial": initial,
    }
    if oracle_ap75 < float(args.min_oracle_ap75):
        result = {
            **base_result,
            "completed": True,
            "status": "no_go_oracle_bound",
            "oracle_ap75": oracle_ap75,
            "history": [],
        }
        save_json(result, run_dir / "eval_metrics.json")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    history: list[dict[str, Any]] = []
    best_ap75 = -1.0
    best_epoch = None
    for epoch in range(1, int(args.epochs) + 1):
        losses = train_one_epoch(
            model,
            energy_head,
            candidates,
            train_loader,
            optimizer,
            device,
            loss_config,
            epoch=epoch,
            min_iou_gain=float(args.min_iou_gain),
            min_score=float(args.min_score),
            require_foreground=bool(args.require_foreground_dominant),
        )
        evaluation = evaluate_candidate_energy(
            model,
            energy_head,
            candidates,
            val_loader,
            device,
            desc=f"candidate energy eval {epoch}",
            **eval_kwargs,
        )
        ap75 = float(evaluation["modes"]["learned"]["ap75"])
        history.append({"epoch": epoch, "train": losses, "evaluation": evaluation})
        print(
            f"epoch {epoch}: loss={losses['loss_total']:.4f} "
            f"learned_AP75={ap75:.4f} identity_AP75={evaluation['modes']['identity']['ap75']:.4f}"
        )
        save_checkpoint(energy_head, run_dir / "candidate_energy_last.pth", {"epoch": epoch})
        if ap75 > best_ap75:
            best_ap75 = ap75
            best_epoch = epoch
            save_checkpoint(
                energy_head,
                run_dir / "candidate_energy_best_ap75.pth",
                {"epoch": epoch, "ap75": ap75},
            )
    result = {
        **base_result,
        "completed": True,
        "status": "trained",
        "oracle_ap75": oracle_ap75,
        "history": history,
        "best_epoch": best_epoch,
        "best_ap75": best_ap75,
    }
    save_json(result, run_dir / "eval_metrics.json")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
