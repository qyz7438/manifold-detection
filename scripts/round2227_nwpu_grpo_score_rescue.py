"""Round 2.227: NWPU GRPO score-rescue policy (Phase 1).

Train a small score-adjustment policy on top of the frozen NWPU baseline.
The policy sees ROI box features and predicts an additive score delta for each
low-confidence candidate.  Reward is structured to reward raising scores of
high-IoU boxes and lowering scores of low-IoU boxes, with group-relative
baseline (GRPO) and a KL anchor to the initial policy.

This is a minimal end-to-end validation of the RLVR direction after Round 2.226
showed that oracle additive re-ranking can lift AP75 by ~5%.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.ops import box_iou
from torchvision.transforms import functional as TF
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spectral_detection_posttrain.core.models.build_detector import build_detector
from spectral_detection_posttrain.eval.detection_metrics import evaluate_detection_predictions
from spectral_detection_posttrain.methods.rlvr.roi_policy_loss import extract_roi_head_outputs_for_boxes, resize_boxes_to_image
from spectral_detection_posttrain.utils.io import ensure_run_dir, save_checkpoint, save_json
from spectral_detection_posttrain.utils.seed import set_seed

GIT = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()

DATA = Path("data/NWPU VHR-10 dataset")
ANNOT = Path("data/NWPU_VHR10_coco.json")
CHECKPOINT = Path("runs/round2100_nwpu_baseline/checkpoint_best.pth")
NUM_CLASSES = 11
MAX_SIZE = 480


class NWPUDataset(Dataset):
    def __init__(self, root: Path, coco_json: Path, img_ids: set[int], max_size: int):
        self.root = Path(root)
        self.max_size = int(max_size)
        self.coco = json.loads(Path(coco_json).read_text(encoding="utf-8"))
        self.img_infos = {img["id"]: img for img in self.coco["images"] if img["id"] in img_ids}
        self.img_ids = list(self.img_infos.keys())
        anns: dict[int, list[dict]] = {}
        for ann in self.coco["annotations"]:
            if ann["image_id"] in img_ids:
                anns.setdefault(ann["image_id"], []).append(ann)
        self.anns = anns

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        info = self.img_infos[img_id]
        img_path = self.root / "positive image set" / info["file_name"]
        if not img_path.exists():
            img_path = self.root / "negative image set" / info["file_name"]
        image = Image.open(str(img_path)).convert("RGB")
        image_t = TF.to_tensor(image)
        boxes, labels = [], []
        for ann in self.anns.get(img_id, []):
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])
            labels.append(ann["category_id"])
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([img_id]),
        }
        _, height, width = image_t.shape
        if max(height, width) > self.max_size:
            scale = self.max_size / float(max(height, width))
            new_h, new_w = int(height * scale), int(width * scale)
            image_t = F.interpolate(
                image_t.unsqueeze(0),
                size=(new_h, new_w),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            target["boxes"] = target["boxes"] * scale
        return image_t, target


def collate(batch):
    return tuple(zip(*batch))


def split_ids() -> tuple[set[int], set[int]]:
    coco = json.loads(ANNOT.read_text(encoding="utf-8"))
    all_ids = list(
        set(
            img["id"]
            for img in coco["images"]
            if (DATA / "positive image set" / img["file_name"]).exists()
        )
    )
    np.random.seed(42)
    np.random.shuffle(all_ids)
    n_train = int(0.7 * len(all_ids))
    return set(all_ids[:n_train]), set(all_ids[n_train:])


def limited_ids(ids: set[int], limit: int | None, seed: int = 42) -> set[int]:
    if limit is None or limit >= len(ids):
        return set(ids)
    rng = np.random.default_rng(seed)
    selected = rng.choice(sorted(ids), size=int(limit), replace=False)
    return set(int(x) for x in selected)


def build_loaders(limit_train: int | None, limit_val: int | None, batch_size: int = 1, seed: int = 42):
    train_ids, val_ids = split_ids()
    train_ids = limited_ids(train_ids, limit_train, seed=seed)
    val_ids = limited_ids(val_ids, limit_val, seed=seed)
    train_ds = NWPUDataset(DATA, ANNOT, train_ids, MAX_SIZE)
    val_ds = NWPUDataset(DATA, ANNOT, val_ids, MAX_SIZE)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate, num_workers=0)
    return train_loader, val_loader


def build_nwpu_model(device: torch.device, checkpoint_path: Path = CHECKPOINT):
    model = build_detector(
        {
            "model": {
                "name": "fasterrcnn_mobilenet_v3_large_320_fpn",
                "model_name": "fasterrcnn_mobilenet_v3_large_320_fpn",
                "pretrained": False,
                "num_classes": NUM_CLASSES,
                "min_size": MAX_SIZE,
                "max_size": MAX_SIZE,
            }
        }
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    return model


def configure_detector_for_all_proposals(model, score_threshold: float = 0.0, nms_threshold: float = 1.0, detections_per_img: int = 1000):
    roi_heads = getattr(model, "roi_heads", None)
    if roi_heads is None:
        return {}
    previous = {}
    for attr, value in [("score_thresh", score_threshold), ("nms_thresh", nms_threshold), ("detections_per_img", detections_per_img)]:
        if hasattr(roi_heads, attr):
            previous[attr] = getattr(roi_heads, attr)
            setattr(roi_heads, attr, value)
    return previous


def restore_detector_config(model, previous: dict[str, object]) -> None:
    roi_heads = getattr(model, "roi_heads", None)
    if roi_heads is None:
        return
    for name, value in previous.items():
        setattr(roi_heads, name, value)


def class_wise_nms(boxes: torch.Tensor, scores: torch.Tensor, labels: torch.Tensor, iou_threshold: float = 0.5):
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long)
    boxes = boxes.to(scores.device)
    labels = labels.to(scores.device)
    keep_indices = []
    for cls in torch.unique(labels).tolist():
        mask = labels == cls
        idx = torch.nonzero(mask, as_tuple=False).flatten().to(boxes.device)
        cls_keep = torchvision.ops.nms(boxes[idx], scores[idx], iou_threshold)
        keep_indices.append(idx[cls_keep])
    return torch.cat(keep_indices) if keep_indices else torch.empty((0,), dtype=torch.long)


class ScoreAdjustmentPolicy(nn.Module):
    """Gaussian policy: outputs mean and log-std for additive score delta."""

    def __init__(self, in_features: int, hidden_dim: int = 128, init_std: float = 0.05, max_delta: float = 0.3):
        super().__init__()
        self.max_delta = float(max_delta)
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden_dim, 1)
        self.log_std = nn.Parameter(torch.full((1,), np.log(init_std)))

    def forward(self, box_features: torch.Tensor):
        h = self.net(box_features)
        mean = torch.tanh(self.mean_head(h).squeeze(-1)) * self.max_delta
        std = self.log_std.exp().expand_as(mean).clamp_min(1e-4)
        return torch.distributions.Normal(mean, std)

    def sample(self, box_features: torch.Tensor):
        dist = self.forward(box_features)
        delta = dist.rsample()
        return delta, dist

    def mean_delta(self, box_features: torch.Tensor):
        return self.forward(box_features).mean


def build_policy(model, hidden_dim: int = 128, init_std: float = 0.05, max_delta: float = 0.3) -> ScoreAdjustmentPolicy:
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    return ScoreAdjustmentPolicy(in_features, hidden_dim, init_std, max_delta).to(next(model.parameters()).device)


@torch.no_grad()
def extract_roi_features_and_logits(model, images: list[torch.Tensor], boxes: list[torch.Tensor]):
    """Return class_logits, box_regression, box_features, scaled_boxes, image_sizes."""
    from collections import OrderedDict
    original_sizes = [tuple(img.shape[-2:]) for img in images]
    transformed, _ = model.transform(images, None)
    features = model.backbone(transformed.tensors)
    if isinstance(features, torch.Tensor):
        features = OrderedDict([("0", features)])
    scaled_boxes = [
        resize_boxes_to_image(b.to(transformed.tensors.device), original, new)
        for b, original, new in zip(boxes, original_sizes, transformed.image_sizes)
    ]
    roi_features = model.roi_heads.box_roi_pool(features, scaled_boxes, transformed.image_sizes)
    box_features = model.roi_heads.box_head(roi_features)
    class_logits, box_regression = model.roi_heads.box_predictor(box_features)
    return class_logits, box_regression, box_features, scaled_boxes, transformed.image_sizes


@torch.no_grad()
def extract_proposals_with_features(model, image: torch.Tensor, gt_boxes: torch.Tensor, device: torch.device):
    """Extract all refined proposals, their box features, and IoU with GT."""
    model.eval()
    prev = configure_detector_for_all_proposals(model, score_threshold=0.0, nms_threshold=1.0, detections_per_img=1000)
    try:
        output = model([image.to(device)])[0]
    finally:
        restore_detector_config(model, prev)

    boxes = output["boxes"].detach()
    scores = output["scores"].detach()
    labels = output["labels"].detach()

    if boxes.numel() == 0:
        return None

    class_logits, box_regression, box_features, scaled_boxes, _ = extract_roi_features_and_logits(
        model, [image.to(device)], [boxes.to(device)]
    )
    box_features = box_features.detach()
    class_logits = class_logits.detach()

    ious = box_iou(boxes, gt_boxes.to(boxes.device)).max(dim=1).values if gt_boxes.numel() > 0 else torch.zeros_like(scores)

    return {
        "boxes": boxes,
        "scores": scores,
        "labels": labels,
        "box_features": box_features,
        "class_logits": class_logits,
        "ious": ious.cpu(),
    }


def compute_grpo_loss(policy, baseline_policy, proposals, cfg: dict):
    """Compute GRPO loss for one image's proposals.

    Returns policy loss, diagnostics dict.
    """
    scores = proposals["scores"].to(proposals["box_features"].device)
    ious = proposals["ious"].to(proposals["box_features"].device)
    features = proposals["box_features"]

    low_conf_mask = scores < float(cfg["low_conf_max"])
    if low_conf_mask.sum().item() == 0:
        zero = policy.mean_head.weight.sum() * 0.0
        return zero, {"active": 0, "mean_reward": 0.0, "mean_advantage": 0.0}

    scores_lc = scores[low_conf_mask].to(features.device)
    ious_lc = ious[low_conf_mask].to(features.device)
    features_lc = features[low_conf_mask]

    K = int(cfg["num_samples"])
    dist = policy.forward(features_lc)
    baseline_dist = baseline_policy.forward(features_lc)

    # Sample K deltas per candidate
    delta = dist.rsample((K,))  # [K, N]
    log_prob = dist.log_prob(delta).sum(dim=0)  # [N] (sum over K? no, need per-sample)

    # Recompute log_prob per sample for policy gradient
    delta_flat = delta.reshape(-1)  # [K*N]
    features_rep = features_lc.unsqueeze(0).expand(K, -1, -1).reshape(K * features_lc.shape[0], -1)
    scores_rep = scores_lc.unsqueeze(0).expand(K, -1).reshape(-1)
    ious_rep = ious_lc.unsqueeze(0).expand(K, -1).reshape(-1)

    dist_rep = policy.forward(features_rep)
    baseline_dist_rep = baseline_policy.forward(features_rep)
    log_prob_rep = dist_rep.log_prob(delta_flat)
    baseline_log_prob_rep = baseline_dist_rep.log_prob(delta_flat)

    # Reward: pull delta toward +max_delta for high-IoU boxes and toward
    # -max_delta for low-IoU boxes.  This avoids the deadlock where delta=0
    # yields zero reward and zero gradient.
    sign = torch.where(ious_rep > 0.5, torch.ones_like(ious_rep), -torch.ones_like(ious_rep))
    target_delta = sign * float(cfg["max_delta"])
    action_reward = -((delta_flat - target_delta) ** 2) * (2.0 * ious_rep - 1.0).abs()
    oracle_bonus = float(cfg["oracle_shaping"]) * ious_rep * (1.0 - scores_rep)
    reward = action_reward + oracle_bonus

    # GRPO advantage: group mean over all (candidate, sample) in this image
    mean_reward = reward.mean()
    std_reward = reward.std().clamp_min(1e-6)
    advantage = (reward - mean_reward) / std_reward
    advantage = advantage.clamp(-float(cfg["max_advantage"]), float(cfg["max_advantage"]))

    policy_loss = -(log_prob_rep * advantage.detach()).mean()

    # KL anchor to baseline policy
    kl_loss = torch.distributions.kl_divergence(dist_rep, baseline_dist_rep).mean()

    total_loss = policy_loss + float(cfg["kl_weight"]) * kl_loss

    diag = {
        "active": int(features_lc.shape[0]),
        "mean_reward": float(mean_reward.item()),
        "mean_advantage": float(advantage.mean().item()),
        "mean_delta": float(delta_flat.mean().item()),
        "std_delta": float(delta_flat.std().item()),
        "policy_loss": float(policy_loss.item()),
        "kl_loss": float(kl_loss.item()),
    }
    return total_loss, diag


@torch.no_grad()
def evaluate_policy(model, policy, val_loader, device, cfg: dict):
    model.eval()
    policy.eval()
    predictions, targets = [], []
    for images, batch_targets in val_loader:
        image = images[0]
        target = batch_targets[0]
        props = extract_proposals_with_features(model, image, target["boxes"], device)
        if props is None or props["boxes"].numel() == 0:
            predictions.append({"boxes": torch.empty((0, 4)), "scores": torch.empty((0,)), "labels": torch.empty((0,), dtype=torch.long)})
            targets.append(target)
            continue

        scores = props["scores"].to(device)
        box_features = props["box_features"]
        delta = torch.zeros_like(scores)
        low_conf_mask = scores < float(cfg["low_conf_max"])
        if low_conf_mask.any():
            delta[low_conf_mask] = policy.mean_delta(box_features[low_conf_mask])

        new_scores = (scores + delta).clamp(0.0, 1.0).cpu()
        keep = class_wise_nms(props["boxes"], new_scores, props["labels"], float(cfg["nms_iou"]))
        predictions.append({
            "boxes": props["boxes"][keep],
            "scores": new_scores[keep],
            "labels": props["labels"][keep],
        })
        targets.append(target)

    return evaluate_detection_predictions(predictions, targets, iou_threshold=0.5, score_threshold=float(cfg["score_threshold"]))


def train(model, policy, baseline_policy, train_loader, val_loader, device, cfg: dict, run_dir: Path):
    optimizer = torch.optim.AdamW(policy.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]))

    best_ap75 = -1.0
    best_epoch = -1
    history = []

    for epoch in range(int(cfg["epochs"])):
        policy.train()
        model.eval()
        epoch_loss = 0.0
        epoch_diag = {"active": 0, "mean_reward": 0.0, "mean_advantage": 0.0, "mean_delta": 0.0, "std_delta": 0.0}

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg['epochs']}")
        for images, batch_targets in pbar:
            image = images[0]
            target = batch_targets[0]
            proposals = extract_proposals_with_features(model, image, target["boxes"], device)
            if proposals is None:
                continue

            loss, diag = compute_grpo_loss(policy, baseline_policy, proposals, cfg)
            if loss.numel() == 0 or not torch.isfinite(loss):
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), float(cfg["grad_clip"]))
            optimizer.step()

            epoch_loss += float(loss.item())
            for k in epoch_diag:
                if k in diag:
                    epoch_diag[k] += diag[k]
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        n_batches = len(train_loader)
        for k in epoch_diag:
            epoch_diag[k] /= max(1, n_batches)

        val_metrics = evaluate_policy(model, policy, val_loader, device, cfg)
        record = {
            "epoch": epoch + 1,
            "train_loss": epoch_loss / max(1, n_batches),
            "train_diag": epoch_diag,
            "val_ap50": float(val_metrics["ap50"]),
            "val_ap75": float(val_metrics["ap75"]),
            "val_num_predictions": int(val_metrics.get("num_predictions", 0)),
            "val_fp_rate": float(val_metrics.get("false_positive_rate", 0.0)),
        }
        history.append(record)
        save_json(history, run_dir / "training_history.json")
        print(f"Epoch {epoch+1}: train_loss={record['train_loss']:.4f} val_AP50={record['val_ap50']:.4f} val_AP75={record['val_ap75']:.4f} #pred={record['val_num_predictions']}")

        if record["val_ap75"] > best_ap75:
            best_ap75 = record["val_ap75"]
            best_epoch = epoch + 1
            torch.save({
                "model": model.state_dict(),
                "policy": policy.state_dict(),
                "baseline_policy": baseline_policy.state_dict(),
                "epoch": epoch + 1,
                "ap75": best_ap75,
            }, run_dir / "checkpoint_best.pth")

    return history, best_epoch, best_ap75


def main():
    parser = argparse.ArgumentParser(description="Round 2.227 NWPU GRPO score rescue.")
    parser.add_argument("--run-name", default="round2227_nwpu_grpo_score_rescue")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit-train", type=int, default=16)
    parser.add_argument("--limit-val", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--kl-weight", type=float, default=0.1)
    parser.add_argument("--oracle-shaping", type=float, default=0.5)
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--max-advantage", type=float, default=3.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--low-conf-max", type=float, default=0.5)
    parser.add_argument("--max-delta", type=float, default=0.3)
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--score-threshold", type=float, default=0.05)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    run_dir = ensure_run_dir(args.run_name)

    cfg = {
        "round": "2.227",
        "git": GIT,
        "checkpoint": str(CHECKPOINT),
        "limit_train": args.limit_train,
        "limit_val": args.limit_val,
        "seed": args.seed,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "kl_weight": args.kl_weight,
        "oracle_shaping": args.oracle_shaping,
        "num_samples": args.num_samples,
        "max_advantage": args.max_advantage,
        "grad_clip": args.grad_clip,
        "low_conf_max": args.low_conf_max,
        "max_delta": args.max_delta,
        "nms_iou": args.nms_iou,
        "score_threshold": args.score_threshold,
    }
    save_json(cfg, run_dir / "round_config.json")

    train_loader, val_loader = build_loaders(limit_train=args.limit_train, limit_val=args.limit_val, seed=args.seed)
    model = build_nwpu_model(device)
    for p in model.parameters():
        p.requires_grad = False

    policy = build_policy(model, hidden_dim=128, init_std=0.05, max_delta=args.max_delta)
    baseline_policy = build_policy(model, hidden_dim=128, init_std=0.05, max_delta=args.max_delta)
    baseline_policy.load_state_dict(policy.state_dict())
    for p in baseline_policy.parameters():
        p.requires_grad = False

    # Baseline eval
    baseline_policy_for_eval = ScoreAdjustmentPolicy(
        model.roi_heads.box_predictor.cls_score.in_features, 128, 0.05, args.max_delta
    ).to(device)
    baseline_policy_for_eval.load_state_dict(policy.state_dict())
    baseline_metrics = evaluate_policy(model, baseline_policy_for_eval, val_loader, device, cfg)
    save_json({
        "ap50": float(baseline_metrics["ap50"]),
        "ap75": float(baseline_metrics["ap75"]),
        "num_predictions": int(baseline_metrics.get("num_predictions", 0)),
        "false_positive_rate": float(baseline_metrics.get("false_positive_rate", 0.0)),
    }, run_dir / "baseline_eval_metrics.json")
    print(f"Baseline: AP50={baseline_metrics['ap50']:.4f} AP75={baseline_metrics['ap75']:.4f} #pred={baseline_metrics.get('num_predictions', 0)}")

    history, best_epoch, best_ap75 = train(model, policy, baseline_policy, train_loader, val_loader, device, cfg, run_dir)

    final_metrics = evaluate_policy(model, policy, val_loader, device, cfg)
    save_json({
        "ap50": float(final_metrics["ap50"]),
        "ap75": float(final_metrics["ap75"]),
        "num_predictions": int(final_metrics.get("num_predictions", 0)),
        "false_positive_rate": float(final_metrics.get("false_positive_rate", 0.0)),
        "best_epoch": best_epoch,
        "best_ap75": best_ap75,
        "baseline_ap50": float(baseline_metrics["ap50"]),
        "baseline_ap75": float(baseline_metrics["ap75"]),
        "delta_ap50": float(final_metrics["ap50"] - baseline_metrics["ap50"]),
        "delta_ap75": float(final_metrics["ap75"] - baseline_metrics["ap75"]),
    }, run_dir / "eval_metrics.json")

    print(f"\nFinal: AP50={final_metrics['ap50']:.4f} AP75={final_metrics['ap75']:.4f}")
    print(f"Best epoch {best_epoch}: AP75={best_ap75:.4f}")
    print(f"Delta AP75: {final_metrics['ap75'] - baseline_metrics['ap75']:+.4f}")


if __name__ == "__main__":
    main()
