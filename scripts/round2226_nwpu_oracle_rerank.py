"""Round 2.226: NWPU oracle continuous-reward re-ranking (Phase 0).

Goal: establish an upper bound for how much AP75 can improve if we had a
perfect LC-HI signal.  We take the baseline detector, disable its internal NMS,
compute an oracle reward per proposal using ground-truth IoU, re-score the
proposals with several oracle fusion strategies, run class-wise NMS again, and
re-evaluate.

This is an RLVR/DPO sanity check: if even an oracle reward does not lift AP75,
the problem is not the reward model but the candidate distribution / NMS stage.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.ops import box_iou
from torchvision.transforms import functional as TF

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spectral_detection_posttrain.core.models.build_detector import build_detector
from spectral_detection_posttrain.eval.detection_metrics import evaluate_detection_predictions
from spectral_detection_posttrain.utils.io import ensure_run_dir, save_json
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


def build_loaders(limit_val: int | None, batch_size: int = 1, seed: int = 42):
    _, val_ids = split_ids()
    val_ids = limited_ids(val_ids, limit_val, seed=seed)
    val_ds = NWPUDataset(DATA, ANNOT, val_ids, MAX_SIZE)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate, num_workers=0)
    return val_loader


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
    state_dict = checkpoint["model"]
    model.load_state_dict(state_dict)
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


@torch.no_grad()
def extract_all_proposals(model, val_loader, device, score_threshold=0.0, nms_threshold=1.0, detections_per_img=1000):
    previous = configure_detector_for_all_proposals(model, score_threshold, nms_threshold, detections_per_img)
    model.eval()
    try:
        results = []
        for images, targets in val_loader:
            outputs = model([img.to(device) for img in images])
            for output, target in zip(outputs, targets):
                results.append({
                    "boxes": output["boxes"].detach().cpu(),
                    "scores": output["scores"].detach().cpu(),
                    "labels": output["labels"].detach().cpu(),
                    "gt_boxes": target["boxes"],
                    "gt_labels": target["labels"],
                    "image_id": int(target["image_id"].item()),
                })
        return results
    finally:
        restore_detector_config(model, previous)


def compute_oracle_reward(boxes: torch.Tensor, scores: torch.Tensor, gt_boxes: torch.Tensor) -> torch.Tensor:
    if gt_boxes.numel() == 0 or boxes.numel() == 0:
        return torch.zeros_like(scores)
    ious = box_iou(boxes, gt_boxes).max(dim=1).values
    return ious * (1 - scores)


def class_wise_nms(boxes: torch.Tensor, scores: torch.Tensor, labels: torch.Tensor, iou_threshold: float = 0.5):
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long)
    keep_indices = []
    for cls in torch.unique(labels).tolist():
        mask = labels == cls
        idx = torch.nonzero(mask, as_tuple=False).flatten()
        cls_keep = torchvision.ops.nms(boxes[idx], scores[idx], iou_threshold)
        keep_indices.append(idx[cls_keep])
    return torch.cat(keep_indices) if keep_indices else torch.empty((0,), dtype=torch.long)


def apply_reranking_strategy(results, strategy: str, alpha: float | None = None, nms_iou: float = 0.5):
    reranked = []
    for r in results:
        boxes = r["boxes"]
        scores = r["scores"]
        labels = r["labels"]
        gt_boxes = r["gt_boxes"]

        reward = compute_oracle_reward(boxes, scores, gt_boxes)

        if strategy == "baseline":
            new_scores = scores.clone()
        elif strategy == "oracle_add":
            new_scores = (scores + alpha * reward).clamp(0.0, 1.0)
        elif strategy == "oracle_mul":
            eps = 1e-6
            new_scores = (scores ** alpha) * ((reward + eps) ** (1 - alpha))
        elif strategy == "oracle_replace":
            new_scores = reward.clone()
        elif strategy == "iou_only":
            new_scores = box_iou(boxes, gt_boxes).max(dim=1).values if gt_boxes.numel() > 0 else torch.zeros_like(scores)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        keep = class_wise_nms(boxes, new_scores, labels, nms_iou)
        reranked.append({
            "boxes": boxes[keep],
            "scores": new_scores[keep],
            "labels": labels[keep],
            "gt_boxes": gt_boxes,
            "gt_labels": r["gt_labels"],
        })
    return reranked


def evaluate_reranked(reranked, score_threshold: float = 0.05):
    predictions = []
    targets = []
    for r in reranked:
        predictions.append({
            "boxes": r["boxes"],
            "scores": r["scores"],
            "labels": r["labels"],
        })
        targets.append({
            "boxes": r["gt_boxes"],
            "labels": r["gt_labels"],
        })
    return evaluate_detection_predictions(predictions, targets, iou_threshold=0.5, score_threshold=score_threshold)


def main():
    parser = argparse.ArgumentParser(description="Round 2.226 NWPU oracle continuous reward re-ranking.")
    parser.add_argument("--run-name", default="round2226_nwpu_oracle_rerank")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--score-threshold", type=float, default=0.05)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    run_dir = ensure_run_dir(args.run_name)

    save_json({
        "round": "2.226",
        "git": GIT,
        "checkpoint": str(CHECKPOINT),
        "limit_val": args.limit_val,
        "seed": args.seed,
        "nms_iou": args.nms_iou,
        "score_threshold": args.score_threshold,
    }, run_dir / "round_config.json")

    val_loader = build_loaders(limit_val=args.limit_val, seed=args.seed)
    model = build_nwpu_model(device)

    print("Extracting all proposals from baseline...")
    all_proposals = extract_all_proposals(
        model, val_loader, device,
        score_threshold=0.0, nms_threshold=1.0, detections_per_img=1000,
    )

    strategies = [
        ("baseline", None),
        ("iou_only", None),
        ("oracle_replace", None),
    ]
    for alpha in [0.1, 0.3, 0.5, 0.7, 0.9]:
        strategies.append(("oracle_add", alpha))
        strategies.append(("oracle_mul", alpha))

    records = []
    baseline_metrics = None
    for strategy, alpha in strategies:
        key = strategy if alpha is None else f"{strategy}_a{alpha}"
        print(f"Running {key}...")
        reranked = apply_reranking_strategy(all_proposals, strategy, alpha, nms_iou=args.nms_iou)
        metrics = evaluate_reranked(reranked, score_threshold=args.score_threshold)
        record = {
            "strategy": strategy,
            "alpha": alpha,
            "ap50": float(metrics["ap50"]),
            "ap75": float(metrics["ap75"]),
            "num_predictions": int(metrics.get("num_predictions", 0)),
            "false_positive_rate": float(metrics.get("false_positive_rate", 0.0)),
        }
        if strategy == "baseline":
            baseline_metrics = record
        records.append(record)
        print(f"  {key}: AP50={record['ap50']:.4f} AP75={record['ap75']:.4f} #pred={record['num_predictions']}")

    if baseline_metrics is not None:
        for rec in records:
            rec["delta_ap50"] = rec["ap50"] - baseline_metrics["ap50"]
            rec["delta_ap75"] = rec["ap75"] - baseline_metrics["ap75"]

    report = {
        "baseline": baseline_metrics,
        "strategies": records,
        "best_ap75": max(records, key=lambda x: x["ap75"]),
        "best_delta_ap75": max(records, key=lambda x: x.get("delta_ap75", -1.0)),
    }
    save_json(report, run_dir / "oracle_rerank_report.json")
    print(f"\nBest AP75: {report['best_ap75']['strategy']} AP75={report['best_ap75']['ap75']:.4f}")
    print(f"Best delta AP75: {report['best_delta_ap75']['strategy']} dAP75={report['best_delta_ap75']['delta_ap75']:+.4f}")
    print(f"Report saved to {run_dir / 'oracle_rerank_report.json'}")


if __name__ == "__main__":
    main()
