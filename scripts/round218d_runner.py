"""Plan 2.18d B v4: train corrector on predicted boxes, not GT boxes."""
import sys, json, subprocess
from pathlib import Path
import torch, torch.nn as nn
from torchvision.ops import box_iou
from tqdm import tqdm

sys.path.insert(0, "E:/CLIproject/RLimage")
from spectral_detection_posttrain.datasets import build_penn_fudan_loaders
from spectral_detection_posttrain.eval.detection_metrics import evaluate_detection_predictions
from spectral_detection_posttrain.models import build_detector
from spectral_detection_posttrain.utils.io import load_checkpoint, save_json, ensure_run_dir
from spectral_detection_posttrain.utils.seed import resolve_device, set_seed

GIT = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
CKPT = "runs/round216p_mid06_s42/checkpoint_last.pth"
SEEDS = [42, 123, 456]
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def build_cfg(seed):
    return {
        "seed": seed, "device": DEV,
        "data": {"root": "./data", "download": True, "max_size": 320,
                 "train_fraction": 0.8, "num_workers": 0},
        "model": {"name": "fasterrcnn_mobilenet_v3_large_320_fpn",
                  "pretrained": True, "num_classes": 2, "min_size": 320, "max_size": 320},
        "train": {"batch_size": 2},
    }


def freeze_all(model):
    for p in model.parameters():
        p.requires_grad = False


class BoxCorrector(nn.Module):
    """Predict (dx,dy,dw,dh) from box coords + score + size features."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 4),
        )

    def forward(self, boxes_xyxy, scores, img_h, img_w):
        """boxes: [N,4] xyxy, scores: [N]"""
        w = boxes_xyxy[:, 2] - boxes_xyxy[:, 0]
        h = boxes_xyxy[:, 3] - boxes_xyxy[:, 1]
        cx = boxes_xyxy[:, 0] + 0.5 * w
        cy = boxes_xyxy[:, 1] + 0.5 * h
        features = torch.stack([
            cx / img_w, cy / img_h,
            w / img_w, h / img_h,
            scores,
            w / h.clamp_min(1e-6),
        ], dim=1)
        return self.net(features)


def apply_deltas(boxes, deltas, img_h, img_w):
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    cx = boxes[:, 0] + 0.5 * w
    cy = boxes[:, 1] + 0.5 * h
    pred_cx = deltas[:, 0] * w + cx
    pred_cy = deltas[:, 1] * h + cy
    pred_w = torch.exp(deltas[:, 2]) * w
    pred_h = torch.exp(deltas[:, 3]) * h
    x1 = (pred_cx - 0.5 * pred_w).clamp(0, img_w)
    y1 = (pred_cy - 0.5 * pred_h).clamp(0, img_h)
    x2 = (pred_cx + 0.5 * pred_w).clamp(0, img_w)
    y2 = (pred_cy + 0.5 * pred_h).clamp(0, img_h)
    return torch.stack([x1, y1, x2, y2], dim=1)


def encode_deltas(boxes, ref_boxes):
    w = ref_boxes[:, 2] - ref_boxes[:, 0]
    h = ref_boxes[:, 3] - ref_boxes[:, 1]
    cx = ref_boxes[:, 0] + 0.5 * w
    cy = ref_boxes[:, 1] + 0.5 * h
    dx = (boxes[:, 0] + 0.5 * (boxes[:, 2] - boxes[:, 0]) - cx) / w.clamp_min(1e-6)
    dy = (boxes[:, 1] + 0.5 * (boxes[:, 3] - boxes[:, 1]) - cy) / h.clamp_min(1e-6)
    dw = ((boxes[:, 2] - boxes[:, 0]) / w.clamp_min(1e-6)).log()
    dh = ((boxes[:, 3] - boxes[:, 1]) / h.clamp_min(1e-6)).log()
    return torch.stack([dx, dy, dw, dh], dim=1)


def train_b(seed, run_name):
    set_seed(seed)
    cfg = build_cfg(seed)
    cfg["model"]["afm_channels"] = 256
    cfg["model"]["afm_type"] = "mplseg_mid"
    model = build_detector(cfg).to(DEV)
    load_checkpoint(model, CKPT, DEV)
    freeze_all(model)
    model.eval()

    corrector = BoxCorrector().to(DEV)
    _, val_loader = build_penn_fudan_loaders(cfg)
    cfg["data"]["max_size"] = 320
    train_loader, _ = build_penn_fudan_loaders(cfg)
    opt = torch.optim.SGD(corrector.parameters(), lr=0.001, momentum=0.9)
    run_dir = ensure_run_dir(run_name)

    for epoch in range(1, 3):
        corrector.train()
        n_train = 0
        for images, targets in tqdm(train_loader, desc=f"{run_name}"):
            images_dev = [img.to(DEV) for img in images]
            targets_dev = [{k: v.to(DEV) for k, v in t.items()} for t in targets]
            batch_loss = torch.tensor(0.0, device=DEV)

            with torch.no_grad():
                outputs = model(images_dev)

            for i, (out, tgt) in enumerate(zip(outputs, targets_dev)):
                pred_boxes = out.get("boxes", None)
                pred_scores = out.get("scores", None)
                gt_boxes = tgt["boxes"]
                if pred_boxes is None or pred_boxes.numel() == 0 or gt_boxes.numel() == 0:
                    continue

                # Match predicted boxes to GT
                ious = box_iou(pred_boxes, gt_boxes)
                best_iou, best_gt = ious.max(dim=1)

                # Train on boxes with IoU >= 0.5
                mask = best_iou >= 0.5
                if mask.sum() < 2:
                    continue

                matched_boxes = pred_boxes[mask]
                matched_scores = pred_scores[mask]
                matched_gt = gt_boxes[best_gt[mask]]

                _, h, w = images_dev[i].shape
                deltas_pred = corrector(matched_boxes, matched_scores, h, w)
                deltas_gt = encode_deltas(matched_gt, matched_boxes)

                loss = nn.functional.smooth_l1_loss(deltas_pred, deltas_gt)
                batch_loss = batch_loss + loss
                n_train += 1

            if batch_loss.item() > 0:
                opt.zero_grad(set_to_none=True)
                batch_loss.backward()
                opt.step()

    # Eval: apply corrections
    corrector.eval()
    model.eval()
    preds, targs = [], []
    for images, targets in val_loader:
        images_dev = [img.to(DEV) for img in images]
        with torch.no_grad():
            outputs = model(images_dev)
        for i, out in enumerate(outputs):
            boxes = out.get("boxes", None)
            scores = out.get("scores", None)
            _, h, w = images_dev[i].shape
            if boxes is not None and boxes.numel() > 0:
                boxes_dev = boxes.to(DEV)
                scores_dev = scores.to(DEV)
                deltas = corrector(boxes_dev, scores_dev, h, w)
                corrected = apply_deltas(boxes_dev, deltas, h, w)
                preds.append({"boxes": corrected.cpu(),
                              "scores": scores.cpu(),
                              "labels": out["labels"].cpu()})
            else:
                preds.append({k: v.cpu() for k, v in out.items()})
        targs.extend([{k: v.cpu() for k, v in t.items()} for t in targets])

    m = evaluate_detection_predictions(preds, targs, iou_threshold=0.5, score_threshold=0.05)
    m.update({"run_name": run_name, "approach": "B_v4_predbox", "git_hash": GIT})
    save_json(m, run_dir / "eval_metrics.json")
    return m


def main():
    for seed in SEEDS:
        print(f"\n=== Seed {seed} ===")
        r = train_b(seed, f"round218d_B_s{seed}")
        print(f"  B v4: AP50={r['ap50']:.4f} AP75={r['ap75']:.4f} Prec={r['precision']:.4f}")

    print("\n## Plan 2.18d B v4 Results")
    for s in SEEDS:
        p = Path(f"runs/round218d_B_s{s}/eval_metrics.json")
        if p.exists():
            d = json.loads(p.read_text())
            print(f"  B_s{s}: AP50={d['ap50']:.4f} AP75={d['ap75']:.4f} Prec={d['precision']:.4f} ECE={d['ece']:.4f}")

    subprocess.run(["E:/anaconda/01/envs/RLimage/python.exe", "scripts/notify_feishu.py",
                    "Plan 2.18d B v4 done"], capture_output=True)

if __name__ == "__main__":
    main()
