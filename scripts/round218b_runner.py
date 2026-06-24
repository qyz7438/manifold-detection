"""Plan 2.18 approaches B and C — fixed implementations.

Approach B: Bbox correction via RoIAlign.
  - Freeze detector. Extract predictions from baseline model.
  - For each predicted box, crop ROI from backbone feature map via RoIAlign.
  - Tiny MLP (256->64->4) predicts bbox deltas from ROI features.
  - Loss: SmoothL1 between corrected and GT boxes.
  - Eval: apply corrections to predictions, measure metrics.

Approach C: Feature-level KL constraint on AFM path.
  - Freeze all weights except AFM.
  - Baseline forward: AFM bypassed (Identity), get box_head input features.
  - AFM forward: AFM active, get box_head input features.
  - Loss: det_loss + 0.1 * ||AFM_features - baseline_features||^2.
"""
import sys, json, subprocess
from pathlib import Path
import torch, torch.nn as nn
from torchvision.ops import roi_align
from tqdm import tqdm

sys.path.insert(0, "E:/CLIproject/RLimage")
from spectral_detection_posttrain.datasets import build_penn_fudan_loaders
from spectral_detection_posttrain.eval.detection_metrics import evaluate_detection_predictions
from spectral_detection_posttrain.models import build_detector
from spectral_detection_posttrain.models.micro_afm import MPLSegAFMBlock
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


@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    preds, targs = [], []
    for images, targets in val_loader:
        outputs = model([img.to(DEV) for img in images])
        preds.extend([{k: v.cpu() for k, v in o.items()} for o in outputs])
        targs.extend([{k: v.cpu() for k, v in t.items()} for t in targets])
    return evaluate_detection_predictions(preds, targs, iou_threshold=0.5, score_threshold=0.05)


# ==============================
# APPROACH B: Bbox correction via RoIAlign
# ==============================

class BboxRefiner(nn.Module):
    """Predict (dx,dy,dw,dh) corrections from RoI-pooled features."""
    def __init__(self, in_channels=256, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 4),
        )

    def forward(self, roi_features):
        return self.net(roi_features)  # [N, 4]


def apply_deltas(boxes, deltas):
    """Apply (dx, dy, dw, dh) to xyxy boxes."""
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    cx = boxes[:, 0] + 0.5 * w
    cy = boxes[:, 1] + 0.5 * h
    pred_cx = deltas[:, 0] * w + cx
    pred_cy = deltas[:, 1] * h + cy
    pred_w = torch.exp(deltas[:, 2]) * w
    pred_h = torch.exp(deltas[:, 3]) * h
    return torch.stack([pred_cx - 0.5 * pred_w, pred_cy - 0.5 * pred_h,
                        pred_cx + 0.5 * pred_w, pred_cy + 0.5 * pred_h], dim=1)


def encode_deltas(boxes, ref_boxes):
    """Encode boxes as deltas relative to reference boxes."""
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

    refiner = BboxRefiner(in_channels=256, hidden=64).to(DEV)
    _, val_loader = build_penn_fudan_loaders(cfg)
    cfg["data"]["max_size"] = 320
    train_loader, _ = build_penn_fudan_loaders(cfg)
    opt = torch.optim.SGD(refiner.parameters(), lr=0.001, momentum=0.9)
    run_dir = ensure_run_dir(run_name)

    for epoch in range(1, 3):
        refiner.train()
        total_loss = 0.0
        n_batches = 0
        for images, targets in tqdm(train_loader, desc=f"{run_name}"):
            images_dev = [img.to(DEV) for img in images]
            targets_dev = [{k: v.to(DEV) for k, v in t.items()} for t in targets]
            n_batches += 1

            batch_loss = torch.tensor(0.0, device=DEV)
            for i, (img, tgt) in enumerate(zip(images_dev, targets_dev)):
                gt_boxes = tgt["boxes"]
                if len(gt_boxes) == 0:
                    continue

                # Get FPN feature maps for this image
                with torch.no_grad():
                    feats = model.backbone(img.unsqueeze(0))
                    if isinstance(feats, dict):
                        feats = list(feats.values())
                    # Use P3 features (highest resolution, index 0)
                    fpn_feat = feats[0]  # [1, C, H, W]

                # RoIAlign GT boxes to get features
                img_idx = torch.zeros(len(gt_boxes), dtype=torch.float32, device=DEV)
                rois = torch.cat([img_idx.unsqueeze(1), gt_boxes], dim=1)
                roi_feats = roi_align(fpn_feat, rois, output_size=7, spatial_scale=1.0)

                deltas_pred = refiner(roi_feats)
                corrected = apply_deltas(gt_boxes, deltas_pred)
                # GT deltas are zero (corrected boxes should match GT)
                loss = nn.functional.smooth_l1_loss(corrected, gt_boxes)
                batch_loss = batch_loss + loss

            if n_batches > 0 and batch_loss.item() > 0:
                opt.zero_grad(set_to_none=True)
                batch_loss.backward()
                opt.step()
                total_loss += float(batch_loss.item())

    # Eval with refinement
    refiner.eval()
    model.eval()
    preds, targs = [], []
    for images, targets in val_loader:
        images_dev = [img.to(DEV) for img in images]
        with torch.no_grad():
            outputs = model(images_dev)
            feats = model.backbone(images_dev[0].unsqueeze(0))
            if isinstance(feats, dict):
                feats = list(feats.values())
            fpn_feat = feats[0]

        for i, out in enumerate(outputs):
            boxes = out.get("boxes", None)
            if boxes is not None and boxes.numel() > 0 and len(outputs) == len(images_dev):
                boxes_dev = boxes.to(DEV)
                scores = out.get("scores", torch.ones(len(boxes_dev)))
                labels = out.get("labels", torch.ones(len(boxes_dev), dtype=torch.int64))

                img_idx = torch.zeros(len(boxes_dev), dtype=torch.float32, device=DEV)
                rois = torch.cat([img_idx.unsqueeze(1), boxes_dev], dim=1)
                roi_feats = roi_align(fpn_feat, rois, output_size=7, spatial_scale=1.0)
                deltas = refiner(roi_feats)
                corrected_boxes = apply_deltas(boxes_dev, deltas)

                # Clamp to image bounds
                _, h, w = images_dev[i].shape
                corrected_boxes[:, [0, 2]] = corrected_boxes[:, [0, 2]].clamp(0, w)
                corrected_boxes[:, [1, 3]] = corrected_boxes[:, [1, 3]].clamp(0, h)

                preds.append({"boxes": corrected_boxes.cpu(), "scores": scores.cpu(),
                              "labels": labels.cpu()})
            else:
                o_cpu = {k: v.cpu() for k, v in out.items()}
                preds.append(o_cpu)

        targs.extend([{k: v.cpu() for k, v in t.items()} for t in targets])

    m = evaluate_detection_predictions(preds, targs, iou_threshold=0.5, score_threshold=0.05)
    m.update({"run_name": run_name, "approach": "B_roi_refiner", "git_hash": GIT})
    save_json(m, run_dir / "eval_metrics.json")
    return m


# ==============================
# APPROACH C: Feature-level constraint
# ==============================

def train_c(seed, run_name):
    set_seed(seed)
    cfg = build_cfg(seed)
    cfg["model"]["afm_channels"] = 256
    cfg["model"]["afm_type"] = "mplseg_mid"
    model = build_detector(cfg).to(DEV)
    load_checkpoint(model, CKPT, DEV)

    # Grab AFM reference
    afm = model.roi_heads.box_head.afm

    # Freeze all except AFM
    freeze_all(model)
    for p in afm.parameters():
        p.requires_grad = True

    # Hook to get AFM input and output
    afm_inputs = {}
    afm_outputs = {}

    def input_hook(m, inp):
        afm_inputs["x"] = inp[0].detach()

    def output_hook(m, inp, out):
        afm_outputs["y"] = out

    h1 = afm.register_forward_pre_hook(input_hook)
    h2 = afm.register_forward_hook(output_hook)

    _, val_loader = build_penn_fudan_loaders(cfg)
    cfg["data"]["max_size"] = 320
    train_loader, _ = build_penn_fudan_loaders(cfg)

    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                          lr=0.001, momentum=0.9, weight_decay=0.0005)
    run_dir = ensure_run_dir(run_name)

    for epoch in range(1, 3):
        model.train()
        for images, targets in tqdm(train_loader, desc=f"{run_name}"):
            images_dev = [img.to(DEV) for img in images]
            targets_dev = [{k: v.to(DEV) for k, v in t.items()} for t in targets]

            # Clear hook state
            afm_inputs.clear()
            afm_outputs.clear()

            # Forward with AFM
            ld = model(images_dev, targets_dev)
            det_loss = sum(ld.values())

            # Feature constraint: AFM output should stay close to input
            x = afm_inputs.get("x")
            y = afm_outputs.get("y")
            feat_loss = torch.tensor(0.0, device=DEV)
            if x is not None and y is not None:
                feat_loss = 0.05 * nn.functional.mse_loss(y, x)

            total = det_loss + feat_loss
            opt.zero_grad(set_to_none=True)
            total.backward()
            opt.step()

    h1.remove()
    h2.remove()
    m = evaluate(model, val_loader)
    m.update({"run_name": run_name, "approach": "C_feat_constraint", "git_hash": GIT})
    save_json(m, run_dir / "eval_metrics.json")
    return m


# Run
def main():
    for seed in SEEDS:
        print(f"\n--- Seed {seed} ---")
        r = train_b(seed, f"round218b_B_s{seed}")
        print(f"  B: AP50={r['ap50']:.4f} AP75={r['ap75']:.4f}")
        r = train_c(seed, f"round218b_C_s{seed}")
        print(f"  C: AP50={r['ap50']:.4f} AP75={r['ap75']:.4f}")

    # Summary
    lines = ["## Plan 2.18b B+C results", ""]
    lines.append("| Run | AP50 | AP75 | Prec | ECE |")
    lines.append("|---:|---:|---:|---:|---:|")
    for s in SEEDS:
        for m in ["B", "C"]:
            p = Path(f"runs/round218b_{m}_s{s}/eval_metrics.json")
            if p.exists():
                d = json.loads(p.read_text())
                lines.append(f"| {m}_s{s} | {d['ap50']:.4f} | {d['ap75']:.4f} | {d['precision']:.4f} | {d['ece']:.4f} |")
    print("\n".join(lines))
    subprocess.run(["E:/anaconda/01/envs/RLimage/python.exe", "scripts/notify_feishu.py",
                    f"Plan 2.18b B+C done: {sum(1 for s in SEEDS for m in ['B','C'] if Path(f'runs/round218b_{m}_s{s}/eval_metrics.json').exists())}/6 OK"], capture_output=True)


if __name__ == "__main__":
    main()
