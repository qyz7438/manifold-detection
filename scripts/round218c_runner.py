"""Plan 2.18 B+C final: B=coord corrector, C=feature constraint (seeds 123,456)."""
import sys, json, subprocess
from pathlib import Path
import torch, torch.nn as nn
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


# =======================
# B v3: Coordinate corrector — learn systematic bbox bias from box coords alone
# =======================

class CoordCorrector(nn.Module):
    """Learn (dx,dy,dw,dh) corrections from normalized box coordinates."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4, 32), nn.ReLU(), nn.Linear(32, 4))

    def forward(self, boxes_xyxy, img_h, img_w):
        """boxes: [N,4] xyxy, returns [N,4] deltas"""
        norm = boxes_xyxy.clone()
        norm[:, [0, 2]] /= img_w
        norm[:, [1, 3]] /= img_h
        return self.net(norm)


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


def train_b(seed, run_name):
    set_seed(seed)
    cfg = build_cfg(seed)
    cfg["model"]["afm_channels"] = 256
    cfg["model"]["afm_type"] = "mplseg_mid"
    model = build_detector(cfg).to(DEV)
    load_checkpoint(model, CKPT, DEV)
    freeze_all(model)
    model.eval()

    corrector = CoordCorrector().to(DEV)
    _, val_loader = build_penn_fudan_loaders(cfg)
    cfg["data"]["max_size"] = 320
    train_loader, _ = build_penn_fudan_loaders(cfg)
    opt = torch.optim.SGD(corrector.parameters(), lr=0.001, momentum=0.9)
    run_dir = ensure_run_dir(run_name)

    for epoch in range(1, 3):
        corrector.train()
        for images, targets in tqdm(train_loader, desc=f"{run_name}"):
            images_dev = [img.to(DEV) for img in images]
            targets_dev = [{k: v.to(DEV) for k, v in t.items()} for t in targets]
            batch_loss = torch.tensor(0.0, device=DEV)

            for i, t in enumerate(targets_dev):
                gt_boxes = t["boxes"]
                if len(gt_boxes) < 2:
                    continue
                _, h, w = images_dev[i].shape
                deltas = corrector(gt_boxes, h, w)
                corrected = apply_deltas(gt_boxes, deltas, h, w)
                loss = nn.functional.smooth_l1_loss(corrected, gt_boxes)
                batch_loss = batch_loss + loss

            if batch_loss.item() > 0:
                opt.zero_grad(set_to_none=True)
                batch_loss.backward()
                opt.step()

    # Eval: apply corrections to predictions
    corrector.eval()
    model.eval()
    preds, targs = [], []
    for images, targets in val_loader:
        images_dev = [img.to(DEV) for img in images]
        with torch.no_grad():
            outputs = model(images_dev)
        for i, out in enumerate(outputs):
            boxes = out.get("boxes", None)
            _, h, w = images_dev[i].shape
            if boxes is not None and boxes.numel() > 0:
                boxes_dev = boxes.to(DEV)
                deltas = corrector(boxes_dev, h, w)
                corrected = apply_deltas(boxes_dev, deltas, h, w)
                preds.append({"boxes": corrected.cpu(),
                              "scores": out["scores"].cpu(),
                              "labels": out["labels"].cpu()})
            else:
                preds.append({k: v.cpu() for k, v in out.items()})
        targs.extend([{k: v.cpu() for k, v in t.items()} for t in targets])

    m = evaluate_detection_predictions(preds, targs, iou_threshold=0.5, score_threshold=0.05)
    m.update({"run_name": run_name, "approach": "B_coord_corrector", "git_hash": GIT})
    save_json(m, run_dir / "eval_metrics.json")
    return m


# =======================
# C v2: Feature constraint (fixed hooks)
# =======================

def train_c(seed, run_name):
    set_seed(seed)
    cfg = build_cfg(seed)
    cfg["model"]["afm_channels"] = 256
    cfg["model"]["afm_type"] = "mplseg_mid"
    model = build_detector(cfg).to(DEV)
    load_checkpoint(model, CKPT, DEV)
    afm = model.roi_heads.box_head.afm

    freeze_all(model)
    for p in afm.parameters():
        p.requires_grad = True

    afm_in = {}
    def pre_hook(m, inp):
        afm_in["x"] = inp[0].detach()
    def fwd_hook(m, inp, out):
        afm_in["y"] = out

    h1 = afm.register_forward_pre_hook(pre_hook)
    h2 = afm.register_forward_hook(fwd_hook)

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
            afm_in.clear()
            ld = model(images_dev, targets_dev)
            det_loss = sum(ld.values())
            feat_loss = torch.tensor(0.0, device=DEV)
            x = afm_in.get("x")
            y = afm_in.get("y")
            if x is not None and y is not None:
                feat_loss = 0.05 * nn.functional.mse_loss(y, x)
            total = det_loss + feat_loss
            opt.zero_grad(set_to_none=True)
            total.backward()
            opt.step()

    h1.remove(); h2.remove()
    m = evaluate(model, val_loader)
    m.update({"run_name": run_name, "approach": "C_feat_constraint", "git_hash": GIT})
    save_json(m, run_dir / "eval_metrics.json")
    return m


def main():
    all_r = []
    for seed in SEEDS:
        print(f"\n=== Seed {seed} ===")
        r = train_b(seed, f"round218c_B_s{seed}")
        all_r.append(r)
        print(f"  B: AP50={r['ap50']:.4f} AP75={r['ap75']:.4f}")
        r = train_c(seed, f"round218c_C_s{seed}")
        all_r.append(r)
        print(f"  C: AP50={r['ap50']:.4f} AP75={r['ap75']:.4f}")

    print("\n## Plan 2.18c Results")
    print("| Run | AP50 | AP75 | Prec | ECE |")
    print("|---:|---:|---:|---:|---:|")
    for r in all_r:
        nm = r["run_name"].replace("round218c_", "")
        print(f"| {nm} | {r['ap50']:.4f} | {r['ap75']:.4f} | {r['precision']:.4f} | {r['ece']:.4f} |")

    subprocess.run(["E:/anaconda/01/envs/RLimage/python.exe", "scripts/notify_feishu.py",
                    f"Plan 2.18c done: {len(all_r)} groups OK"], capture_output=True)

if __name__ == "__main__":
    main()
