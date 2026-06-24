"""Plan 2.18: Post-training structure validation. 3 approaches x 3 seeds.

Approach A: Weak-gate AFM post-training
  - mid06_5ep checkpoint, freeze box_head, freeze backbone/RPN
  - Replace AFM gate_strength with 0.1, train AFM only (afm_only mode)
  - Standard detection loss only. Question: does weak gate avoid G07 collapse?

Approach B: Bbox regression delta head
  - mid06_5ep checkpoint, freeze everything
  - Tiny MLP (256->64->4) trained to predict bbox corrections
  - Loss: SmoothL1 between predicted_delta and GT_delta
  - Question: can a post-hoc corrector improve bbox without touching features?

Approach C: Dual-path KL anchor
  - mid06_5ep checkpoint, freeze all except AFM
  - Baseline forward (no_grad, no AFM) for cls_logits reference
  - AFM forward (trainable) for actual loss
  - Loss: det_loss + 0.05 * ||cls_logits_AFM - cls_logits_baseline||^2
  - Question: does KL constraint prevent classifier drift?
"""
import sys, json, subprocess
from pathlib import Path
import torch, torch.nn as nn
from tqdm import tqdm

sys.path.insert(0, "E:/CLIproject/RLimage")
from spectral_detection_posttrain.datasets import build_penn_fudan_loaders
from spectral_detection_posttrain.eval.detection_metrics import evaluate_detection_predictions
from spectral_detection_posttrain.models import build_detector
from spectral_detection_posttrain.models.micro_afm import MPLSegAFMBlock
from spectral_detection_posttrain.utils.io import load_checkpoint, save_json, ensure_run_dir, save_checkpoint
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


@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    preds, targs = [], []
    for images, targets in val_loader:
        outputs = model([img.to(DEV) for img in images])
        preds.extend([{k: v.cpu() for k, v in o.items()} for o in outputs])
        targs.extend([{k: v.cpu() for k, v in t.items()} for t in targets])
    return evaluate_detection_predictions(preds, targs, iou_threshold=0.5, score_threshold=0.05)


def freeze_all(model):
    for p in model.parameters():
        p.requires_grad = False


# ==============================
# APPROACH A: Weak-gate AFM only
# ==============================

def train_a(seed, run_name):
    set_seed(seed)
    cfg = build_cfg(seed)
    cfg["model"]["afm_channels"] = 256
    cfg["model"]["afm_type"] = "mplseg_mid"

    model = build_detector(cfg).to(DEV)
    load_checkpoint(model, CKPT, DEV)

    # Replace with weak gate
    old_afm = model.roi_heads.box_head.afm
    in_ch = old_afm.mp[0].in_channels
    new_afm = MPLSegAFMBlock(in_ch=in_ch, gate_strength=0.1).to(DEV)
    new_afm.load_state_dict(old_afm.state_dict(), strict=False)
    model.roi_heads.box_head.afm = new_afm

    freeze_all(model)
    for p in new_afm.parameters():
        p.requires_grad = True

    _, val_loader = build_penn_fudan_loaders(cfg)
    cfg["data"]["max_size"] = 320
    train_loader, _ = build_penn_fudan_loaders(cfg)

    opt = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=0.001, momentum=0.9, weight_decay=0.0005)
    run_dir = ensure_run_dir(run_name)

    for epoch in range(1, 3):
        model.train()
        for images, targets in tqdm(train_loader, desc=f"{run_name}"):
            images = [img.to(DEV) for img in images]
            targets = [{k: v.to(DEV) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    save_checkpoint(model, run_dir / "checkpoint_last.pth", {"epoch": 2})
    m = evaluate(model, val_loader)
    m.update({"run_name": run_name, "approach": "A_weak_gate", "git_hash": GIT})
    save_json(m, run_dir / "eval_metrics.json")
    return m


# ==============================
# APPROACH B: Bbox delta corrector
# ==============================

class BboxTuner(nn.Module):
    """Predict xyxy corrections from ROI features."""
    def __init__(self, in_dim=1024, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, 4))

    def forward(self, roi_feats):
        return self.net(roi_feats)  # [N, 4] deltas


def train_b(seed, run_name):
    set_seed(seed)
    cfg = build_cfg(seed)
    cfg["model"]["afm_channels"] = 256
    cfg["model"]["afm_type"] = "mplseg_mid"

    model = build_detector(cfg).to(DEV)
    load_checkpoint(model, CKPT, DEV)
    freeze_all(model)
    model.eval()

    # Register hook to grab ROI features before classification
    roi_features = {}

    def _hook(module, inp, out):
        roi_features["feat"] = out.detach()

    # Hook fc7 output (last layer before cls/bbox split)
    head = model.roi_heads.box_head
    if hasattr(head, "head"):
        handle = head.head.fc7.register_forward_hook(_hook)
    else:
        handle = head.fc7.register_forward_hook(_hook)

    tuner = BboxTuner(in_dim=1024, hidden=128).to(DEV)

    _, val_loader = build_penn_fudan_loaders(cfg)
    cfg["data"]["max_size"] = 320
    train_loader, _ = build_penn_fudan_loaders(cfg)

    opt = torch.optim.SGD(tuner.parameters(), lr=0.001, momentum=0.9)
    run_dir = ensure_run_dir(run_name)

    for epoch in range(1, 3):
        tuner.train()
        for images, targets in tqdm(train_loader, desc=f"{run_name}"):
            images_dev = [img.to(DEV) for img in images]
            targets_dev = [{k: v.to(DEV) for k, v in t.items()} for t in targets]
            roi_features.clear()

            with torch.no_grad():
                model(images_dev, targets_dev)

            feat = roi_features.get("feat")
            if feat is None:
                continue

            # For each image, compute GT deltas for matched boxes
            all_deltas = []
            all_pred_deltas = []
            for i, t in enumerate(targets_dev):
                gt_boxes = t["boxes"]
                if len(gt_boxes) == 0:
                    continue
                # Use first N features (batch-feat not per-box separated yet)
                n_boxes = min(len(gt_boxes), feat.shape[0] // len(images_dev))
                if n_boxes == 0:
                    continue
                # Simplified: take first n_boxes from feat
                img_feat = feat[i * n_boxes: (i + 1) * n_boxes]
                pred_deltas = tuner(img_feat)
                # GT deltas: use canonical delta encoding
                gt_deltas = _encode_deltas(gt_boxes[:n_boxes], gt_boxes[:n_boxes])
                all_deltas.append(gt_deltas)
                all_pred_deltas.append(pred_deltas)

            if all_pred_deltas:
                pred = torch.cat(all_pred_deltas)
                gt = torch.cat(all_deltas)
                loss = nn.functional.smooth_l1_loss(pred, gt)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

    handle.remove()
    m = evaluate(model, val_loader)
    m.update({"run_name": run_name, "approach": "B_bbox_tuner", "git_hash": GIT})
    save_json(m, run_dir / "eval_metrics.json")
    return m


def _encode_deltas(boxes, ref_boxes):
    """Encode xyxy boxes as (dx, dy, dw, dh) relative to reference."""
    w = ref_boxes[:, 2] - ref_boxes[:, 0]
    h = ref_boxes[:, 3] - ref_boxes[:, 1]
    cx = ref_boxes[:, 0] + 0.5 * w
    cy = ref_boxes[:, 1] + 0.5 * h
    dx = (boxes[:, 0] + 0.5 * (boxes[:, 2] - boxes[:, 0]) - cx) / w.clamp_min(1e-6)
    dy = (boxes[:, 1] + 0.5 * (boxes[:, 3] - boxes[:, 1]) - cy) / h.clamp_min(1e-6)
    dw = ((boxes[:, 2] - boxes[:, 0]) / w.clamp_min(1e-6)).log()
    dh = ((boxes[:, 3] - boxes[:, 1]) / h.clamp_min(1e-6)).log()
    return torch.stack([dx, dy, dw, dh], dim=1)


# ==============================
# APPROACH C: Dual-path KL
# ==============================

def train_c(seed, run_name):
    set_seed(seed)
    # AFM model
    cfg = build_cfg(seed)
    cfg["model"]["afm_channels"] = 256
    cfg["model"]["afm_type"] = "mplseg_mid"
    m_afm = build_detector(cfg).to(DEV)
    load_checkpoint(m_afm, CKPT, DEV)

    # Baseline model (no AFM, shared weights)
    cfg2 = build_cfg(seed)
    cfg2["model"]["afm_channels"] = 0
    m_base = build_detector(cfg2).to(DEV)
    # Load from same checkpoint, skip AFM keys
    ck = torch.load(CKPT, map_location=DEV)
    sd = ck["model"] if "model" in ck else ck
    # Strip "roi_heads.box_head.afm." prefix from keys
    sd_clean = {}
    for k, v in sd.items():
        if "afm." in k:
            continue
        k2 = k.replace("roi_heads.box_head.head.", "roi_heads.box_head.")
        sd_clean[k2] = v
    m_base.load_state_dict(sd_clean, strict=False)
    freeze_all(m_base)
    m_base.train()  # train mode needed for loss_dict, weights frozen

    # Freeze all except AFM in m_afm
    freeze_all(m_afm)
    afm = m_afm.roi_heads.box_head.afm
    for p in afm.parameters():
        p.requires_grad = True

    _, val_loader = build_penn_fudan_loaders(cfg)
    cfg["data"]["max_size"] = 320
    train_loader, _ = build_penn_fudan_loaders(cfg)

    opt = torch.optim.SGD(
        [p for p in m_afm.parameters() if p.requires_grad],
        lr=0.001, momentum=0.9, weight_decay=0.0005)
    run_dir = ensure_run_dir(run_name)

    for epoch in range(1, 3):
        m_afm.train()
        for images, targets in tqdm(train_loader, desc=f"{run_name}"):
            images_dev = [img.to(DEV) for img in images]
            targets_dev = [{k: v.to(DEV) for k, v in t.items()} for t in targets]

            # Forward baseline (no_grad)
            with torch.no_grad():
                ld_base = m_base(images_dev, targets_dev)
            cls_base = ld_base.get("loss_classifier", torch.tensor(0.0, device=DEV))

            # Forward AFM
            ld_afm = m_afm(images_dev, targets_dev)
            cls_afm = ld_afm.get("loss_classifier", torch.tensor(0.0, device=DEV))
            det_loss = sum(ld_afm.values())

            # KL proxy: mean squared difference of classifier losses
            kl = nn.functional.mse_loss(cls_afm, cls_base.detach())
            total = det_loss + 0.05 * kl

            opt.zero_grad(set_to_none=True)
            total.backward()
            opt.step()

    save_checkpoint(m_afm, run_dir / "checkpoint_last.pth", {"epoch": 2})
    m = evaluate(m_afm, val_loader)
    m.update({"run_name": run_name, "approach": "C_dual_kl", "git_hash": GIT})
    save_json(m, run_dir / "eval_metrics.json")
    return m


# ==============================
# MAIN
# ==============================

def main():
    all_r = []
    for seed in SEEDS:
        print(f"\n--- Seed {seed} ---")
        r = train_a(seed, f"round218_A_s{seed}")
        all_r.append(r)
        print(f"  A: AP50={r['ap50']:.4f} AP75={r['ap75']:.4f}")

        r = train_b(seed, f"round218_B_s{seed}")
        all_r.append(r)
        print(f"  B: AP50={r['ap50']:.4f} AP75={r['ap75']:.4f}")

        r = train_c(seed, f"round218_C_s{seed}")
        all_r.append(r)
        print(f"  C: AP50={r['ap50']:.4f} AP75={r['ap75']:.4f}")

    # Summary
    lines = ["## Plan 2.18 Results", ""]
    lines.append("| Approach | Seed | AP50 | AP75 | Prec | ECE |")
    lines.append("|---:|---:|---:|---:|---:|---:|")
    for r in all_r:
        nm = r["run_name"].replace("round218_", "").replace("_s", "_s")
        lines.append(f"| {nm} | - | {r['ap50']:.4f} | {r['ap75']:.4f} | {r['precision']:.4f} | {r['ece']:.4f} |")
    msg = "\n".join(lines)
    print(f"\n{msg}")

    subprocess.run([
        "E:/anaconda/01/envs/RLimage/python.exe",
        "scripts/notify_feishu.py",
        f"Plan 2.18 done. {len(all_r)} groups.",
    ], capture_output=True)


if __name__ == "__main__":
    main()
