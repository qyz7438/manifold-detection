"""Penn-Fudan Segmentation Dataset + FCN Training for Plan 4.x.

Minimal self-contained pipeline: FCN-ResNet50 + optional AFM on Penn-Fudan masks.
Supports baseline training, mid06 fine-tune, and A/C post-training.
"""
from __future__ import annotations

import os, sys, json, random, subprocess, time
from pathlib import Path
import numpy as np
import torch, torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.models.segmentation import fcn_resnet50
from torchvision.transforms import functional as TF
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, "E:/CLIproject/RLimage")
from spectral_detection_posttrain.models.micro_afm import MPLSegAFMBlock
from spectral_detection_posttrain.utils.seed import set_seed
from spectral_detection_posttrain.utils.io import save_json, ensure_run_dir

GIT = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
DEV = "cuda" if torch.cuda.is_available() else "cpu"

# ─── Dataset ────────────────────────────────────────────────
class PennFudanSegDataset(Dataset):
    """Penn-Fudan Pedestrian segmentation dataset."""
    def __init__(self, root: str, split: str = "train", fraction: float = 0.8, max_size: int = 320):
        self.root = Path(root)
        self.max_size = max_size
        imgs = sorted((self.root / "PNGImages").glob("*.png"))
        n_train = int(len(imgs) * fraction)
        self.images = imgs[:n_train] if split == "train" else imgs[n_train:]
        self.masks = [self.root / "PedMasks" / (p.stem + "_mask.png") for p in self.images]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("RGB")
        mask = Image.open(self.masks[idx])
        # resize
        w, h = img.size
        scale = min(self.max_size / max(w, h), 1.0)
        nw, nh = int(w * scale), int(h * scale)
        img = TF.resize(img, [nh, nw])
        mask = TF.resize(mask, [nh, nw], interpolation=TF.InterpolationMode.NEAREST)
        img_t = TF.to_tensor(img)
        mask_t = torch.as_tensor(np.array(mask), dtype=torch.long)
        # Penn-Fudan masks: 0=background, 1=pedestrian, 2=border → merge border to person
        mask_t[mask_t == 2] = 1
        return img_t, mask_t


def build_seg_loaders(root: str, max_size: int = 320, batch_size: int = 1, num_workers: int = 0):
    train_ds = PennFudanSegDataset(root, "train", max_size=max_size)
    val_ds = PennFudanSegDataset(root, "val", max_size=max_size)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=_collate),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=_collate),
    )


def _collate(batch):
    return tuple(zip(*batch))


# ─── AFM Wrapper ────────────────────────────────────────────
class FCNSegAFM(nn.Module):
    """Insert AFM into FCN classifier head."""
    def __init__(self, base_model, afm_type: str = "mplseg_mid", in_ch: int = 2048, gate_strength: float = 0.6):
        super().__init__()
        self.backbone = base_model.backbone
        self.classifier = base_model.classifier
        if afm_type == "none":
            self.afm = None
        else:
            self.afm = MPLSegAFMBlock(in_ch=in_ch, gate_strength=gate_strength)

    def forward(self, x: torch.Tensor):
        features = self.backbone(x)
        feat = features["out"]
        if self.afm is not None:
            feat = self.afm(feat)
        result = self.classifier(feat)
        result = F.interpolate(result, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return {"out": result}


# ─── Metrics ─────────────────────────────────────────────────
@torch.no_grad()
def compute_miou(pred, target, num_classes=2):
    """Mean IoU over batch."""
    pred = pred.argmax(1)
    ious = []
    for c in range(num_classes):
        pred_c = (pred == c)
        target_c = (target == c)
        intersection = (pred_c & target_c).sum().float()
        union = (pred_c | target_c).sum().float()
        if union > 0:
            ious.append((intersection / union).item())
    return float(np.mean(ious)) if ious else 0.0


@torch.no_grad()
def evaluate_seg(model, loader):
    model.eval()
    total_miou = 0.0
    total = 0
    for images, masks in loader:
        x = torch.stack([img.to(DEV) for img in images])
        targets = torch.stack([m.to(DEV) for m in masks])
        out = model(x)["out"]
        targets = F.interpolate(targets.unsqueeze(1).float(),
            size=out.shape[-2:], mode="nearest").squeeze(1).long()
        for i in range(len(x)):
            miou = compute_miou(out[i:i+1], targets[i:i+1], num_classes=2)
            total_miou += miou
            total += 1
    return total_miou / max(total, 1)


# ─── Training Functions ──────────────────────────────────────
def train_baseline(seed, epochs, run_name, afm_type="none"):
    set_seed(seed)
    model = fcn_resnet50(weights=None, weights_backbone=None)
    model = FCNSegAFM(model, afm_type=afm_type).to(DEV)
    train_loader, val_loader = build_seg_loaders("./data/PennFudanPed")
    opt = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)
    run_dir = ensure_run_dir(run_name)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for images, masks in tqdm(train_loader, desc=f"{run_name} e{epoch}"):
            x = torch.stack([img.to(DEV) for img in images])
            targets = torch.stack([m.to(DEV) for m in masks])
            out = model(x)["out"]
            targets = F.interpolate(targets.unsqueeze(1).float(),
                size=out.shape[-2:], mode="nearest").squeeze(1).long()
            loss = F.cross_entropy(out, targets)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total_loss += loss.item()

    torch.save(model.state_dict(), run_dir / "checkpoint_last.pth")
    miou = evaluate_seg(model, val_loader)
    result = {"run_name": run_name, "miou": miou, "afm_type": afm_type,
              "epochs": epochs, "seed": seed, "git_hash": GIT}
    save_json(result, run_dir / "eval_metrics.json")
    return result


def train_post_A(seed, ckpt_path, epochs, run_name):
    """Post-training Approach A: weak gate, AFM-only."""
    set_seed(seed)
    model = fcn_resnet50(weights=None, weights_backbone=None)
    model = FCNSegAFM(model, afm_type="mplseg_mid").to(DEV)
    ckpt = torch.load(ckpt_path, map_location=DEV)
    model.load_state_dict(ckpt, strict=False)

    in_ch = model.afm.mp[0].in_channels
    new_afm = MPLSegAFMBlock(in_ch=in_ch, gate_strength=0.1).to(DEV)
    new_afm.load_state_dict(model.afm.state_dict(), strict=False)
    model.afm = new_afm

    for p in model.parameters():
        p.requires_grad = False
    for p in model.afm.parameters():
        p.requires_grad = True

    train_loader, val_loader = build_seg_loaders("./data/PennFudanPed")
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                          lr=0.001, momentum=0.9, weight_decay=0.0005)
    run_dir = ensure_run_dir(run_name)

    for epoch in range(1, epochs + 1):
        model.train()
        for images, masks in tqdm(train_loader, desc=f"{run_name} e{epoch}"):
            x = torch.stack([img.to(DEV) for img in images])
            targets = torch.stack([m.to(DEV) for m in masks])
            out = model(x)["out"]
            targets = F.interpolate(targets.unsqueeze(1).float(),
                size=out.shape[-2:], mode="nearest").squeeze(1).long()
            loss = F.cross_entropy(out, targets)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    miou = evaluate_seg(model, val_loader)
    result = {"run_name": run_name, "miou": miou, "approach": "A",
              "epochs": epochs, "seed": seed, "git_hash": GIT}
    save_json(result, run_dir / "eval_metrics.json")
    return result


def train_post_C(seed, ckpt_path, epochs, run_name):
    """Post-training Approach C: feature constraint."""
    set_seed(seed)
    model = fcn_resnet50(weights=None, weights_backbone=None)
    model = FCNSegAFM(model, afm_type="mplseg_mid").to(DEV)
    ckpt = torch.load(ckpt_path, map_location=DEV)
    model.load_state_dict(ckpt, strict=False)

    for p in model.parameters():
        p.requires_grad = False
    for p in model.afm.parameters():
        p.requires_grad = True

    afm_in = {}
    def pre_hook(m, inp): afm_in["x"] = inp[0].detach()
    def fwd_hook(m, inp, out): afm_in["y"] = out
    h1 = model.afm.register_forward_pre_hook(pre_hook)
    h2 = model.afm.register_forward_hook(fwd_hook)

    train_loader, val_loader = build_seg_loaders("./data/PennFudanPed")
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                          lr=0.001, momentum=0.9, weight_decay=0.0005)
    run_dir = ensure_run_dir(run_name)

    for epoch in range(1, epochs + 1):
        model.train()
        for images, masks in tqdm(train_loader, desc=f"{run_name} e{epoch}"):
            x = torch.stack([img.to(DEV) for img in images])
            targets = torch.stack([m.to(DEV) for m in masks])
            afm_in.clear()
            out = model(x)["out"]
            targets = F.interpolate(targets.unsqueeze(1).float(),
                size=out.shape[-2:], mode="nearest").squeeze(1).long()
            seg_loss = F.cross_entropy(out, targets)
            feat_loss = torch.tensor(0.0, device=DEV)
            xi = afm_in.get("x"); yi = afm_in.get("y")
            if xi is not None and yi is not None:
                feat_loss = 0.05 * F.mse_loss(yi, xi)
            total = seg_loss + feat_loss
            opt.zero_grad(set_to_none=True)
            total.backward()
            opt.step()

    h1.remove(); h2.remove()
    miou = evaluate_seg(model, val_loader)
    result = {"run_name": run_name, "miou": miou, "approach": "C",
              "epochs": epochs, "seed": seed, "git_hash": GIT}
    save_json(result, run_dir / "eval_metrics.json")
    return result


def main():
    SEEDS = [42, 123, 456]
    all_r = []

    # Phase 1: Baseline + mid06 (3 seeds)
    print("=== Plan 4.1-4.2: Baseline + mid06 ===")
    for seed in SEEDS:
        r = train_baseline(seed, 3, f"round41_baseline_s{seed}", afm_type="none")
        all_r.append(r)
        print(f"  baseline_s{seed}: mIoU={r['miou']:.4f}")

        r = train_baseline(seed, 3, f"round42_mid06_s{seed}", afm_type="mplseg_mid")
        all_r.append(r)
        print(f"  mid06_s{seed}: mIoU={r['miou']:.4f}")

    # Phase 2: A/C post-training (3 seeds, from mid06 seed42 checkpoint)
    print("\n=== Plan 4.5: A/C Post-training ===")
    ckpt = "runs/round42_mid06_s42/checkpoint_last.pth"
    for seed in SEEDS:
        r = train_post_A(seed, ckpt, 2, f"round45_A_s{seed}")
        all_r.append(r)
        print(f"  A_s{seed}: mIoU={r['miou']:.4f}")

        r = train_post_C(seed, ckpt, 2, f"round45_C_s{seed}")
        all_r.append(r)
        print(f"  C_s{seed}: mIoU={r['miou']:.4f}")

    lines = ["## Plans 4.1-4.5 PF Segmentation", "",
             "| Run | mIoU |", "|---:|---:|"]
    for r in all_r:
        lines.append(f"| {r['run_name']} | {r['miou']:.4f} |")
    msg = "\n".join(lines)
    print(f"\n{msg}")
    subprocess.run([sys.executable, "scripts/notify_feishu.py", msg[:800]], capture_output=True)


if __name__ == "__main__":
    main()
