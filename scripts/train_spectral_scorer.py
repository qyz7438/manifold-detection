"""Train a spectral quality scorer: predict IoU from ROI FFT features.

Random proposal sampling (not RPN top-K) → cover full proposal space → unbiased scorer.
"""
import sys, json, random, subprocess
from pathlib import Path
import torch, torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torchvision.ops import roi_align, box_iou

sys.path.insert(0, "E:/CLIproject/RLimage")
from spectral_detection_posttrain.datasets import build_penn_fudan_loaders
from spectral_detection_posttrain.models import build_detector
from spectral_detection_posttrain.utils.io import save_json, ensure_run_dir
from spectral_detection_posttrain.utils.seed import set_seed

DEV = "cuda" if torch.cuda.is_available() else "cpu"
CKPT = "runs/round227_v1_baseline_20ep/checkpoint_best.pth"
N_PROPOSALS_PER_IMG = 200  # random proposals per image
ROI_SIZE = 7
EPOCHS = 10
LR = 0.001


class SpectralScorer(nn.Module):
    """Predict IoU from FFT spectral features of a ROI."""

    def __init__(self, roi_size=ROI_SIZE):
        super().__init__()
        n_freq = roi_size * (roi_size // 2 + 1)  # 7*4=28 bins
        # Extract 4 stats per ROI: DC, HF_energy, entropy, phase_coherence
        in_dim = 4
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def extract_features(self, roi_features):
        """roi_features: (N, 256, H, W) → (N, 4) spectral stats"""
        N = roi_features.shape[0]
        fft = torch.fft.rfft2(roi_features.float(), dim=(-2, -1), norm="ortho")
        mag = torch.abs(fft).mean(dim=1).flatten(1)  # (N, nbins)
        total = mag.sum(dim=1, keepdim=True).clamp_min(1e-6)
        hf = mag[:, mag.shape[1] // 2:].sum(dim=1) / total.squeeze(1)
        mag_norm = mag / total
        entropy = -(mag_norm * torch.log(mag_norm + 1e-6)).sum(dim=1)
        max_e = torch.log(torch.tensor(float(mag.shape[1]), device=roi_features.device))
        e_norm = 1.0 - entropy / max_e
        pha = torch.angle(fft + 1e-6)
        pha_var = pha.std(dim=(1, 2, 3)).clamp_max(1.0)
        dc = mag[:, 0] / total.squeeze(1)  # DC ratio
        return torch.stack([dc, hf, e_norm, 1.0 - pha_var], dim=1)  # (N, 4)

    def forward(self, roi_features):
        feat = self.extract_features(roi_features)
        return self.net(feat).squeeze(1)  # (N,)


def random_proposals(fpn_feats, fpn_keys, img_h, img_w, n=N_PROPOSALS_PER_IMG):
    """Generate random proposals across all FPN levels and image regions."""
    all_boxes = []
    for _ in range(n):
        # Random FPN level
        lv = random.choice(fpn_keys)
        feat = fpn_feats[lv]  # (1, 256, H_l, W_l)
        stride = 2 ** (int(lv) + 2)
        fh, fw = feat.shape[-2:]
        # Random position on this FPN level
        px = random.randint(0, fw - 1)
        py = random.randint(0, fh - 1)
        # Random scale at this level (base_size ranges ~ 32*stride to 512*stride)
        base = 32 * stride
        w = random.uniform(base * 0.5, base * 4)
        h = random.uniform(w * 0.5, w * 2.0)
        h = max(h, base * 0.25)
        # Convert to image coordinates
        cx = px * stride + stride / 2
        cy = py * stride + stride / 2
        x1 = max(0, cx - w / 2)
        y1 = max(0, cy - h / 2)
        x2 = min(img_w, cx + w / 2)
        y2 = min(img_h, cy + h / 2)
        if x2 > x1 + 4 and y2 > y1 + 4:
            all_boxes.append([x1, y1, x2, y2])
    return torch.tensor(all_boxes, dtype=torch.float32, device=DEV)


def assign_fpn_level(boxes):
    w = boxes[:, 2] - boxes[:, 0]; h = boxes[:, 3] - boxes[:, 1]
    area = (w * h).clamp_min(1)
    return torch.floor(torch.log2(torch.sqrt(area) / 224) + 4).long().clamp(2, 5)


def build_loaders():
    return build_penn_fudan_loaders({
        "data": {"root": "./data", "max_size": 320, "train_fraction": 0.8, "num_workers": 0},
        "train": {"batch_size": 1},
    })


def build_model():
    cfg = {"model": {"name": "fasterrcnn_mobilenet_v3_large_320_fpn",
                     "model_name": "fasterrcnn_mobilenet_v3_large_320_fpn",
                     "pretrained": True, "num_classes": 2, "min_size": 320, "max_size": 320}}
    return build_detector(cfg)


def main():
    set_seed(42)

    # Load frozen model for backbone + FPN
    model = build_model().to(DEV)
    ckpt = torch.load(CKPT, map_location=DEV)
    model.load_state_dict(ckpt["model"])
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    scorer = SpectralScorer().to(DEV)
    opt = torch.optim.Adam(scorer.parameters(), lr=LR)
    train_loader, val_loader = build_loaders()
    run_dir = ensure_run_dir("runs/spectral_scorer_train")

    fpn_cache = {}

    def fpn_hook(module, inp, out):
        fpn_cache["f"] = {k: out[k] for k in out if k != "pool"}

    hk = model.backbone.register_forward_hook(fpn_hook)

    for epoch in range(1, EPOCHS + 1):
        scorer.train()
        total_loss = 0.0
        n_batches = 0

        for images, targets in tqdm(train_loader, desc=f"scorer e{epoch}"):
            images = [img.to(DEV) for img in images]
            targets = [{k: v.to(DEV) for k, v in t.items()} for t in targets]
            fpn_cache.clear()

            # Run model to get FPN features
            with torch.no_grad():
                _ = model(images, targets)

            fpn_feats = fpn_cache.get("f")
            if fpn_feats is None:
                continue

            fpn_keys = sorted(fpn_feats.keys(), key=int)
            img_h, img_w = images[0].shape[-2:]

            # Generate random proposals
            rand_boxes = random_proposals(fpn_feats, fpn_keys, img_h, img_w)  # (N, 4)
            if len(rand_boxes) < 4:
                continue

            # ROI Align on random proposals
            lvls = assign_fpn_level(rand_boxes)
            roi_list = []
            for j in range(len(rand_boxes)):
                ki = min(len(fpn_keys) - 1, max(0, lvls[j].item() - 2))
                feat_m = fpn_feats[fpn_keys[ki]]
                box_ri = torch.cat([torch.zeros(1, 1, device=DEV), rand_boxes[j:j+1]], dim=1)
                scale = 1.0 / (2 ** (int(fpn_keys[ki]) + 2))
                roi_list.append(roi_align(feat_m, box_ri, ROI_SIZE, spatial_scale=scale))
            roi_batch = torch.cat(roi_list, dim=0)

            # GT IoU
            if len(targets[0]["boxes"]) > 0:
                gt_iou = box_iou(rand_boxes, targets[0]["boxes"]).max(dim=1).values  # (N,)
            else:
                gt_iou = torch.zeros(len(rand_boxes), device=DEV)

            # Scorer prediction
            pred = scorer(roi_batch)  # (N,)
            loss = F.mse_loss(pred, gt_iou)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        print(f"  epoch {epoch}: loss={avg_loss:.4f}")

        # Quick eval on validation
        if epoch % 3 == 0:
            scorer.eval()
            val_loss = 0.0
            val_n = 0
            with torch.no_grad():
                for images, targets in tqdm(val_loader, desc="val", leave=False):
                    images = [img.to(DEV) for img in images]
                    targets = [{k: v.to(DEV) for k, v in t.items()} for t in targets]
                    fpn_cache.clear()
                    _ = model(images, targets)
                    fpn_feats = fpn_cache.get("f")
                    if fpn_feats is None:
                        continue
                    rand_boxes = random_proposals(fpn_feats, fpn_keys, img_h, img_w)
                    if len(rand_boxes) < 4:
                        continue
                    lvls = assign_fpn_level(rand_boxes)
                    roi_list = []
                    for j in range(len(rand_boxes)):
                        ki = min(len(fpn_keys) - 1, max(0, lvls[j].item() - 2))
                        feat_m = fpn_feats[fpn_keys[ki]]
                        box_ri = torch.cat([torch.zeros(1, 1, device=DEV), rand_boxes[j:j+1]], dim=1)
                        scale = 1.0 / (2 ** (int(fpn_keys[ki]) + 2))
                        roi_list.append(roi_align(feat_m, box_ri, ROI_SIZE, spatial_scale=scale))
                    roi_batch = torch.cat(roi_list, dim=0)
                    if len(targets[0]["boxes"]) > 0:
                        gt_iou = box_iou(rand_boxes, targets[0]["boxes"]).max(dim=1).values
                    else:
                        gt_iou = torch.zeros(len(rand_boxes), device=DEV)
                    pred = scorer(roi_batch)
                    val_loss += F.mse_loss(pred, gt_iou).item()
                    val_n += 1
            print(f"  val_loss={val_loss / max(val_n, 1):.4f}")

    hk.remove()

    # Save trained scorer
    torch.save(scorer.state_dict(), run_dir / "scorer.pth")
    print(f"Scorer saved to {run_dir / 'scorer.pth'}")

    # Test accuracy
    scorer.eval()
    test_err = 0.0
    test_n = 0
    with torch.no_grad():
        for images, targets in val_loader:
            images = [img.to(DEV) for img in images]
            fpn_cache.clear()
            _ = model(images, targets)
            fpn_feats = fpn_cache.get("f")
            if fpn_feats is None:
                continue
            rand_boxes = random_proposals(fpn_feats, fpn_keys, img_h, img_w)
            if len(rand_boxes) < 10:
                continue
            lvls = assign_fpn_level(rand_boxes)
            roi_list = []
            for j in range(len(rand_boxes)):
                ki = min(len(fpn_keys) - 1, max(0, lvls[j].item() - 2))
                feat_m = fpn_feats[fpn_keys[ki]]
                box_ri = torch.cat([torch.zeros(1, 1, device=DEV), rand_boxes[j:j+1]], dim=1)
                scale = 1.0 / (2 ** (int(fpn_keys[ki]) + 2))
                roi_list.append(roi_align(feat_m, box_ri, ROI_SIZE, spatial_scale=scale))
            roi_batch = torch.cat(roi_list, dim=0)
            if len(targets[0]["boxes"]) > 0:
                gt_iou = box_iou(rand_boxes, targets[0]["boxes"]).max(dim=1).values
            else:
                gt_iou = torch.zeros(len(rand_boxes), device=DEV)
            pred = scorer(roi_batch)
            err = (pred - gt_iou).abs().mean().item()
            test_err += err
            test_n += 1
            break  # one batch for quick test
    print(f"Test MAE: {test_err / max(test_n, 1):.4f}")


if __name__ == "__main__":
    main()
