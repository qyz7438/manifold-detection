"""Plan 2.40: FPN-level FFT + Box Head stochastic reweighting + no anchor_generator.

Improvements:
  1. FFT at original FPN resolution (14x14-56x56, not 7x7)
  5. Box Head only — bypass anchor_generator entirely
  4. Stochastic reweighting from 2.31 base (sample + log_prob, not full REINFORCE)
"""
import sys, json, subprocess
from pathlib import Path
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torchvision.ops import box_iou

sys.path.insert(0, "E:/CLIproject/RLimage")
from spectral_detection_posttrain.datasets import build_penn_fudan_loaders
from spectral_detection_posttrain.eval.detection_metrics import evaluate_detection_predictions
from spectral_detection_posttrain.models import build_detector
from spectral_detection_posttrain.utils.io import save_json, ensure_run_dir
from spectral_detection_posttrain.utils.seed import set_seed

GIT = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
DEV = "cuda" if torch.cuda.is_available() else "cpu"
CKPT = "runs/round227_v1_baseline_20ep/checkpoint_best.pth"
K_SAMPLE = 32
ALPHAS = [0.1, 0.5]
BETA = 0.5       # log_prob loss weight
EPOCHS = 15


def fft_quality_fpn(fpn_feats, proposals, fpn_keys, image_sizes):
    """FPN-level FFT quality: extract patches at original FPN resolution, no ROI Align.

    Args:
        fpn_feats: dict of {level: tensor(B, C, H_l, W_l)}
        proposals: List[Tensor(N_i, 4)] per image, in image coords
        fpn_keys: sorted list of FPN level keys
        image_sizes: List[(h, w)] per image
    Returns:
        quality: (total_N,) spectral quality scores
    """
    all_quality = []
    fpn_strides = [2 ** (int(k) + 2) for k in fpn_keys]

    for img_i, props in enumerate(proposals):
        if len(props) == 0:
            continue
        img_h, img_w = image_sizes[img_i]

        for box in props:
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1
            if w < 4 or h < 4:
                all_quality.append(torch.tensor(0.0, device=DEV))
                continue

            # Assign to FPN level by box area
            area = (w * h).clamp(min=1)
            level = int(torch.floor(torch.log2(torch.sqrt(area) / 224) + 4).clamp(2, 5).item())
            ki = min(len(fpn_keys) - 1, max(0, level - min(int(k) for k in fpn_keys)))

            feat = fpn_feats[fpn_keys[ki]]  # (B, C, H_l, W_l)
            stride = fpn_strides[ki]

            # Map box to FPN coordinates (no ROI Align — use raw indexing)
            fx1 = int(x1 / stride)
            fy1 = int(y1 / stride)
            fx2 = max(fx1 + 1, int(x2 / stride + 0.5))
            fy2 = max(fy1 + 1, int(y2 / stride + 0.5))

            # Clamp to feature map bounds
            _, _, fH, fW = feat.shape
            fx1 = max(0, min(fx1, fW - 1))
            fx2 = max(fx1 + 1, min(fx2, fW))
            fy1 = max(0, min(fy1, fH - 1))
            fy2 = max(fy1 + 1, min(fy2, fH))

            # Extract patch, compute channel gradient magnitude → edge map
            patch = feat[img_i, :, fy1:fy2, fx1:fx2]  # (C, pH, pW)
            if patch.shape[-2] < 3 or patch.shape[-1] < 3:
                all_quality.append(torch.tensor(0.0, device=DEV))
                continue

            # Gradient magnitude across channels → 1-channel edge map
            gx = (patch[:, :, 1:] - patch[:, :, :-1]).pow(2).sum(dim=0)  # (pH, pW-1)
            gy = (patch[:, 1:, :] - patch[:, :-1, :]).pow(2).sum(dim=0)  # (pH-1, pW)
            edge = torch.zeros(patch.shape[-2], patch.shape[-1], device=DEV)
            edge[:, :-1] += gx
            edge[:-1, :] += gy
            edge = torch.sqrt(edge.clamp(min=1e-6))

            fft = torch.fft.fft2(edge.float())
            mag = torch.abs(fft)
            mag_flat = mag.flatten()
            total = mag_flat.sum().clamp_min(1e-6)
            mag_norm = mag_flat / total

            # Spectral stats
            n = len(mag_flat)
            hf = mag_flat[n // 2:].sum() / total
            entropy = -(mag_norm * torch.log(mag_norm + 1e-6)).sum()
            max_e = torch.log(torch.tensor(float(n), device=DEV))
            e_norm = 1.0 - entropy / max_e

            # Phase coherence
            pha = torch.angle(fft + 1e-6)
            pha_var = pha.std().clamp_max(1.0)

            quality = 0.3 * hf + 0.4 * e_norm + 0.3 * (1.0 - pha_var)
            all_quality.append(quality.clamp(0.0, 1.0))

    if all_quality:
        return torch.stack(all_quality)
    return torch.zeros(0, device=DEV)


def build_loaders():
    return build_penn_fudan_loaders({
        "data": {"root": "./data", "max_size": 320, "train_fraction": 0.8, "num_workers": 0},
        "train": {"batch_size": 2},
    })


def build_model():
    cfg = {"model": {"name": "fasterrcnn_mobilenet_v3_large_320_fpn",
                     "model_name": "fasterrcnn_mobilenet_v3_large_320_fpn",
                     "pretrained": True, "num_classes": 2, "min_size": 320, "max_size": 320}}
    return build_detector(cfg)


def freeze_except(model, trainable_parts):
    for p in model.parameters():
        p.requires_grad = False
    for part in trainable_parts:
        if isinstance(part, torch.nn.Module):
            for p in part.parameters():
                p.requires_grad = True


@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    preds, targs = [], []
    for images, targets in val_loader:
        out = model([img.to(DEV) for img in images])
        preds.extend([{k: v.cpu() for k, v in o.items()} for o in out])
        targs.extend([{k: v.cpu() for k, v in t.items()} for t in targets])
    return evaluate_detection_predictions(preds, targs, iou_threshold=0.5, score_threshold=0.05)


def main():
    all_r = []

    for alpha in ALPHAS:
        run_name = f"round240_swr_a{alpha}_s42"
        set_seed(42)

        model = build_model().to(DEV)
        ckpt = torch.load(CKPT, map_location=DEV)
        model.load_state_dict(ckpt["model"])

        # Train box_head + box_predictor; freeze backbone + RPN
        freeze_except(model, [model.roi_heads.box_head, model.roi_heads.box_predictor])

        train_loader, val_loader = build_loaders()
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)
        run_dir = ensure_run_dir(run_name)
        history = []
        best_ap50 = -1.0
        quality_baseline = None

        fpn_cache = {}
        proposal_cache = {}

        def fpn_hook(module, inp, out):
            fpn_cache["f"] = {k: out[k] for k in out if k != "pool"}

        def rpn_hook(module, inp, out):
            proposal_cache["p"] = out[0]  # proposals (decoded boxes)

        hk_fpn = model.backbone.register_forward_hook(fpn_hook)
        hk_rpn = model.rpn.register_forward_hook(rpn_hook)

        for epoch in range(1, EPOCHS + 1):
            model.train()
            total_det, total_rl = 0.0, 0.0
            avg_q = 0.0

            for images, targets in tqdm(train_loader, desc=f"{run_name} e{epoch}"):
                images = [img.to(DEV) for img in images]
                targets_t = [{k: v.to(DEV) for k, v in t.items()} for t in targets]
                fpn_cache.clear()
                proposal_cache.clear()

                # 1. Standard detection forward
                ld = model(images, targets_t)
                det_loss = sum(ld.values())

                # 2. FPN-level FFT quality on proposals
                fpn_feats = fpn_cache.get("f")
                proposals = proposal_cache.get("p")
                rl_loss = torch.tensor(0.0, device=DEV)
                quality = torch.tensor([], device=DEV)

                if fpn_feats is not None and proposals is not None:
                    fpn_keys = sorted(fpn_feats.keys(), key=int)
                    images_t, _ = model.transform(images, None)
                    quality = fft_quality_fpn(fpn_feats, proposals, fpn_keys, images_t.image_sizes)

                # 3. Weighted detection loss (2.31 style: quality × loss)
                if quality.numel() > 0:
                    box_reg = ld.get("loss_box_reg", torch.tensor(0.0, device=DEV))
                    rew = (quality * box_reg).mean()
                    avg_q = quality.mean().item()
                else:
                    rew = torch.tensor(0.0, device=DEV)

                loss = det_loss + alpha * rew
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

                total_det += det_loss.item()
                total_rl += rew.item()

            ep_m = evaluate(model, val_loader)
            row = {"epoch": epoch, "val_ap50": ep_m["ap50"], "val_ap75": ep_m["ap75"]}
            history.append(row)
            print(f"  e{epoch}: AP50={ep_m['ap50']:.4f} det={total_det:.1f} rew={total_rl:.3f} q={avg_q:.4f}")
            if ep_m["ap50"] > best_ap50:
                best_ap50 = ep_m["ap50"]

        hk_fpn.remove()
        hk_rpn.remove()

        ep_m.update({"run_name": run_name, "alpha": alpha,
                     "epochs": EPOCHS, "seed": 42,
                     "best_ap50": best_ap50, "history": history, "git_hash": GIT})
        save_json(ep_m, run_dir / "eval_metrics.json")
        all_r.append(ep_m)
        print(f"  DONE a{alpha}: AP50={ep_m['ap50']:.4f} AP75={ep_m['ap75']:.4f}")

    print("\n## Plan 2.40 Results")
    for r in all_r:
        print(f"  a{r['alpha']}: AP50={r['ap50']:.4f} AP75={r['ap75']:.4f}")


if __name__ == "__main__":
    main()
