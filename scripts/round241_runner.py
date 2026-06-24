"""Plan 2.41: RFT (Rejection Sampling FT) — minimal viable.

Freeze RPN → box_head outputs μ → sample M=8 deltas per proposal →
decode to image coords → crop pixel patches → FFT → top-1 delta → MSE(μ, δ_best).
"""
import sys, json, subprocess, math
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
M_SAMPLES = 8
ALPHAS = [0.1, 0.5]
EPOCHS = 15
PIXEL_SIZE = 64  # resize all patches to 64x64 for FFT


def pixel_fft_quality(pixel_patches):
    """FFT quality on physical pixel patches (not feature space).

    pixel_patches: (N, 3, 64, 64) — RGB image patches resized to fixed size.
    """
    N = pixel_patches.shape[0]
    gray = pixel_patches.mean(dim=1)  # (N, 64, 64) physical luminance
    fft = torch.fft.fft2(gray.float()).abs()
    mag_flat = fft.flatten(1)  # (N, 4096)
    total = mag_flat.sum(dim=1, keepdim=True).clamp_min(1e-6)
    hf = mag_flat[:, mag_flat.shape[1] // 2:].sum(dim=1) / total.squeeze(1)
    mag_norm = mag_flat / total
    entropy = -(mag_norm * torch.log(mag_norm + 1e-6)).sum(dim=1)
    max_e = torch.log(torch.tensor(float(mag_flat.shape[1]), device=pixel_patches.device))
    e_norm = 1.0 - entropy / max_e
    pha_var = torch.angle(torch.fft.fft2(gray.float()) + 1e-6).flatten(1).std(dim=1).clamp_max(1.0)
    quality = 0.3 * hf + 0.4 * e_norm + 0.3 * (1.0 - pha_var)
    return quality.clamp(0.0, 1.0)


def decode_boxes(proposals, deltas):
    """proposals: (N, 4) xyxy, deltas: (N, 4) dxdydwdh → decoded xyxy."""
    widths = proposals[:, 2] - proposals[:, 0]
    heights = proposals[:, 3] - proposals[:, 1]
    ctr_x = proposals[:, 0] + 0.5 * widths
    ctr_y = proposals[:, 1] + 0.5 * heights
    pred_ctr_x = deltas[:, 0] * widths + ctr_x
    pred_ctr_y = deltas[:, 1] * heights + ctr_y
    pred_w = torch.exp(deltas[:, 2]) * widths
    pred_h = torch.exp(deltas[:, 3]) * heights
    refined = torch.zeros_like(deltas)
    refined[:, 0] = pred_ctr_x - 0.5 * pred_w
    refined[:, 1] = pred_ctr_y - 0.5 * pred_h
    refined[:, 2] = pred_ctr_x + 0.5 * pred_w
    refined[:, 3] = pred_ctr_y + 0.5 * pred_h
    return refined.clamp(min=0)


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
        run_name = f"round241_rft_a{alpha}_s42"
        set_seed(42)

        model = build_model().to(DEV)
        ckpt = torch.load(CKPT, map_location=DEV)
        model.load_state_dict(ckpt["model"])

        freeze_except(model, [model.roi_heads.box_head, model.roi_heads.box_predictor])

        train_loader, val_loader = build_loaders()
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)
        run_dir = ensure_run_dir(run_name)
        history = []
        best_ap50 = -1.0

        fpn_cache = {}
        proposal_cache = {}
        roi_cache = {}

        def fpn_hook(module, inp, out):
            fpn_cache["f"] = {k: out[k] for k in out if k != "pool"}

        def rpn_hook(module, inp, out):
            proposal_cache["p"] = out[0]

        def roi_hook(module, inp):
            roi_cache["x"] = inp[0]

        hk_fpn = model.backbone.register_forward_hook(fpn_hook)
        hk_rpn = model.rpn.register_forward_hook(rpn_hook)
        hk_roi = model.roi_heads.box_head.register_forward_pre_hook(roi_hook)

        for epoch in range(1, EPOCHS + 1):
            model.train()
            total_det, total_rft = 0.0, 0.0
            avg_q = 0.0

            for images, targets in tqdm(train_loader, desc=f"{run_name} e{epoch}"):
                images_dev = [img.to(DEV) for img in images]
                targets_t = [{k: v.to(DEV) for k, v in t.items()} for t in targets]
                fpn_cache.clear()
                proposal_cache.clear()
                roi_cache.clear()

                # 1. Standard detection forward
                ld = model(images_dev, targets_t)
                det_loss = sum(ld.values())

                # 2. RFT: sample deltas → pixel patches → FFT → top-1 → MSE
                roi_feats = roi_cache.get("x")
                proposals = proposal_cache.get("p")
                rft_loss = torch.tensor(0.0, device=DEV)

                if roi_feats is not None and proposals is not None and roi_feats.shape[0] > 0:
                    N = roi_feats.shape[0]
                    box_ft = model.roi_heads.box_head(roi_feats)
                    mu = model.roi_heads.box_predictor.bbox_pred(box_ft)[:, -4:]  # (N, 4)
                    sigma = torch.full_like(mu, 0.1)

                    # Sample M deltas per proposal
                    eps = torch.randn(N, M_SAMPLES, 4, device=DEV)
                    deltas = mu.unsqueeze(1) + sigma.unsqueeze(1) * eps  # (N, M, 4)

                    # Decode to image coordinates
                    # N can differ between roi_feats and proposals; use min
                    proposals_cat_raw = torch.cat(proposals, dim=0)
                    N_prop = min(N, proposals_cat_raw.shape[0])
                    proposals_cat = proposals_cat_raw[:N_prop].unsqueeze(1).expand(-1, M_SAMPLES, -1).reshape(N_prop * M_SAMPLES, 4)
                    all_deltas = deltas[:N_prop].reshape(N_prop * M_SAMPLES, 4)
                    N = N_prop
                    mu = mu[:N]  # align mu with min(N, proposals)
                    all_boxes = decode_boxes(proposals_cat, all_deltas)  # (N*M, 4)

                    # Build per-image proposal counts for batch indexing
                    n_per_img = [p.shape[0] for p in proposals]
                    img_indices = torch.cat([torch.full((n,), i, dtype=torch.long) for i, n in enumerate(n_per_img)], dim=0)
                    img_indices = img_indices[:N]  # align with truncated proposals

                    # Crop pixel patches from original images
                    pixel_patches = []
                    for bi in range(min(N * M_SAMPLES, 256)):  # limit to 256 for speed
                        pi = min(bi // M_SAMPLES, N - 1)  # proposal index
                        img_i = img_indices[pi].item()
                        img = images[img_i]  # (3, H, W)
                        box = all_boxes[bi]
                        x1, y1, x2, y2 = box.round().long().clamp(min=0)
                        x1 = max(0, min(x1, img.shape[-1] - 1))
                        x2 = max(x1 + 1, min(x2, img.shape[-1]))
                        y1 = max(0, min(y1, img.shape[-2] - 1))
                        y2 = max(y1 + 1, min(y2, img.shape[-2]))
                        patch = img[:, y1:y2, x1:x2]  # (3, h, w)
                        if patch.shape[-1] >= 4 and patch.shape[-2] >= 4:
                            patch = F.interpolate(patch.unsqueeze(0).float(), size=(PIXEL_SIZE, PIXEL_SIZE),
                                                  mode='bilinear', align_corners=False).squeeze(0)
                            pixel_patches.append(patch)
                        else:
                            pixel_patches.append(torch.zeros(3, PIXEL_SIZE, PIXEL_SIZE))

                    if pixel_patches:
                        patch_batch = torch.stack(pixel_patches).to(DEV)  # (K, 3, 64, 64)
                        qualities = pixel_fft_quality(patch_batch)  # (K,)

                        # Reshape and find best delta per proposal
                        K_valid = (N * M_SAMPLES)
                        q_pad = torch.zeros(K_valid, device=DEV)
                        q_pad[:len(qualities)] = qualities
                        q_matrix = q_pad.view(N, M_SAMPLES)  # (N, M)
                        best_idx = q_matrix.argmax(dim=1)  # (N,)

                        # MSE: μ → δ_best
                        best_deltas = deltas[torch.arange(N, device=DEV), best_idx]  # (N, 4)
                        rft_loss = F.mse_loss(mu, best_deltas.detach())
                        avg_q = qualities.mean().item()

                loss = det_loss + alpha * rft_loss
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

                total_det += det_loss.item()
                total_rft += rft_loss.item()

            ep_m = evaluate(model, val_loader)
            row = {"epoch": epoch, "val_ap50": ep_m["ap50"], "val_ap75": ep_m["ap75"]}
            history.append(row)
            print(f"  e{epoch}: AP50={ep_m['ap50']:.4f} det={total_det:.1f} rft={total_rft:.3f} q={avg_q:.4f}")
            if ep_m["ap50"] > best_ap50:
                best_ap50 = ep_m["ap50"]

        hk_fpn.remove(); hk_rpn.remove(); hk_roi.remove()

        ep_m.update({"run_name": run_name, "alpha": alpha,
                     "epochs": EPOCHS, "seed": 42,
                     "best_ap50": best_ap50, "history": history, "git_hash": GIT})
        save_json(ep_m, run_dir / "eval_metrics.json")
        all_r.append(ep_m)
        print(f"  DONE a{alpha}: AP50={ep_m['ap50']:.4f} AP75={ep_m['ap75']:.4f}")

    print("\n## Plan 2.41 Results")
    for r in all_r:
        print(f"  a{r['alpha']}: AP50={r['ap50']:.4f} AP75={r['ap75']:.4f}")


if __name__ == "__main__":
    main()
