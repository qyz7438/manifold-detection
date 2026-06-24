"""Plan 2.55: Bug-fixed DPO with delta-dependent pixel FFT quality.

Key fix over 2.52: quality computed from DECODED BOX pixel patches (delta-dependent),
not from proposal ROI features. Two deltas get different quality → valid preference.

Bug fixes retained from audit:
  Bug 1: ref_deltas = deltas.detach()
  Bug 2: ref_sigma fully detached
  Bug 3: quality now delta-dependent (original image pixel patches)

Quality formula: same 2.31 spectral_quality (HF+entropy+phase_coherence) but on
64x64 pixel patches from original images at decoded box coordinates.
"""
import sys, json, subprocess, math, copy
from pathlib import Path
import torch, torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, "E:/CLIproject/RLimage")
from spectral_detection_posttrain.datasets import build_penn_fudan_loaders
from spectral_detection_posttrain.eval.detection_metrics import evaluate_detection_predictions
from spectral_detection_posttrain.models import build_detector
from spectral_detection_posttrain.utils.io import save_json, ensure_run_dir
from spectral_detection_posttrain.utils.seed import set_seed

GIT = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
DEV = "cuda" if torch.cuda.is_available() else "cpu"
CKPT = "runs/round227_v1_baseline_20ep/checkpoint_best.pth"
K_SAMPLES = 2
BETAS = [0.5, 1.0]
EPOCHS = 10
PIXEL_SIZE = 64


def pixel_fft_quality(pixel_patches):
    """2.31 formula on pixel patches: HF+entropy+phase_coherence."""
    gray = pixel_patches.mean(dim=1)
    fft = torch.fft.fft2(gray.float()).abs()
    mag_flat = fft.flatten(1)
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
    widths = proposals[:, 2] - proposals[:, 0]; heights = proposals[:, 3] - proposals[:, 1]
    ctr_x = proposals[:, 0] + 0.5 * widths; ctr_y = proposals[:, 1] + 0.5 * heights
    pred_ctr_x = deltas[:, 0] * widths + ctr_x; pred_ctr_y = deltas[:, 1] * heights + ctr_y
    pred_w = torch.exp(deltas[:, 2]) * widths; pred_h = torch.exp(deltas[:, 3]) * heights
    refined = torch.zeros_like(deltas)
    refined[:, 0] = pred_ctr_x - 0.5 * pred_w; refined[:, 1] = pred_ctr_y - 0.5 * pred_h
    refined[:, 2] = pred_ctr_x + 0.5 * pred_w; refined[:, 3] = pred_ctr_y + 0.5 * pred_h
    return refined.clamp(min=0)


def gaussian_log_prob(deltas, mu, sigma):
    eps = (deltas - mu.unsqueeze(1)) / sigma.unsqueeze(1)
    return -0.5 * (eps.pow(2) + 2 * torch.log(sigma.unsqueeze(1)) + math.log(2 * math.pi)).sum(dim=-1)


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
        if isinstance(part, nn.Module):
            for p in part.parameters():
                p.requires_grad = True


def sum_losses(ld):
    if isinstance(ld, dict): return sum(ld.values()).item()
    if isinstance(ld, (list, tuple)):
        t = 0.0
        for d in ld:
            if isinstance(d, dict):
                for v in d.values(): t += v.sum().item()
        return t
    return sum(ld).item()


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

    for beta in BETAS:
        run_name = f"round255_dpo_pix_b{beta}_s42"
        set_seed(42)

        model = build_model().to(DEV)
        ckpt = torch.load(CKPT, map_location=DEV)
        model.load_state_dict(ckpt["model"])

        ref_model = copy.deepcopy(model)
        freeze_except(ref_model, []); ref_model.eval()

        freeze_except(model, [model.roi_heads.box_head, model.roi_heads.box_predictor])

        train_loader, val_loader = build_loaders()
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)
        run_dir = ensure_run_dir(run_name)
        history = []; best_ap50 = -1.0

        proposal_cache = {}; roi_cache = {}

        def rpn_hook(m, i, o): proposal_cache["p"] = o[0]
        def roi_hook(m, i): roi_cache["x"] = i[0]

        hk_rpn = model.rpn.register_forward_hook(rpn_hook)
        hk_roi = model.roi_heads.box_head.register_forward_pre_hook(roi_hook)

        for epoch in range(1, EPOCHS + 1):
            model.train()
            total_det, total_dpo = 0.0, 0.0

            for images, targets in tqdm(train_loader, desc=f"{run_name} e{epoch}"):
                images_dev = [img.to(DEV) for img in images]
                targets_t = [{k: v.to(DEV) for k, v in t.items()} for t in targets]
                proposal_cache.clear(); roi_cache.clear()

                ld = model(images_dev, targets_t)
                if isinstance(ld, dict):
                    det_loss = sum(ld.values())
                elif isinstance(ld, (list, tuple)):
                    det_loss = sum(sum(d.values()) for d in ld if isinstance(d, dict))
                else:
                    det_loss = sum(ld)

                roi_feats = roi_cache.get("x")
                proposals = proposal_cache.get("p")
                dpo_loss = torch.tensor(0.0, device=DEV)

                if roi_feats is not None and proposals is not None and roi_feats.shape[0] > 0:
                    N = roi_feats.shape[0]
                    box_ft = model.roi_heads.box_head(roi_feats)
                    mu = model.roi_heads.box_predictor.bbox_pred(box_ft)[:, -4:]
                    sigma = torch.full_like(mu, 0.1, requires_grad=False)

                    eps = torch.randn(N, K_SAMPLES, 4, device=DEV)
                    deltas = mu.unsqueeze(1) + sigma.unsqueeze(1) * eps

                    log_probs = gaussian_log_prob(deltas, mu, sigma)
                    with torch.no_grad():
                        ref_ft = ref_model.roi_heads.box_head(roi_feats)
                        ref_mu = ref_model.roi_heads.box_predictor.bbox_pred(ref_ft)[:, -4:]
                        ref_sigma = torch.full_like(ref_mu, 0.1)
                    ref_deltas = deltas.detach()
                    log_probs_ref = gaussian_log_prob(ref_deltas, ref_mu, ref_sigma)

                    proposals_cat = torch.cat(proposals, dim=0)
                    N = min(N, proposals_cat.shape[0])
                    mu = mu[:N]; deltas = deltas[:N]; sigma = sigma[:N]
                    log_probs = log_probs[:N]; log_probs_ref = log_probs_ref[:N]

                    # Decode boxes + crop pixel patches (delta-dependent quality)
                    ad = deltas.reshape(N * K_SAMPLES, 4)
                    pe = proposals_cat[:N].unsqueeze(1).expand(-1, K_SAMPLES, -1).reshape(N * K_SAMPLES, 4)
                    all_boxes = decode_boxes(pe, ad)

                    npi = [p.shape[0] for p in proposals]
                    ii = torch.cat([torch.full((n,), i, dtype=torch.long) for i, n in enumerate(npi)], dim=0)[:N]

                    patches = []
                    for idx in range(min(N * K_SAMPLES, 256)):
                        pj = min(idx // K_SAMPLES, N - 1)
                        img_i = ii[pj].item(); img = images[img_i]; box = all_boxes[idx]
                        x1, y1 = max(0, int(box[0].round().item())), max(0, int(box[1].round().item()))
                        x2, y2 = min(img.shape[-1], max(x1 + 1, int(box[2].round().item()))), min(img.shape[-2], max(y1 + 1, int(box[3].round().item())))
                        crop = img[:, y1:y2, x1:x2]
                        if crop.shape[-1] >= 4 and crop.shape[-2] >= 4:
                            crop = F.interpolate(crop.unsqueeze(0).float(), size=(PIXEL_SIZE, PIXEL_SIZE), mode='bilinear', align_corners=False).squeeze(0)
                            patches.append(crop)
                        else:
                            patches.append(torch.zeros(3, PIXEL_SIZE, PIXEL_SIZE))

                    Kt = N * K_SAMPLES
                    if patches:
                        pb = torch.stack(patches).to(DEV)
                        q_all = pixel_fft_quality(pb)
                        qp = torch.zeros(Kt, device=DEV); qp[:len(q_all)] = q_all
                        q_matrix = qp.view(N, K_SAMPLES)  # (N, 2), delta-dependent!
                    else:
                        q_matrix = torch.zeros(N, K_SAMPLES, device=DEV)

                    q_diff = (q_matrix[:, 0] - q_matrix[:, 1]).abs()
                    chosen = q_matrix[:, 0] >= q_matrix[:, 1]
                    valid = q_diff > 0.02  # filter: only use pairs with meaningful diff

                    lp_c = torch.where(chosen & valid, log_probs[:, 0], log_probs[:, 1])
                    lp_r = torch.where(chosen & valid, log_probs[:, 1], log_probs[:, 0])
                    lp_ref_c = torch.where(chosen & valid, log_probs_ref[:, 0], log_probs_ref[:, 1])
                    lp_ref_r = torch.where(chosen & valid, log_probs_ref[:, 1], log_probs_ref[:, 0])
                    ratio = lp_c - lp_ref_c - lp_r + lp_ref_r
                    if valid.any():
                        dpo_loss = -F.logsigmoid(beta * ratio[valid]).mean()
                    else:
                        dpo_loss = torch.tensor(0.0, device=DEV)

                loss = det_loss + dpo_loss
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

                total_det += det_loss.item()
                total_dpo += dpo_loss.item()

            ep_m = evaluate(model, val_loader)
            row = {"epoch": epoch, "val_ap50": ep_m["ap50"], "val_ap75": ep_m["ap75"]}
            history.append(row)
            print(f"  e{epoch}: AP50={ep_m['ap50']:.4f} det={total_det:.1f} dpo={total_dpo:.3f}")
            if ep_m["ap50"] > best_ap50:
                best_ap50 = ep_m["ap50"]

        hk_rpn.remove(); hk_roi.remove()
        ep_m.update({"run_name": run_name, "beta": beta, "epochs": EPOCHS, "seed": 42,
                     "best_ap50": best_ap50, "history": history, "git_hash": GIT})
        save_json(ep_m, run_dir / "eval_metrics.json")
        all_r.append(ep_m)
        print(f"  DONE b{beta}: AP50={ep_m['ap50']:.4f} AP75={ep_m['ap75']:.4f}")

    print("\n## Plan 2.55 DPO-Pixel (Bug-Fixed, Delta-Dependent) Results")
    for r in all_r:
        print(f"  b{r['beta']}: AP50={r['ap50']:.4f} AP75={r['ap75']:.4f}")


if __name__ == "__main__":
    main()
