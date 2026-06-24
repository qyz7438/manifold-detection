"""Plan 2.52: DPO with ROI FFT quality (bug-fixed).

Fixes from audit:
  Bug 1: ref_deltas = deltas.detach() — reference log_probs must NOT carry gradient
  Bug 2: ref_sigma fully detached (not shared with training sigma)
  Bug 3: quality from ROI FFT (14x14 FPN features, semantic space), not pixel FFT

Single-channel pairwise DPO: higher spectral_quality → chosen, lower → rejected.
No Pareto voting. No pixel patches. No edge/radial channels.
"""
import sys, json, subprocess, math, copy
from pathlib import Path
import torch
import torch.nn.functional as F
import torchvision
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
ROI_SIZE = 14


def spectral_quality(roi_features):
    """2.31-style: ROI FFT (semantic space) → quality. Returns (N,) in [0,1]."""
    N = roi_features.shape[0]
    fft = torch.fft.rfft2(roi_features.float(), dim=(-2, -1), norm="ortho")
    mag = torch.abs(fft).mean(dim=1)
    mag_flat = mag.flatten(1)
    total = mag_flat.sum(dim=1, keepdim=True).clamp_min(1e-6)
    hf = mag_flat[:, mag_flat.shape[1] // 2:].sum(dim=1) / total.squeeze(1)
    mag_norm = mag_flat / total
    entropy = -(mag_norm * torch.log(mag_norm + 1e-6)).sum(dim=1)
    max_e = torch.log(torch.tensor(float(mag_flat.shape[1]), device=roi_features.device))
    e_norm = 1.0 - entropy / max_e
    pha = torch.angle(fft + 1e-6)
    pha_var = pha.std(dim=(1, 2, 3)).clamp_max(1.0)
    quality = 0.3 * hf + 0.4 * e_norm + 0.3 * (1.0 - pha_var)
    return quality.clamp(0.0, 1.0)


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
        if isinstance(part, torch.nn.Module):
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
        run_name = f"round252_dpo_roi_b{beta}_s42"
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

        fpn_cache = {}; proposal_cache = {}; roi_cache = {}

        def fpn_hook(m, i, o): fpn_cache["f"] = {k: o[k] for k in o if k != "pool"}
        def rpn_hook(m, i, o): proposal_cache["p"] = o[0]
        def roi_hook(m, i): roi_cache["x"] = i[0]

        hk_fpn = model.backbone.register_forward_hook(fpn_hook)
        hk_rpn = model.rpn.register_forward_hook(rpn_hook)
        hk_roi = model.roi_heads.box_head.register_forward_pre_hook(roi_hook)

        total_valid_pairs = 0

        for epoch in range(1, EPOCHS + 1):
            model.train()
            total_det, total_dpo = 0.0, 0.0

            for images, targets in tqdm(train_loader, desc=f"{run_name} e{epoch}"):
                images_dev = [img.to(DEV) for img in images]
                targets_t = [{k: v.to(DEV) for k, v in t.items()} for t in targets]
                fpn_cache.clear(); proposal_cache.clear(); roi_cache.clear()

                ld = model(images_dev, targets_t)
                if isinstance(ld, dict):
                    det_loss = sum(ld.values())
                elif isinstance(ld, (list, tuple)):
                    det_loss = sum(sum(d.values()) for d in ld if isinstance(d, dict))
                else:
                    det_loss = sum(ld)

                roi_feats = roi_cache.get("x")
                proposals = proposal_cache.get("p")
                fpn_feats = fpn_cache.get("f")
                dpo_loss = torch.tensor(0.0, device=DEV)

                if roi_feats is not None and proposals is not None and roi_feats.shape[0] > 0:
                    N = roi_feats.shape[0]
                    box_ft = model.roi_heads.box_head(roi_feats)
                    mu = model.roi_heads.box_predictor.bbox_pred(box_ft)[:, -4:]
                    sigma = torch.full_like(mu, 0.1)

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

                    # ROI FFT quality (Bug 3 fix: semantic space)
                    q_quality = torch.zeros(N, K_SAMPLES, device=DEV)
                    if fpn_feats is not None:
                        fpn_keys = sorted(fpn_feats.keys(), key=int)
                        pc = proposals_cat[:N]
                        P = min(pc.shape[0], 256)
                        if P > 0:
                            w = pc[:, 2] - pc[:, 0]; h_ = pc[:, 3] - pc[:, 1]
                            area = (w * h_).clamp_min(1)
                            lvl = torch.floor(torch.log2(torch.sqrt(area) / 224) + 4).long().clamp(2, 5)
                            roi14 = []
                            for i in range(P):
                                ki = min(len(fpn_keys) - 1, max(0, lvl[i].item() - 2))
                                feat = fpn_feats[fpn_keys[ki]]
                                bx = pc[i:i + 1]
                                ri = torch.cat([torch.zeros(1, 1, device=DEV), bx], dim=1)
                                sc = 1.0 / (2 ** (int(fpn_keys[ki]) + 2))
                                r14 = torchvision.ops.roi_align(feat, ri, output_size=ROI_SIZE, spatial_scale=sc)
                                roi14.append(r14)
                            if roi14:
                                q_val = spectral_quality(torch.cat(roi14, dim=0))
                                q_quality[:P, 0] = q_val[:P]
                                q_quality[:P, 1] = q_val[:P]

                    # This archived prototype cannot assign different quality to the two
                    # sampled deltas: both columns are computed from the same proposal ROI.
                    # Treat ties as invalid so the script does not create random DPO labels.
                    q_diff = q_quality[:, 0] - q_quality[:, 1]
                    chosen = q_diff >= 0
                    valid = q_diff.abs() > 1e-6

                    if valid.any():
                        lp_c = torch.where(chosen, log_probs[:, 0], log_probs[:, 1])
                        lp_r = torch.where(chosen, log_probs[:, 1], log_probs[:, 0])
                        lp_ref_c = torch.where(chosen, log_probs_ref[:, 0], log_probs_ref[:, 1])
                        lp_ref_r = torch.where(chosen, log_probs_ref[:, 1], log_probs_ref[:, 0])
                        ratio = lp_c - lp_ref_c - lp_r + lp_ref_r
                        dpo_loss = -F.logsigmoid(beta * ratio[valid]).mean()
                        total_valid_pairs += int(valid.sum().item())

                loss = det_loss + dpo_loss
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

                total_det += det_loss
                total_dpo += dpo_loss.item()

            ep_m = evaluate(model, val_loader)
            row = {"epoch": epoch, "val_ap50": ep_m["ap50"], "val_ap75": ep_m["ap75"]}
            history.append(row)
            print(f"  e{epoch}: AP50={ep_m['ap50']:.4f} det={total_det:.1f} dpo={total_dpo:.3f}")
            if ep_m["ap50"] > best_ap50:
                best_ap50 = ep_m["ap50"]

        hk_fpn.remove(); hk_rpn.remove(); hk_roi.remove()

        ep_m.update({"run_name": run_name, "beta": beta, "epochs": EPOCHS, "seed": 42,
                     "valid_dpo_pairs": total_valid_pairs,
                     "best_ap50": best_ap50, "history": history, "git_hash": GIT})
        save_json(ep_m, run_dir / "eval_metrics.json")
        all_r.append(ep_m)
        print(f"  DONE b{beta}: AP50={ep_m['ap50']:.4f} AP75={ep_m['ap75']:.4f}")

    print("\n## Plan 2.52 DPO-ROI (Bug-Fixed) Results")
    for r in all_r:
        print(f"  b{r['beta']}: AP50={r['ap50']:.4f} AP75={r['ap75']:.4f}")


if __name__ == "__main__":
    main()
