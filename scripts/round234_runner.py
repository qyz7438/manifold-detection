"""Plan 2.34: BBox exploration + spectral reward via REINFORCE.

box_head outputs Gaussian policy (μ, logσ). Sample M candidate deltas per
positive proposal, score with spectral quality, update via REINFORCE.
"""
import sys, json, subprocess
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torchvision.ops import roi_align

sys.path.insert(0, "E:/CLIproject/RLimage")
from spectral_detection_posttrain.datasets import build_penn_fudan_loaders
from spectral_detection_posttrain.eval.detection_metrics import evaluate_detection_predictions
from spectral_detection_posttrain.models import build_detector
from spectral_detection_posttrain.utils.io import save_json, ensure_run_dir
from spectral_detection_posttrain.utils.seed import set_seed

GIT = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
DEV = "cuda" if torch.cuda.is_available() else "cpu"
CKPT = "runs/round227_v1_baseline_20ep/checkpoint_best.pth"
SEEDS = [42]
M_SAMPLES = 8
SIGMAS = [0.05, 0.1]
EPOCHS = 5


def spectral_quality(roi_features):
    N = roi_features.shape[0]
    fft = torch.fft.rfft2(roi_features.float(), dim=(-2, -1), norm="ortho")
    mag = torch.abs(fft).mean(dim=1)
    mag_flat = mag.flatten(1)
    total = mag_flat.sum(dim=1, keepdim=True).clamp_min(1e-6)
    hf_ratio = mag_flat[:, 14:].sum(dim=1) / total.squeeze(1)
    mag_norm = mag_flat / total
    entropy = -(mag_norm * torch.log(mag_norm + 1e-6)).sum(dim=1)
    max_entropy = torch.log(torch.tensor(float(mag_flat.shape[1]), device=roi_features.device))
    entropy_norm = 1.0 - entropy / max_entropy
    pha = torch.angle(fft + 1e-6)
    pha_var = pha.std(dim=(1, 2, 3)).clamp_max(1.0)
    phase_coherence = 1.0 - pha_var
    quality = 0.3 * hf_ratio + 0.4 * entropy_norm + 0.3 * phase_coherence
    return quality.clamp(0.0, 1.0)


def decode_boxes(anchors, deltas):
    """Decode bbox deltas to absolute boxes."""
    wx, wy, ww, wh = deltas.unbind(1)
    ax = (anchors[:, 0] + anchors[:, 2]) / 2
    ay = (anchors[:, 1] + anchors[:, 3]) / 2
    aw = anchors[:, 2] - anchors[:, 0]
    ah = anchors[:, 3] - anchors[:, 1]
    dx = wx * aw + ax; dy = wy * ah + ay
    dw = torch.exp(ww) * aw; dh = torch.exp(wh) * ah
    proposals = torch.stack([dx - dw/2, dy - dh/2, dx + dw/2, dy + dh/2], dim=1)
    return proposals.clamp(min=0)


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

    for sigma_init in SIGMAS:
        for seed in SEEDS:
            run_name = f"round234_d2_bbox_rl_s{sigma_init}_s{seed}"
            set_seed(seed)

            model = build_model().to(DEV)
            ckpt = torch.load(CKPT, map_location=DEV)
            model.load_state_dict(ckpt["model"])

            # Freeze backbone; train box_head + box_predictor
            freeze_except(model, [model.roi_heads.box_head, model.roi_heads.box_predictor])

            train_loader, val_loader = build_loaders()
            params = [p for p in model.parameters() if p.requires_grad]
            opt = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)
            run_dir = ensure_run_dir(run_name)
            history = []
            best_ap50 = -1.0
            reward_baseline = None

            # Hooks for ROI features and proposals during forward
            roi_cache = {}
            proposals_cache = {}
            fpn_cache = {}

            def roi_hook(module, inp):
                roi_cache["x"] = inp[0]

            def rpn_hook(module, inp, out):
                proposals_cache["p"] = out[0]

            def fpn_hook(module, inp, out):
                fpn_cache["f"] = {k: v for k, v in out.items() if k != "pool"}

            hk_roi = model.roi_heads.box_head.register_forward_pre_hook(roi_hook)
            hk_rpn = model.rpn.register_forward_hook(rpn_hook)
            hk_fpn = model.backbone.register_forward_hook(fpn_hook)

            for epoch in range(1, EPOCHS + 1):
                model.train()
                total_rl_loss = 0.0
                total_det_loss = 0.0
                avg_reward = 0.0

                for images, targets in tqdm(train_loader, desc=f"{run_name} e{epoch}"):
                    images = [img.to(DEV) for img in images]
                    targets = [{k: v.to(DEV) for k, v in t.items()} for t in targets]
                    roi_cache.clear()
                    proposals_cache.clear()

                    # Standard forward for detection loss
                    ld = model(images, targets)
                    det_loss = sum(ld.values())

                    # --- RL: BBox exploration with real ROI Align on candidate boxes ---
                    roi_feats = roi_cache.get("x")
                    proposals = proposals_cache.get("p")
                    fpn_feats = fpn_cache.get("f")

                    rl_loss = torch.tensor(0.0, device=DEV)
                    all_rewards = torch.tensor([], device=DEV)

                    if roi_feats is not None and proposals is not None and fpn_feats is not None and roi_feats.shape[0] > 0:
                        N = roi_feats.shape[0]
                        box_ft = model.roi_heads.box_head(roi_feats)
                        means = model.roi_heads.box_predictor.bbox_pred(box_ft)[:, -4:]
                        sigmas = torch.full_like(means, sigma_init)

                        proposals_flat = torch.cat(proposals, dim=0)
                        P = min(N, proposals_flat.shape[0], 16)
                        anchors = proposals_flat[:P].detach()

                        # FPN levels sorted by stride
                        fpn_keys = sorted(fpn_feats.keys(), key=lambda k: int(k))
                        fpn_strides = [2 ** (int(k) + 2) for k in fpn_keys]

                        def assign_level(boxes):
                            w, h = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
                            area = (w * h).clamp_min(1)
                            lvl = torch.floor(torch.log2(torch.sqrt(area) / 224) + 4).long()
                            return lvl.clamp(2, 5)

                        total_rew, total_lp = [], []
                        for i in range(P):
                            anc = anchors[i:i+1].expand(M_SAMPLES, 4)
                            noise = torch.randn(M_SAMPLES, 4, device=DEV)
                            delta = means[i:i+1].expand(M_SAMPLES, 4) + noise * sigma_init
                            cand = decode_boxes(anc, delta)
                            img_h, img_w = images[0].shape[-2:]
                            cand[:, 0].clamp_(0, img_w); cand[:, 2].clamp_(0, img_w)
                            cand[:, 1].clamp_(0, img_h); cand[:, 3].clamp_(0, img_h)

                            # Assign candidate boxes to FPN levels and ROI Align
                            lvls = assign_level(cand)  # (M,)
                            roi_feats_m = []
                            for j in range(M_SAMPLES):
                                lv = lvls[j].item()
                                # Map level to FPN key index
                                ki = min(len(fpn_keys)-1, max(0, lv - 2))
                                feat_map = fpn_feats[fpn_keys[ki]]  # (B, 256, H, W)
                                box = cand[j:j+1]  # (1, 4)
                                bi = torch.zeros(1, 1, device=DEV)
                                box_ri = torch.cat([bi, box], dim=1)
                                scale = 1.0 / fpn_strides[ki]
                                rf = roi_align(feat_map, box_ri, output_size=7, spatial_scale=scale)
                                roi_feats_m.append(rf)
                            roi_m = torch.cat(roi_feats_m, dim=0)  # (M, 256, 7, 7)
                            rw = spectral_quality(roi_m)
                            dist = torch.distributions.Normal(means[i:i+1], sigmas[i:i+1])
                            lp = dist.log_prob(delta).sum(dim=1)
                            total_rew.append(rw)
                            total_lp.append(lp)

                        if total_rew:
                            all_rw = torch.cat(total_rew)
                            all_lp = torch.cat(total_lp)
                            all_rewards = all_rw
                            mean_rw = all_rw.mean().item()
                            if reward_baseline is None:
                                reward_baseline = mean_rw
                            else:
                                reward_baseline = 0.9 * reward_baseline + 0.1 * mean_rw
                            advantages = all_rw - reward_baseline
                            rl_loss = -(advantages.detach() * all_lp).mean()
                            avg_reward = mean_rw

                    loss = det_loss + rl_loss
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()

                    total_det_loss += det_loss.item()
                    total_rl_loss += rl_loss.item()

                ep_m = evaluate(model, val_loader)
                row = {"epoch": epoch, "val_ap50": ep_m["ap50"], "val_ap75": ep_m["ap75"],
                       "det_loss": total_det_loss, "rl_loss": total_rl_loss, "avg_reward": avg_reward}
                history.append(row)
                print(f"  e{epoch}: AP50={ep_m['ap50']:.4f} det={total_det_loss:.2f} "
                      f"rl={total_rl_loss:.4f} avg_r={avg_reward:.3f}")
                if ep_m["ap50"] > best_ap50:
                    best_ap50 = ep_m["ap50"]

            hk_roi.remove()
            hk_rpn.remove()
            hk_fpn.remove()

            ep_m.update({"run_name": run_name, "sigma_init": sigma_init,
                         "epochs": EPOCHS, "seed": seed,
                         "best_ap50": best_ap50, "history": history, "git_hash": GIT})
            save_json(ep_m, run_dir / "eval_metrics.json")
            all_r.append(ep_m)
            print(f"  DONE σ={sigma_init}: AP50={ep_m['ap50']:.4f} AP75={ep_m['ap75']:.4f}")

    print("\n## Plan 2.34 Results")
    for r in all_r:
        print(f"  σ={r['sigma_init']}: AP50={r['ap50']:.4f} AP75={r['ap75']:.4f}")


if __name__ == "__main__":
    main()
