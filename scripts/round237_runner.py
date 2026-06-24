"""Plan 2.37: RPN REINFORCE with per-proposal frequency reward (fixed).

Each sampled anchor → independent ROI Align → FFT → compare with matched GT → reward.
"""
import sys, json, subprocess
from pathlib import Path
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torchvision.ops import roi_align, box_iou

sys.path.insert(0, "E:/CLIproject/RLimage")
from spectral_detection_posttrain.datasets import build_penn_fudan_loaders
from spectral_detection_posttrain.eval.detection_metrics import evaluate_detection_predictions
from spectral_detection_posttrain.models import build_detector
from spectral_detection_posttrain.utils.io import save_json, ensure_run_dir
from spectral_detection_posttrain.utils.seed import set_seed

GIT = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
DEV = "cuda" if torch.cuda.is_available() else "cpu"
CKPT = "runs/round227_v1_baseline_20ep/checkpoint_best.pth"
K_SAMPLES = 20
ALPHAS = [0.1, 0.5]
EPOCHS = 5
PATCH_SIZE = 7


def fourier_box_reward(roi_feats_pred, roi_feats_gt):
    """Compare FFT spectra: predicted ROI vs matched GT ROI.

    Args:
        roi_feats_pred: (K, C, P, P) - sampled box features
        roi_feats_gt:  (K, C, P, P) - matched GT box features (or zeros for bg)

    Returns:
        rewards: (K,) - negative spectral distance (higher = better)
    """
    K = roi_feats_pred.shape[0]

    # Channel average + 2D FFT
    pred_gray = roi_feats_pred.mean(dim=1)  # (K, P, P)
    gt_gray = roi_feats_gt.mean(dim=1)      # (K, P, P)

    A_pred = torch.fft.fft2(pred_gray.float()).abs()  # (K, P, P)
    A_gt = torch.fft.fft2(gt_gray.float()).abs()
    A_pred = torch.log1p(A_pred)
    A_gt = torch.log1p(A_gt)

    # Radial frequency masks
    freq = torch.fft.fftfreq(PATCH_SIZE, device=roi_feats_pred.device)
    Y, X = torch.meshgrid(freq, freq, indexing='ij')
    radius = torch.sqrt(X**2 + Y**2)

    masks = {'low': radius <= 1.0, 'mid': (radius > 1.0) & (radius <= 2.5), 'high': radius > 2.5}
    weights = {'low': 0.5, 'mid': 0.2, 'high': 0.3}

    rewards = torch.zeros(K, device=roi_feats_pred.device)
    for name, mask in masks.items():
        diff = (A_pred[:, mask] - A_gt[:, mask]).abs().mean(dim=-1)
        rewards -= weights[name] * diff

    return rewards


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
        run_name = f"round237_rpn_rl_v2_a{alpha}_s42"
        set_seed(42)

        model = build_model().to(DEV)
        ckpt = torch.load(CKPT, map_location=DEV)
        model.load_state_dict(ckpt["model"])

        freeze_except(model, [model.rpn.head])

        train_loader, val_loader = build_loaders()
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)
        run_dir = ensure_run_dir(run_name)
        history = []
        best_ap50 = -1.0
        reward_baseline = None

        fpn_cache = {}

        def fpn_hook(module, inp, out):
            fpn_cache["f"] = {k: out[k] for k in out if k != "pool"}

        hk_fpn = model.backbone.register_forward_hook(fpn_hook)

        for epoch in range(1, EPOCHS + 1):
            model.train()
            total_rl = 0.0
            total_det = 0.0
            avg_reward = 0.0

            for images, targets in tqdm(train_loader, desc=f"{run_name} e{epoch}"):
                images = [img.to(DEV) for img in images]
                targets = [{k: v.to(DEV) for k, v in t.items()} for t in targets]
                fpn_cache.clear()

                # 1. Standard forward → det_loss
                ld = model(images, targets)
                det_loss = sum(ld.values())

                # 2. RL: RPN sampling + per-proposal FFT reward
                fpn_feats = fpn_cache.get("f")
                rl_loss = torch.tensor(0.0, device=DEV)

                if fpn_feats is not None:
                    fpn_keys = sorted(fpn_feats.keys(), key=int)
                    fpn_levels = [fpn_feats[k] for k in fpn_keys]

                    # Manual anchor computation (bypass buggy anchor_generator)
                    ag = model.rpn.anchor_generator
                    image_size = images[0].shape[-2:]
                    all_anchors = []
                    for li in range(len(fpn_keys)):
                        if li >= len(ag.sizes):
                            break
                        feat = fpn_feats[fpn_keys[li]]  # (B, 256, H_l, W_l)
                        _, _, H, W = feat.shape
                        stride = 2 ** (int(fpn_keys[li]) + 2)
                        sizes = torch.tensor(ag.sizes[li], device=DEV, dtype=torch.float32)
                        ratios = torch.tensor(ag.aspect_ratios[li], device=DEV, dtype=torch.float32)
                        # Vectorized grid of anchors for this level
                        cy, cx = torch.meshgrid(
                            (torch.arange(H, device=DEV) + 0.5) * stride,
                            (torch.arange(W, device=DEV) + 0.5) * stride, indexing='ij')
                        cy, cx = cy.flatten(), cx.flatten()  # (H*W,)
                        # For each (cx,cy), each size, each ratio
                        all_cx, all_cy = [], []
                        all_wa, all_ha = [], []
                        for sz in sizes:
                            for r in ratios:
                                all_cx.append(cx)
                                all_cy.append(cy)
                                all_wa.append(torch.full_like(cx, sz * torch.sqrt(r)))
                                all_ha.append(torch.full_like(cx, sz / torch.sqrt(r)))
                        all_cx = torch.cat(all_cx); all_cy = torch.cat(all_cy)
                        all_wa = torch.cat(all_wa); all_ha = torch.cat(all_ha)
                        anchors_l = torch.stack([
                            all_cx - all_wa/2, all_cy - all_ha/2,
                            all_cx + all_wa/2, all_cy + all_ha/2
                        ], dim=1)  # (N_anchors, 4)
                        all_anchors.append(anchors_l)

                    # Old policy (no grad)
                    with torch.no_grad():
                        old_obj, _ = model.rpn.head(fpn_levels)
                    # New policy (with grad)
                    new_obj, _ = model.rpn.head(fpn_levels)

                    all_rewards = []
                    all_log_probs = []

                    for li in range(min(len(fpn_keys), len(all_anchors))):
                        obj_f = new_obj[li].flatten()
                        probs = torch.softmax(obj_f, dim=0)
                        old_probs = torch.softmax(old_obj[li].flatten().detach(), dim=0)
                        lvl_anchors = all_anchors[li]

                        k = min(K_SAMPLES, len(probs), len(lvl_anchors))
                        sampled = torch.multinomial(probs, k, replacement=False)
                        samp_anchors = lvl_anchors[sampled].clone()  # (k, 4)

                        # Per-proposal ROI Align on FPN → FFT → compare with GT
                        feat_map = fpn_feats[fpn_keys[li]]  # (B, 256, H, W)
                        stride = 2 ** (int(fpn_keys[li]) + 2)
                        scale = 1.0 / stride

                        # ROI Align for sampled proposals
                        box_ri = torch.cat([torch.zeros(k, 1, device=DEV), samp_anchors], dim=1)
                        samp_roi = roi_align(feat_map, box_ri, PATCH_SIZE, spatial_scale=scale)

                        # Match to GT boxes for this image
                        gt_roi = torch.zeros_like(samp_roi)
                        tgt = targets[0]  # first image's targets
                        if len(tgt["boxes"]) > 0:
                            ious = box_iou(samp_anchors, tgt["boxes"])  # (k, M)
                            best_iou, best_idx = ious.max(dim=1)  # (k,)
                            for j in range(k):
                                if best_iou[j] > 0.3:  # matched
                                    gt_box = tgt["boxes"][best_idx[j]:best_idx[j]+1]
                                    gt_ri = torch.cat([torch.zeros(1, 1, device=DEV), gt_box], dim=1)
                                    gt_roi[j:j+1] = roi_align(feat_map, gt_ri, PATCH_SIZE, spatial_scale=scale)

                        # FFT reward
                        rewards = fourier_box_reward(samp_roi, gt_roi)  # (k,)

                        all_rewards.append(rewards)
                        all_log_probs.append(torch.log(old_probs[sampled] + 1e-6) - torch.log(probs[sampled] + 1e-6))

                        all_log_probs[-1] = -all_log_probs[-1]  # log(new/old) for ratio
                        # Actually we want -log_prob for REINFORCE minimization
                        all_log_probs[-1] = torch.log(probs[sampled] + 1e-6)  # use new policy log_prob directly

                    if all_rewards:
                        cat_rew = torch.cat(all_rewards)
                        cat_lp = torch.cat(all_log_probs)

                        if reward_baseline is None:
                            reward_baseline = cat_rew.mean().item()
                        else:
                            reward_baseline = 0.9 * reward_baseline + 0.1 * cat_rew.mean().item()

                        advantages = cat_rew - reward_baseline
                        rl_loss = -(advantages.detach() * cat_lp).mean()
                        avg_reward = cat_rew.mean().item()

                loss = det_loss + alpha * rl_loss
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

                total_det += det_loss.item()
                total_rl += rl_loss.item()

            ep_m = evaluate(model, val_loader)
            row = {"epoch": epoch, "val_ap50": ep_m["ap50"], "val_ap75": ep_m["ap75"]}
            history.append(row)
            print(f"  e{epoch}: AP50={ep_m['ap50']:.4f} det={total_det:.1f} rl={total_rl:.3f} r={avg_reward:.3f}")
            if ep_m["ap50"] > best_ap50:
                best_ap50 = ep_m["ap50"]

        hk_fpn.remove()

        ep_m.update({"run_name": run_name, "alpha": alpha,
                     "epochs": EPOCHS, "seed": 42,
                     "best_ap50": best_ap50, "history": history, "git_hash": GIT})
        save_json(ep_m, run_dir / "eval_metrics.json")
        all_r.append(ep_m)
        print(f"  DONE a{alpha}: AP50={ep_m['ap50']:.4f} AP75={ep_m['ap75']:.4f}")

    print("\n## Plan 2.37 Results")
    for r in all_r:
        print(f"  a{r['alpha']}: AP50={r['ap50']:.4f} AP75={r['ap75']:.4f}")


if __name__ == "__main__":
    main()
