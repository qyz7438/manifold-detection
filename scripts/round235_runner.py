"""Plan 2.35: PPO for RPN with spectral quality reward.

RPN stochastic sampling → spectral quality → PPO clipped objective.
No bbox exploration (too noisy for 136-image dataset).
"""
import sys, json, subprocess
from pathlib import Path
import torch
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
K_SAMPLES = 20
CLIP_EPS = 0.2
ALPHAS = [0.1, 0.5]
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


def assign_level(boxes):
    w, h = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
    area = (w * h).clamp_min(1)
    return torch.floor(torch.log2(torch.sqrt(area) / 224) + 4).long().clamp(2, 5)


def main():
    all_r = []

    for alpha in ALPHAS:
        run_name = f"round235_d3_ppo_a{alpha}_s42"
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
        reward_baseline = 0.5

        fpn_cache = {}
        roi_cache = {}

        def fpn_hook(module, inp, out):
            fpn_cache["f"] = {k: out[k] for k in out if k != "pool"}

        def roi_hook(module, inp):
            roi_cache["x"] = inp[0]

        hk_fpn = model.backbone.register_forward_hook(fpn_hook)
        hk_roi = model.roi_heads.box_head.register_forward_pre_hook(roi_hook)

        for epoch in range(1, EPOCHS + 1):
            model.train()
            total_ppo = 0.0
            total_det = 0.0
            avg_reward = 0.0

            for images, targets in tqdm(train_loader, desc=f"{run_name} e{epoch}"):
                images = [img.to(DEV) for img in images]
                targets = [{k: v.to(DEV) for k, v in t.items()} for t in targets]
                roi_cache.clear()
                fpn_cache.clear()

                # 1. Detection forward
                ld = model(images, targets)
                det_loss = sum(ld.values())

                # 2. PPO: per-proposal ROI spectral quality → individual rewards
                fpn_feats = fpn_cache.get("f")
                roi_feats = roi_cache.get("x")
                ppo_loss = torch.tensor(0.0, device=DEV)

                if fpn_feats is not None and roi_feats is not None and roi_feats.shape[0] > 0:
                    fpn_keys = sorted(fpn_feats.keys(), key=int)
                    fpn_levels = [fpn_feats[k] for k in fpn_keys]

                    # Per-proposal spectral quality rewards (N independent scores)
                    rewards = spectral_quality(roi_feats)  # (N,)

                    # Old policy (no grad)
                    with torch.no_grad():
                        old_obj, _ = model.rpn.head(fpn_levels)
                    # New policy (with grad)
                    new_obj, _ = model.rpn.head(fpn_levels)

                    all_old_lp, all_new_lp, all_rewards = [], [], []

                    for li in range(len(fpn_keys)):
                        nf = new_obj[li].flatten()
                        probs = torch.softmax(nf, dim=0)
                        op = torch.softmax(old_obj[li].flatten().detach(), dim=0)

                        k = min(K_SAMPLES, len(probs))
                        sampled = torch.multinomial(probs, k, replacement=False)

                        # Each sampled position gets reward from a random proposal
                        ri = torch.randint(0, len(rewards), (k,), device=DEV)
                        all_rewards.append(rewards[ri])
                        all_old_lp.append(torch.log(op[sampled] + 1e-6))
                        all_new_lp.append(torch.log(probs[sampled] + 1e-6))

                    cat_rew = torch.cat(all_rewards)
                    cat_old = torch.cat(all_old_lp)
                    cat_new = torch.cat(all_new_lp)
                    ratio = torch.exp(cat_new - cat_old)
                    advantages = cat_rew.detach() - reward_baseline
                    clipped = ratio.clamp(1.0 - CLIP_EPS, 1.0 + CLIP_EPS)
                    ppo_loss = -torch.min(ratio * advantages, clipped * advantages).mean()
                    avg_reward = cat_rew.mean().item()
                    reward_baseline = 0.9 * reward_baseline + 0.1 * avg_reward

                loss = det_loss + alpha * ppo_loss
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

                total_det += det_loss.item()
                total_ppo += ppo_loss.item()

            ep_m = evaluate(model, val_loader)
            row = {"epoch": epoch, "val_ap50": ep_m["ap50"], "val_ap75": ep_m["ap75"]}
            history.append(row)
            print(f"  e{epoch}: AP50={ep_m['ap50']:.4f} ppo={total_ppo:.3f} r={avg_reward:.3f}")
            if ep_m["ap50"] > best_ap50:
                best_ap50 = ep_m["ap50"]

        hk_fpn.remove()
        hk_roi.remove()

        ep_m.update({"run_name": run_name, "alpha": alpha,
                     "epochs": EPOCHS, "seed": 42,
                     "best_ap50": best_ap50, "history": history, "git_hash": GIT})
        save_json(ep_m, run_dir / "eval_metrics.json")
        all_r.append(ep_m)
        print(f"  DONE a{alpha}: AP50={ep_m['ap50']:.4f} AP75={ep_m['ap75']:.4f}")

    print("\n## Plan 2.35 Results")
    for r in all_r:
        print(f"  a{r['alpha']}: AP50={r['ap50']:.4f} AP75={r['ap75']:.4f}")


if __name__ == "__main__":
    main()
