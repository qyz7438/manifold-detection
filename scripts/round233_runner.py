"""Plan 3.6 Direction 1: RPN sampling + REINFORCE with spectral quality reward.

RPN stochastic sampling → spectral quality → policy gradient update.
"""
import sys, json, subprocess
from pathlib import Path
import torch
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
SEEDS = [42]
K_SAMPLES = 20          # proposals to sample per FPN level
ALPHAS = [0.1, 0.5]     # RL loss weight
EPOCHS = 5


def spectral_quality(roi_features):
    """Fixed heuristic: HF energy + entropy + phase coherence → quality score."""
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
        "train": {"batch_size": 1},  # batch_size=1 for RL sampling simplicity
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
        for seed in SEEDS:
            run_name = f"round233_d1_rl_{alpha}_s{seed}"
            set_seed(seed)

            model = build_model().to(DEV)
            ckpt = torch.load(CKPT, map_location=DEV)
            model.load_state_dict(ckpt["model"])

            # Freeze backbone + box_head; train RPN
            freeze_except(model, [model.rpn.head, model.rpn.anchor_generator])

            train_loader, val_loader = build_loaders()
            params = [p for p in model.parameters() if p.requires_grad]
            opt = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)
            run_dir = ensure_run_dir(run_name)
            history = []
            best_ap50 = -1.0
            reward_baseline = None  # EMA baseline for variance reduction

            for epoch in range(1, EPOCHS + 1):
                model.train()
                total_rl_loss = 0.0
                total_det_loss = 0.0
                avg_reward = 0.0

                for images, targets in tqdm(train_loader, desc=f"{run_name} e{epoch}"):
                    images = [img.to(DEV) for img in images]
                    targets = [{k: v.to(DEV) for k, v in t.items()} for t in targets]

                    # --- RL: hook RPN objectness + ROI features during detection forward ---
                    images_t, _ = model.transform(images, None)
                    rpn_objs = {}
                    roi_feats = {}

                    def rpn_hook(module, inp, out):
                        rpn_objs["x"] = [o.detach() for o in out[0]]  # list of obj_logits per level

                    def box_hook(module, inp):
                        roi_feats["x"] = inp[0]  # (N, 256, 7, 7)

                    hk1 = model.rpn.head.register_forward_hook(rpn_hook)
                    hk2 = model.roi_heads.box_head.register_forward_pre_hook(box_hook)

                    ld = model(images, targets)
                    det_loss = sum(ld.values())

                    hk1.remove(); hk2.remove()

                    # REINFORCE: gradient through sampled RPN objectness
                    rl_loss = torch.tensor(0.0, device=DEV)
                    all_rewards = torch.tensor([], device=DEV)
                    roi = roi_feats.get("x")
                    obj_list = rpn_objs.get("x")

                    if roi is not None and obj_list is not None and roi.shape[0] > 0:
                        rewards = spectral_quality(roi)  # (N,) spectral quality
                        all_rewards = rewards

                        # Re-run RPN head to get objectness with grad
                        fpn_feat = model.backbone(images_t.tensors)
                        fpn_keys = [k for k in fpn_feat.keys() if k != "pool"]
                        fpn_list = [fpn_feat[k] for k in fpn_keys]
                        obj_logits_g, _ = model.rpn.head(fpn_list)  # with grad

                        # Sample per level: softmax → categorical sample → log_prob
                        for li, obj_l in enumerate(obj_logits_g):
                            obj_flat = obj_l.flatten()  # (B*A*H*W,)
                            probs = torch.softmax(obj_flat, dim=0)
                            k = min(K_SAMPLES, len(obj_flat))
                            sampled = torch.multinomial(probs, k, replacement=False)
                            log_probs = torch.log(probs[sampled] + 1e-6)
                            # reward: mean spectral quality of all ROIs (since we can't map sampled idx to specific ROI)
                            bl = reward_baseline if reward_baseline is not None else rewards.mean().item()
                            rl_loss -= (rewards.mean().item() - bl) * log_probs.mean()

                    loss = det_loss + alpha * rl_loss
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()

                    total_det_loss += det_loss.item()
                    total_rl_loss += rl_loss.item()
                    if len(all_rewards) > 0:
                        avg_reward = all_rewards.mean().item()
                        if reward_baseline is None:
                            reward_baseline = avg_reward
                        else:
                            reward_baseline = 0.9 * reward_baseline + 0.1 * avg_reward

                ep_m = evaluate(model, val_loader)
                row = {"epoch": epoch, "val_ap50": ep_m["ap50"], "val_ap75": ep_m["ap75"],
                       "det_loss": total_det_loss, "rl_loss": total_rl_loss,
                       "avg_reward": avg_reward}
                history.append(row)
                print(f"  e{epoch}: AP50={ep_m['ap50']:.4f} det={total_det_loss:.2f} "
                      f"rl={total_rl_loss:.4f} avg_r={avg_reward:.3f} baseline={reward_baseline:.3f}")
                if ep_m["ap50"] > best_ap50:
                    best_ap50 = ep_m["ap50"]

            ep_m.update({"run_name": run_name, "alpha": alpha,
                         "epochs": EPOCHS, "seed": seed,
                         "best_ap50": best_ap50, "history": history, "git_hash": GIT})
            save_json(ep_m, run_dir / "eval_metrics.json")
            all_r.append(ep_m)
            print(f"  DONE a{alpha}: AP50={ep_m['ap50']:.4f} AP75={ep_m['ap75']:.4f}")

    print("\n## Plan 3.6 D1 Results")
    for r in all_r:
        print(f"  a{r['alpha']}: AP50={r['ap50']:.4f} AP75={r['ap75']:.4f}")


if __name__ == "__main__":
    main()
