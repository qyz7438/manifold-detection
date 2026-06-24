"""Plan 2.31: Per-box frequency quality scorer as human judge.

Scorer takes ROI features → FFT → predicts IoU quality.
Reward = quality_pred * box_regression_loss (more weight = learn more from good boxes).
No structure modification to the detector.
"""
import sys, json, subprocess
from pathlib import Path
import torch, torch.nn as nn, torchvision
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
SEEDS = [42, 123, 456]
EPOCHS = 15
ALPHAS = [0.1, 0.5, 1.0]


def spectral_quality(roi_features):
    """Fixed heuristic: ROI spectral quality score (0-1). No training.

    High quality = structured spectrum (low entropy) + strong edges (high HF) + coherent phase.
    """
    N = roi_features.shape[0]
    # 2D FFT on spatial dims of each ROI
    fft = torch.fft.rfft2(roi_features.float(), dim=(-2, -1), norm="ortho")  # (N, 256, 7, 4)
    mag = torch.abs(fft).mean(dim=1)  # (N, 7, 4) avg magnitude over channels

    # High-frequency energy ratio (edges → good localization)
    mag_flat = mag.flatten(1)  # (N, 28)
    total = mag_flat.sum(dim=1, keepdim=True).clamp_min(1e-6)
    hf_ratio = mag_flat[:, 14:].sum(dim=1) / total.squeeze(1)  # (N,)

    # Spectral entropy (low entropy = structured = good)
    mag_norm = mag_flat / total
    entropy = -(mag_norm * torch.log(mag_norm + 1e-6)).sum(dim=1)  # (N,)
    max_entropy = torch.log(torch.tensor(28.0, device=roi_features.device))
    entropy_norm = 1.0 - entropy / max_entropy  # 0=flat, 1=sharp

    # Phase coherence: low variance = coherent edges
    pha = torch.angle(fft + 1e-6)
    pha_var = pha.std(dim=(1, 2, 3)).clamp_max(1.0)  # (N,)
    phase_coherence = 1.0 - pha_var  # 0=chaotic, 1=coherent

    # Combine: weighted average
    quality = 0.3 * hf_ratio + 0.4 * entropy_norm + 0.3 * phase_coherence
    return quality.clamp(0.0, 1.0)  # (N,)


def build_loaders(seed):
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

    for alpha in ALPHAS:
        for seed in SEEDS:
            run_name = f"round231_v6_quality_a{alpha}_s{seed}"
            set_seed(seed)

            model = build_model().to(DEV)
            ckpt = torch.load(CKPT, map_location=DEV)
            model.load_state_dict(ckpt["model"])

            # Train RPN + box_head; freeze backbone only
            freeze_except(model, [model.rpn.head, model.roi_heads.box_head,
                          model.roi_heads.box_predictor])

            train_loader, val_loader = build_loaders(seed)
            params = [p for p in model.parameters() if p.requires_grad]
            opt = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)
            run_dir = ensure_run_dir(run_name)
            history = []
            best_ap50 = -1.0

            fpn_cache = {}
            proposal_cache = {}

            def fpn_hook(module, inp, out):
                fpn_cache["f"] = {k: out[k] for k in out if k != "pool"}

            def rpn_hook(module, inp, out):
                proposal_cache["p"] = out[0]  # list of proposal tensors per image

            hk_fpn = model.backbone.register_forward_hook(fpn_hook)
            hk_rpn = model.rpn.register_forward_hook(rpn_hook)

            for epoch in range(1, EPOCHS + 1):
                model.train()
                for images, targets in tqdm(train_loader, desc=f"{run_name} e{epoch}"):
                    images = [img.to(DEV) for img in images]
                    targets = [{k: v.to(DEV) for k, v in t.items()} for t in targets]
                    fpn_cache.clear()
                    proposal_cache.clear()

                    ld = model(images, targets)
                    det_loss = sum(ld.values())

                    # Spectral quality using 14×14 ROI Align (not box_head's 7×7)
                    fpn_feats = fpn_cache.get("f")
                    proposals = proposal_cache.get("p")
                    quality = torch.tensor([], device=DEV)

                    if fpn_feats is not None and proposals is not None:
                        fpn_keys = sorted(fpn_feats.keys(), key=int)
                        # Collect proposals from all images, sample up to 256
                        all_props = []
                        for p in proposals:
                            all_props.append(p)
                        if all_props:
                            prop_boxes = torch.cat(all_props, dim=0)[:256]  # (P, 4)
                            P = prop_boxes.shape[0]
                            if P > 0:
                                # Assign to FPN levels and ROI Align with 14×14
                                w = prop_boxes[:, 2] - prop_boxes[:, 0]
                                h = prop_boxes[:, 3] - prop_boxes[:, 1]
                                area = (w * h).clamp_min(1)
                                lvl = torch.floor(torch.log2(torch.sqrt(area) / 224) + 4).long().clamp(2, 5)
                                roi14_list = []
                                for i in range(P):
                                    ki = min(len(fpn_keys)-1, max(0, lvl[i].item()-2))
                                    feat = fpn_feats[fpn_keys[ki]]
                                    bx = prop_boxes[i:i+1]
                                    ri = torch.cat([torch.zeros(1,1,device=DEV), bx], dim=1)
                                    scale = 1.0/(2**(int(fpn_keys[ki])+2))
                                    r14 = torchvision.ops.roi_align(feat, ri, output_size=14, spatial_scale=scale)
                                    roi14_list.append(r14)
                                if roi14_list:
                                    roi14 = torch.cat(roi14_list, dim=0)
                                    quality = spectral_quality(roi14)

                    if quality.numel() > 0:
                        box_reg = ld.get("loss_box_reg", torch.tensor(0.0, device=DEV))
                        rew = (quality * box_reg).mean()
                    else:
                        rew = torch.tensor(0.0, device=DEV)

                    loss = det_loss + alpha * rew
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()

                ep_m = evaluate(model, val_loader)
                row = {"epoch": epoch, "val_ap50": ep_m["ap50"], "val_ap75": ep_m["ap75"]}
                history.append(row)
                print(f"  e{epoch} a={alpha}: AP50={ep_m['ap50']:.4f} AP75={ep_m['ap75']:.4f}")
                if ep_m["ap50"] > best_ap50:
                    best_ap50 = ep_m["ap50"]

            hk_fpn.remove()
            hk_rpn.remove()
            ep_m.update({"run_name": run_name, "alpha": alpha,
                         "epochs": EPOCHS, "seed": seed,
                         "best_ap50": best_ap50, "history": history, "git_hash": GIT})
            save_json(ep_m, run_dir / "eval_metrics.json")
            all_r.append(ep_m)
            print(f"  DONE a{alpha} s{seed}: AP50={ep_m['ap50']:.4f} AP75={ep_m['ap75']:.4f}")

    print("\n## Plan 2.31 Results")
    for r in all_r:
        print(f"  a{r['alpha']} s{r['seed']}: AP50={r['ap50']:.4f} AP75={r['ap75']:.4f}")


if __name__ == "__main__":
    main()
