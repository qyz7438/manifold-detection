"""Plan 2.32: Spectral consistency reward.

Scorer judges ROI quality by spectral similarity to class prototype.
Good ROIs have spectra similar to the running class average → higher weight.
RoI features (before box_head) → FFT → compare with class prototype → reward.
"""
import sys, json, subprocess
from pathlib import Path
import torch, torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torchvision.ops import box_iou, nms

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
LAMBDAS = [0.01, 0.05, 0.1]


def extract_fft_signature(roi_features):
    """Extract compact FFT signature from ROI features (N, 256, 7, 7)."""
    fft = torch.fft.rfft2(roi_features.float(), dim=(-2, -1), norm="ortho")
    mag = torch.abs(fft).mean(dim=1)  # (N, 7, 4) avg over channels
    mag_flat = mag.flatten(1)  # (N, 28)
    mag_flat = F.normalize(mag_flat, dim=1)  # unit norm
    return mag_flat  # (N, 28)


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

    # Class prototype: exponentially moving average of positive ROI FFT signatures
    proto = None  # (28,) running average

    for lam in LAMBDAS:
        for seed in SEEDS:
            run_name = f"round232_v7_spectral_a{lam}_s{seed}"
            set_seed(seed)

            model = build_model().to(DEV)
            ckpt = torch.load(CKPT, map_location=DEV)
            model.load_state_dict(ckpt["model"])

            freeze_except(model, [model.rpn.head, model.roi_heads.box_head,
                          model.roi_heads.box_predictor])

            train_loader, val_loader = build_loaders(seed)
            params = [p for p in model.parameters() if p.requires_grad]
            opt = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)
            run_dir = ensure_run_dir(run_name)
            history = []
            best_ap50 = -1.0
            proto = None  # reset prototype per run

            roi_features_cache = {}

            def box_head_pre_hook(module, inp):
                roi_features_cache["x"] = inp[0]

            hk = model.roi_heads.box_head.register_forward_pre_hook(box_head_pre_hook)

            for epoch in range(1, EPOCHS + 1):
                model.train()
                total_spec_loss = 0.0

                for images, targets in tqdm(train_loader, desc=f"{run_name} e{epoch}"):
                    images = [img.to(DEV) for img in images]
                    targets = [{k: v.to(DEV) for k, v in t.items()} for t in targets]
                    roi_features_cache.clear()

                    ld = model(images, targets)
                    det_loss = sum(ld.values())

                    roi_feats = roi_features_cache.get("x")
                    spec_loss = torch.tensor(0.0, device=DEV)

                    if roi_feats is not None and roi_feats.shape[0] > 0:
                        sig = extract_fft_signature(roi_feats)  # (N, 28)

                        # Initialize or update class prototype (EMA)
                        if proto is None:
                            proto = sig.mean(dim=0)
                        else:
                            proto = 0.9 * proto + 0.1 * sig.mean(dim=0)

                        # Quality reward: similarity to prototype
                        similarity = (sig * proto.unsqueeze(0)).sum(dim=1)  # (N,) cos_sim
                        quality = (similarity + 1.0) / 2.0  # map [-1,1] to [0,1]

                        # Reward: weight box regression by spectral quality
                        box_reg = ld.get("loss_box_reg",
                                         torch.tensor(0.0, device=DEV))
                        spec_loss = (quality.detach() * box_reg).mean()

                    loss = det_loss + lam * spec_loss
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
                    total_spec_loss += spec_loss.item()

                ep_m = evaluate(model, val_loader)
                row = {"epoch": epoch, "val_ap50": ep_m["ap50"], "val_ap75": ep_m["ap75"],
                       "spec_loss": total_spec_loss}
                history.append(row)
                print(f"  e{epoch} l={lam}: AP50={ep_m['ap50']:.4f} spec={total_spec_loss:.4f}")
                if ep_m["ap50"] > best_ap50:
                    best_ap50 = ep_m["ap50"]

            hk.remove()
            ep_m.update({"run_name": run_name, "lambda": lam,
                         "epochs": EPOCHS, "seed": seed,
                         "best_ap50": best_ap50, "history": history, "git_hash": GIT})
            save_json(ep_m, run_dir / "eval_metrics.json")
            all_r.append(ep_m)
            print(f"  DONE l={lam} s{seed}: AP50={ep_m['ap50']:.4f} AP75={ep_m['ap75']:.4f}")

    print("\n## Plan 2.32 Results")
    for r in all_r:
        print(f"  l{r['lambda']} s{r['seed']}: AP50={r['ap50']:.4f} AP75={r['ap75']:.4f}")


if __name__ == "__main__":
    main()
