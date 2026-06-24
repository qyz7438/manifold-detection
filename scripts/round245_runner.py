"""Plan 2.45: Inverted Pixel FFT Soft Reweighting.

Same as 2.31 but quality = 1 - pixel_fft_quality. Hypothesis: raw pixel FFT
quality (HF+entropy) correlates negatively with IoU, so inverted quality
should give better reweighting.
"""
import sys, json, subprocess
from pathlib import Path
import torch, torch.nn as nn, torchvision
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
ALPHAS = [0.1, 0.5]
EPOCHS = 10
PIXEL_SIZE = 64


def pixel_fft_quality(pixel_patches):
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

    for alpha in ALPHAS:
        run_name = f"round245_inv_a{alpha}_s42"
        set_seed(42)

        model = build_model().to(DEV)
        ckpt = torch.load(CKPT, map_location=DEV)
        model.load_state_dict(ckpt["model"])

        freeze_except(model, [model.rpn.head, model.roi_heads.box_head,
                      model.roi_heads.box_predictor])

        train_loader, val_loader = build_loaders()
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
            proposal_cache["p"] = out[0]

        hk_fpn = model.backbone.register_forward_hook(fpn_hook)
        hk_rpn = model.rpn.register_forward_hook(rpn_hook)

        for epoch in range(1, EPOCHS + 1):
            model.train()
            total_det, total_spec = 0.0, 0.0
            avg_q = 0.0

            for images, targets in tqdm(train_loader, desc=f"{run_name} e{epoch}"):
                images_dev = [img.to(DEV) for img in images]
                targets_t = [{k: v.to(DEV) for k, v in t.items()} for t in targets]
                fpn_cache.clear()
                proposal_cache.clear()

                ld = model(images_dev, targets_t)
                det_loss = sum(ld.values())
                box_reg = ld.get("loss_box_reg", torch.tensor(0.0, device=DEV))

                proposals = proposal_cache.get("p")
                spec_loss = torch.tensor(0.0, device=DEV)

                if proposals is not None:
                    pc = torch.cat(proposals, dim=0)[:256]
                    P = pc.shape[0]
                    if P > 0:
                        # Crop pixel patches from original images
                        npi = [p.shape[0] for p in proposals]
                        ii = torch.cat([torch.full((n,), i, dtype=torch.long) for i, n in enumerate(npi)], dim=0)[:P]
                        patches = []
                        for idx in range(P):
                            img_i = ii[idx].item()
                            img = images[img_i]
                            box = pc[idx]
                            x1, y1, x2, y2 = box.round().long().clamp(min=0)
                            x1, x2 = max(0, min(x1, img.shape[-1]-1)), max(x1+1, min(x2, img.shape[-1]))
                            y1, y2 = max(0, min(y1, img.shape[-2]-1)), max(y1+1, min(y2, img.shape[-2]))
                            crop = img[:, y1:y2, x1:x2]
                            if crop.shape[-1] >= 4 and crop.shape[-2] >= 4:
                                crop = F.interpolate(crop.unsqueeze(0).float(), size=(PIXEL_SIZE, PIXEL_SIZE), mode='bilinear', align_corners=False).squeeze(0)
                                patches.append(crop)
                            else:
                                patches.append(torch.zeros(3, PIXEL_SIZE, PIXEL_SIZE))

                        if patches:
                            pb = torch.stack(patches).to(DEV)
                            q = pixel_fft_quality(pb)
                            q_inv = 1.0 - q  # KEY: inverted quality
                            spec_loss = (q_inv * box_reg).mean()
                            avg_q = q_inv.mean().item()

                loss = det_loss + alpha * spec_loss
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

                total_det += det_loss.item()
                total_spec += spec_loss.item()

            ep_m = evaluate(model, val_loader)
            row = {"epoch": epoch, "val_ap50": ep_m["ap50"], "val_ap75": ep_m["ap75"]}
            history.append(row)
            print(f"  e{epoch}: AP50={ep_m['ap50']:.4f} det={total_det:.1f} spec={total_spec:.3f} q_inv={avg_q:.4f}")
            if ep_m["ap50"] > best_ap50:
                best_ap50 = ep_m["ap50"]

        hk_fpn.remove(); hk_rpn.remove()

        ep_m.update({"run_name": run_name, "alpha": alpha,
                     "epochs": EPOCHS, "seed": 42,
                     "best_ap50": best_ap50, "history": history, "git_hash": GIT})
        save_json(ep_m, run_dir / "eval_metrics.json")
        all_r.append(ep_m)
        print(f"  DONE a{alpha}: AP50={ep_m['ap50']:.4f} AP75={ep_m['ap75']:.4f}")

    print("\n## Plan 2.45 Inverted Quality Results")
    for r in all_r:
        print(f"  a{r['alpha']}: AP50={r['ap50']:.4f} AP75={r['ap75']:.4f}")


if __name__ == "__main__":
    main()
