"""Plan 3.5: Spectral quality reward cross-validation.

Phase 1: Convergence baselines for PF+ResNet50 and VOC+MobV3.
Phase 2: FFT quality → bbox loss weight on both combos.
"""
import sys, json, subprocess
from pathlib import Path
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, "E:/CLIproject/RLimage")
from spectral_detection_posttrain.datasets import build_penn_fudan_loaders
from spectral_detection_posttrain.datasets.voc_detection import build_voc_detection_loaders
from spectral_detection_posttrain.eval.detection_metrics import evaluate_detection_predictions
from spectral_detection_posttrain.models import build_detector
from spectral_detection_posttrain.utils.io import save_json, ensure_run_dir
from spectral_detection_posttrain.utils.seed import set_seed

GIT = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS = [42, 123, 456]
ALPHAS = [0.1, 0.5, 1.0]
EPOCHS = 15

COMBOS = [
    {
        "key": "pf_r50", "dataset": "penn_fudan",
        "model_name": "fasterrcnn_resnet50_fpn", "num_classes": 2, "max_size": 320,
        "ckpt_dir": "runs/round35_pf_r50_conv_s42",
    },
    {
        "key": "voc_mob", "dataset": "voc",
        "model_name": "fasterrcnn_mobilenet_v3_large_320_fpn", "num_classes": 4, "max_size": 480,
        "ckpt_dir": "runs/round35_voc_mob_conv_s42",
    },
    {
        "key": "vocfull_mob", "dataset": "voc_full",
        "model_name": "fasterrcnn_mobilenet_v3_large_320_fpn", "num_classes": 21, "max_size": 480,
        "ckpt_dir": "runs/round236_vocfull_conv_s42",
    },
]


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


def build_loaders(combo, seed):
    cfg = {"data": {"root": "./data", "max_size": combo["max_size"], "train_fraction": 0.8, "num_workers": 0},
           "train": {"batch_size": 2}}
    if combo["dataset"] == "voc":
        cfg["data"].update({"root": "E:/pythonProject1", "year": "2012", "download": False,
                            "classes": ["person", "car", "dog"], "train_set": "train", "val_set": "val"})
        return build_voc_detection_loaders(cfg, limit_train=300, limit_val=150)
    if combo["dataset"] == "voc_full":
        VOC20 = ["aeroplane","bicycle","bird","boat","bottle","bus","car","cat","chair",
                 "cow","diningtable","dog","horse","motorbike","person","pottedplant",
                 "sheep","sofa","train","tvmonitor"]
        cfg["data"].update({"root": "E:/pythonProject1", "year": "2012", "download": False,
                            "classes": VOC20, "train_set": "train", "val_set": "val"})
        return build_voc_detection_loaders(cfg, limit_train=500, limit_val=200)
    return build_penn_fudan_loaders(cfg)


def build_model(combo):
    cfg = {"model": {"name": combo["model_name"], "model_name": combo["model_name"],
                     "pretrained": True, "num_classes": combo["num_classes"],
                     "min_size": 320, "max_size": combo["max_size"]}}
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

    for combo in COMBOS:
        ckpt_path = Path(combo["ckpt_dir"]) / "checkpoint_best.pth"
        if not ckpt_path.exists():
            print(f"SKIP {combo['key']}: no checkpoint at {ckpt_path}")
            continue

        for alpha in ALPHAS:
            for seed in SEEDS:
                run_name = f"round35_{combo['key']}_q_a{alpha}_s{seed}"
                set_seed(seed)

                model = build_model(combo).to(DEV)
                ckpt = torch.load(str(ckpt_path), map_location=DEV)
                model.load_state_dict(ckpt["model"])

                freeze_except(model, [model.rpn.head, model.roi_heads.box_head,
                                      model.roi_heads.box_predictor])

                train_loader, val_loader = build_loaders(combo, seed)
                params = [p for p in model.parameters() if p.requires_grad]
                opt = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)
                run_dir = ensure_run_dir(run_name)
                history = []
                best_ap50 = -1.0

                roi_cache = {}
                def bh_pre_hook(module, inp):
                    roi_cache["x"] = inp[0]
                hk = model.roi_heads.box_head.register_forward_pre_hook(bh_pre_hook)

                for epoch in range(1, EPOCHS + 1):
                    model.train()
                    for images, targets in tqdm(train_loader, desc=f"{run_name} e{epoch}"):
                        images = [img.to(DEV) for img in images]
                        targets = [{k: v.to(DEV) for k, v in t.items()} for t in targets]
                        roi_cache.clear()

                        ld = model(images, targets)
                        det_loss = sum(ld.values())

                        roi_feats = roi_cache.get("x")
                        if roi_feats is not None and roi_feats.shape[0] > 0:
                            quality = spectral_quality(roi_feats)
                            box_reg = ld.get("loss_box_reg",
                                             torch.tensor(0.0, device=DEV))
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
                    if ep_m["ap50"] > best_ap50:
                        best_ap50 = ep_m["ap50"]

                hk.remove()
                ep_m.update({"run_name": run_name, "combo": combo["key"], "alpha": alpha,
                             "epochs": EPOCHS, "seed": seed,
                             "best_ap50": best_ap50, "history": history, "git_hash": GIT})
                save_json(ep_m, run_dir / "eval_metrics.json")
                all_r.append(ep_m)
                print(f"  {combo['key']} a{alpha} s{seed}: AP50={ep_m['ap50']:.4f} AP75={ep_m['ap75']:.4f}")

    print("\n## Plan 3.5 Results")
    for r in all_r:
        print(f"  {r['combo']} a{r['alpha']} s{r['seed']}: AP50={r['ap50']:.4f} AP75={r['ap75']:.4f}")


if __name__ == "__main__":
    main()
