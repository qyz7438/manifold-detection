"""Plan 2.20: A+C post-training on PF+ResNet50 and VOC+MobV3.

Approach A: weak gate (0.1), AFM-only, 2 epochs
Approach C: feature constraint (MSE in/out), AFM-only, 2 epochs
"""
import sys, json, subprocess
from pathlib import Path
import torch, torch.nn as nn
from tqdm import tqdm

sys.path.insert(0, "E:/CLIproject/RLimage")
from spectral_detection_posttrain.datasets import build_penn_fudan_loaders
from spectral_detection_posttrain.datasets.voc_detection import build_voc_detection_loaders
from spectral_detection_posttrain.eval.detection_metrics import evaluate_detection_predictions
from spectral_detection_posttrain.models import build_detector
from spectral_detection_posttrain.models.micro_afm import MPLSegAFMBlock
from spectral_detection_posttrain.utils.io import load_checkpoint, save_json, ensure_run_dir
from spectral_detection_posttrain.utils.seed import resolve_device, set_seed

GIT = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS = [42, 123, 456]

COMBOS = [
    {
        "label": "r50_pf",
        "ckpt": "runs/round219_r50_pf_mid06_s42/checkpoint_last.pth",
        "model_name": "fasterrcnn_resnet50_fpn",
        "dataset": "penn_fudan",
        "num_classes": 2,
        "max_size": 480,
    },
    {
        "label": "mob_voc",
        "ckpt": "runs/round219_voc_mob_mid06_s42/checkpoint_last.pth",
        "model_name": "fasterrcnn_mobilenet_v3_large_320_fpn",
        "dataset": "voc",
        "num_classes": 4,
        "max_size": 480,
    },
]


def freeze_all(model):
    for p in model.parameters():
        p.requires_grad = False


def build_loaders(combo, seed):
    cfg = {
        "seed": seed, "device": DEV,
        "data": {"root": "./data", "download": True,
                 "max_size": combo["max_size"], "train_fraction": 0.8, "num_workers": 0},
        "train": {"batch_size": 2},
    }
    if combo["dataset"] == "voc":
        cfg["data"].update({"root": "E:/pythonProject1", "year": "2012", "download": False,
                            "classes": ["person", "car", "dog"],
                            "train_set": "train", "val_set": "val"})
        cfg["data"]["max_size"] = 480
        return build_voc_detection_loaders(cfg, limit_train=300, limit_val=150)
    return build_penn_fudan_loaders(cfg)


def build_model(combo):
    cfg = {
        "model": {
            "name": combo["model_name"],
            "model_name": combo["model_name"],
            "pretrained": True,
            "num_classes": combo["num_classes"],
            "min_size": 320,
            "max_size": combo["max_size"],
            "afm_channels": 256,
            "afm_type": "mplseg_mid",
        }
    }
    return build_detector(cfg)


@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    preds, targs = [], []
    for images, targets in val_loader:
        out = model([img.to(DEV) for img in images])
        preds.extend([{k: v.cpu() for k, v in o.items()} for o in out])
        targs.extend([{k: v.cpu() for k, v in t.items()} for t in targets])
    return evaluate_detection_predictions(preds, targs, iou_threshold=0.5, score_threshold=0.05)


def train_a(combo, seed, run_name):
    """Approach A: weak gate, AFM-only."""
    set_seed(seed)
    model = build_model(combo).to(DEV)
    load_checkpoint(model, combo["ckpt"], DEV)

    afm = model.roi_heads.box_head.afm
    in_ch = afm.mp[0].in_channels
    new_afm = MPLSegAFMBlock(in_ch=in_ch, gate_strength=0.1).to(DEV)
    new_afm.load_state_dict(afm.state_dict(), strict=False)
    model.roi_heads.box_head.afm = new_afm

    freeze_all(model)
    for p in new_afm.parameters():
        p.requires_grad = True

    train_loader, val_loader = build_loaders(combo, seed)
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                          lr=0.001, momentum=0.9, weight_decay=0.0005)
    run_dir = ensure_run_dir(run_name)

    for epoch in range(1, 3):
        model.train()
        for images, targets in tqdm(train_loader, desc=f"{run_name}"):
            images = [img.to(DEV) for img in images]
            targets = [{k: v.to(DEV) for k, v in t.items()} for t in targets]
            ld = model(images, targets)
            loss = sum(ld.values())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    m = evaluate(model, val_loader)
    m.update({"run_name": run_name, "approach": "A_weak_gate", "git_hash": GIT,
              "model": combo["model_name"], "dataset": combo["dataset"]})
    save_json(m, run_dir / "eval_metrics.json")
    return m


def train_c(combo, seed, run_name):
    """Approach C: feature constraint, AFM-only."""
    set_seed(seed)
    model = build_model(combo).to(DEV)
    load_checkpoint(model, combo["ckpt"], DEV)
    afm = model.roi_heads.box_head.afm

    freeze_all(model)
    for p in afm.parameters():
        p.requires_grad = True

    afm_in = {}
    def pre_hook(m, inp):
        afm_in["x"] = inp[0].detach()
    def fwd_hook(m, inp, out):
        afm_in["y"] = out
    h1 = afm.register_forward_pre_hook(pre_hook)
    h2 = afm.register_forward_hook(fwd_hook)

    train_loader, val_loader = build_loaders(combo, seed)
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                          lr=0.001, momentum=0.9, weight_decay=0.0005)
    run_dir = ensure_run_dir(run_name)

    for epoch in range(1, 3):
        model.train()
        for images, targets in tqdm(train_loader, desc=f"{run_name}"):
            images = [img.to(DEV) for img in images]
            targets = [{k: v.to(DEV) for k, v in t.items()} for t in targets]
            afm_in.clear()
            ld = model(images, targets)
            det_loss = sum(ld.values())
            feat_loss = torch.tensor(0.0, device=DEV)
            x = afm_in.get("x")
            y = afm_in.get("y")
            if x is not None and y is not None:
                feat_loss = 0.05 * nn.functional.mse_loss(y, x)
            total = det_loss + feat_loss
            opt.zero_grad(set_to_none=True)
            total.backward()
            opt.step()

    h1.remove(); h2.remove()
    m = evaluate(model, val_loader)
    m.update({"run_name": run_name, "approach": "C_feat_constraint", "git_hash": GIT,
              "model": combo["model_name"], "dataset": combo["dataset"]})
    save_json(m, run_dir / "eval_metrics.json")
    return m


def main():
    all_r = []
    for combo in COMBOS:
        label = combo["label"]
        print(f"\n{'='*40}\n{label}: {combo['model_name']} + {combo['dataset']}")
        for seed in SEEDS:
            r = train_a(combo, seed, f"round220_{label}_A_s{seed}")
            all_r.append(r)
            print(f"  A_s{seed}: AP50={r['ap50']:.4f} AP75={r['ap75']:.4f}")
            r = train_c(combo, seed, f"round220_{label}_C_s{seed}")
            all_r.append(r)
            print(f"  C_s{seed}: AP50={r['ap50']:.4f} AP75={r['ap75']:.4f}")

    # Send to Feishu
    lines = ["## Plan 2.20 A+C Post-training", ""]
    lines.append("| Run | AP50 | AP75 | Prec | ECE |")
    lines.append("|---:|---:|---:|---:|---:|")
    for r in all_r:
        lines.append(f"| {r['run_name']} | {r['ap50']:.4f} | {r['ap75']:.4f} | {r['precision']:.4f} | {r['ece']:.4f} |")
    msg = "\n".join(lines)
    print(f"\n{msg}")
    subprocess.run(["E:/anaconda/01/envs/RLimage/python.exe", "scripts/notify_feishu.py",
                    f"Plan 2.20: {len(all_r)} groups OK"], capture_output=True)
    subprocess.run(["E:/anaconda/01/envs/RLimage/python.exe", "scripts/notify_feishu.py", msg[:800]], capture_output=True)


if __name__ == "__main__":
    main()
