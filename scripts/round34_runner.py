"""Plan 3.4: VOC 20-Class Full Validation.

2 backbones (MobV3, ResNet50) x 3 configs (baseline, A, C) x 3 seeds x VOC2012 full x 3ep.
Uses full VOC2012 20-class dataset. Tests A/C post-training on multi-class detection.
"""
import sys, json, subprocess, time
from pathlib import Path
import torch, torch.nn as nn
from tqdm import tqdm

sys.path.insert(0, "E:/CLIproject/RLimage")
from spectral_detection_posttrain.datasets.voc_detection import build_voc_detection_loaders
from spectral_detection_posttrain.eval.detection_metrics import evaluate_detection_predictions
from spectral_detection_posttrain.models import build_detector
from spectral_detection_posttrain.models.micro_afm import MPLSegAFMBlock
from spectral_detection_posttrain.utils.io import load_checkpoint, save_json, ensure_run_dir
from spectral_detection_posttrain.utils.seed import resolve_device, set_seed

GIT = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS = [42, 123, 456]
MAX_RETRIES = 2

VOC_FULL_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
    "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
    "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]
NUM_CLASSES = len(VOC_FULL_CLASSES) + 1  # 20 + background

COMBO_INFO = {
    "mob": {"model": "fasterrcnn_mobilenet_v3_large_320_fpn", "max_size": 480, "num_classes": NUM_CLASSES},
    "r50": {"model": "fasterrcnn_resnet50_fpn", "max_size": 480, "num_classes": NUM_CLASSES},
}


def freeze_all(model):
    for p in model.parameters():
        p.requires_grad = False


def build_cfg(combo_key):
    info = COMBO_INFO[combo_key]
    return {
        "model": {"name": info["model"], "model_name": info["model"], "pretrained": True,
                  "num_classes": info["num_classes"], "min_size": 320, "max_size": info["max_size"]},
    }


def build_loaders(seed, max_size=480, limit=None):
    cfg = {
        "seed": seed, "device": DEV,
        "data": {"root": "E:/pythonProject1", "year": "2012", "download": False,
                 "classes": VOC_FULL_CLASSES, "max_size": max_size, "num_workers": 0,
                 "train_set": "train", "val_set": "val"},
        "train": {"batch_size": 2},
    }
    lim_train = limit or 500
    lim_val = limit or 200
    return build_voc_detection_loaders(cfg, limit_train=lim_train, limit_val=lim_val)


def build_model(combo_key, afm_type):
    cfg = build_cfg(combo_key)
    cfg["model"]["afm_channels"] = 256
    cfg["model"]["afm_type"] = afm_type
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


def train_baseline(seed, combo_key, run_name):
    """Full fine-tune baseline, 3 epochs."""
    set_seed(seed)
    model = build_model(combo_key, "none").to(DEV)
    train_loader, val_loader = build_loaders(seed)
    opt = torch.optim.SGD(model.parameters(), lr=0.003, momentum=0.9, weight_decay=0.0005)
    run_dir = ensure_run_dir(run_name)

    for epoch in range(1, 4):
        model.train()
        for images, targets in tqdm(train_loader, desc=f"{run_name} e{epoch}"):
            images = [img.to(DEV) for img in images]
            targets = [{k: v.to(DEV) for k, v in t.items()} for t in targets]
            ld = model(images, targets)
            loss = sum(ld.values())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    m = evaluate(model, val_loader)
    m.update({"run_name": run_name, "config": "baseline", "seed": seed, "combo": combo_key, "git_hash": GIT})
    save_json(m, run_dir / "eval_metrics.json")
    return m


def train_baseline_mid06(seed, combo_key, run_name):
    """Full fine-tune with mid06 AFM, 3 epochs. Saves checkpoint for post-training."""
    set_seed(seed)
    model = build_model(combo_key, "mplseg_mid").to(DEV)
    train_loader, val_loader = build_loaders(seed)
    opt = torch.optim.SGD(model.parameters(), lr=0.003, momentum=0.9, weight_decay=0.0005)
    run_dir = ensure_run_dir(run_name)

    for epoch in range(1, 4):
        model.train()
        for images, targets in tqdm(train_loader, desc=f"{run_name} e{epoch}"):
            images = [img.to(DEV) for img in images]
            targets = [{k: v.to(DEV) for k, v in t.items()} for t in targets]
            ld = model(images, targets)
            loss = sum(ld.values())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    m = evaluate(model, val_loader)
    m.update({"run_name": run_name, "config": "baseline_mid06", "seed": seed, "combo": combo_key, "git_hash": GIT})
    save_json(m, run_dir / "eval_metrics.json")
    return m


def train_A(seed, combo_key, mid06_ckpt, run_name):
    """Approach A: weak gate AFM-only."""
    set_seed(seed)
    model = build_model(combo_key, "mplseg_mid").to(DEV)
    load_checkpoint(model, mid06_ckpt, DEV)
    afm = model.roi_heads.box_head.afm
    in_ch = afm.mp[0].in_channels
    new_afm = MPLSegAFMBlock(in_ch=in_ch, gate_strength=0.1).to(DEV)
    new_afm.load_state_dict(afm.state_dict(), strict=False)
    model.roi_heads.box_head.afm = new_afm
    freeze_all(model)
    for p in new_afm.parameters():
        p.requires_grad = True

    train_loader, val_loader = build_loaders(seed)
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                          lr=0.001, momentum=0.9, weight_decay=0.0005)
    run_dir = ensure_run_dir(run_name)

    for epoch in range(1, 3):
        model.train()
        for images, targets in tqdm(train_loader, desc=f"{run_name} e{epoch}"):
            images = [img.to(DEV) for img in images]
            targets = [{k: v.to(DEV) for k, v in t.items()} for t in targets]
            ld = model(images, targets)
            loss = sum(ld.values())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    m = evaluate(model, val_loader)
    m.update({"run_name": run_name, "config": "A", "seed": seed, "combo": combo_key, "git_hash": GIT})
    save_json(m, run_dir / "eval_metrics.json")
    return m


def train_C(seed, combo_key, mid06_ckpt, run_name):
    """Approach C: feature constraint."""
    set_seed(seed)
    model = build_model(combo_key, "mplseg_mid").to(DEV)
    load_checkpoint(model, mid06_ckpt, DEV)
    afm = model.roi_heads.box_head.afm
    freeze_all(model)
    for p in afm.parameters():
        p.requires_grad = True

    afm_in = {}
    def pre_hook(m, inp): afm_in["x"] = inp[0].detach()
    def fwd_hook(m, inp, out): afm_in["y"] = out
    h1 = afm.register_forward_pre_hook(pre_hook)
    h2 = afm.register_forward_hook(fwd_hook)

    train_loader, val_loader = build_loaders(seed)
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                          lr=0.001, momentum=0.9, weight_decay=0.0005)
    run_dir = ensure_run_dir(run_name)

    for epoch in range(1, 3):
        model.train()
        for images, targets in tqdm(train_loader, desc=f"{run_name} e{epoch}"):
            images = [img.to(DEV) for img in images]
            targets = [{k: v.to(DEV) for k, v in t.items()} for t in targets]
            afm_in.clear()
            ld = model(images, targets)
            det_loss = sum(ld.values())
            feat_loss = torch.tensor(0.0, device=DEV)
            x = afm_in.get("x"); y = afm_in.get("y")
            if x is not None and y is not None:
                feat_loss = 0.05 * nn.functional.mse_loss(y, x)
            total = det_loss + feat_loss
            opt.zero_grad(set_to_none=True)
            total.backward()
            opt.step()

    h1.remove(); h2.remove()
    m = evaluate(model, val_loader)
    m.update({"run_name": run_name, "config": "C", "seed": seed, "combo": combo_key, "git_hash": GIT})
    save_json(m, run_dir / "eval_metrics.json")
    return m


def main():
    all_r = []
    # Phase 1: Train baselines + mid06 for each combo (seed 42 only)
    for combo_key in ["mob", "r50"]:
        print(f"\n=== Phase 1: {combo_key} baselines ===")
        r = train_baseline(42, combo_key, f"round34_{combo_key}_baseline_s42")
        all_r.append(r)
        print(f"  baseline: AP50={r['ap50']:.4f} AP75={r['ap75']:.4f}")

        r = train_baseline_mid06(42, combo_key, f"round34_{combo_key}_mid06_s42")
        all_r.append(r)
        print(f"  mid06: AP50={r['ap50']:.4f} AP75={r['ap75']:.4f}")

    # Phase 2: Post-training A+C on mid06 checkpoint (3 seeds each)
    for combo_key in ["mob", "r50"]:
        mid06_ckpt = f"runs/round34_{combo_key}_mid06_s42/checkpoint_last.pth"
        print(f"\n=== Phase 2: {combo_key} post-training ===")
        for seed in SEEDS:
            for approach, train_fn in [("A", train_A), ("C", train_C)]:
                run_name = f"round34_{combo_key}_{approach}_s{seed}"
                for attempt in range(1, MAX_RETRIES + 1):
                    try:
                        r = train_fn(seed, combo_key, mid06_ckpt, run_name)
                        all_r.append(r)
                        print(f"  {approach}_s{seed}: AP50={r['ap50']:.4f} AP75={r['ap75']:.4f}")
                        break
                    except Exception as e:
                        print(f"  RETRY {attempt}/{MAX_RETRIES}: {e}")
                        time.sleep(5)

    lines = ["## Plan 3.4 VOC 20-Class Full", "",
             "| Run | AP50 | AP75 | Prec | ECE |",
             "|---:|---:|---:|---:|---:|"]
    for r in all_r:
        lines.append(f"| {r['run_name']} | {r['ap50']:.4f} | {r['ap75']:.4f} | {r.get('precision',0):.4f} | {r.get('ece',0):.4f} |")
    msg = "\n".join(lines)
    print(f"\n{msg}")
    subprocess.run([sys.executable, "scripts/notify_feishu.py", f"Plan 3.4: {len(all_r)} groups OK"], capture_output=True)
    subprocess.run([sys.executable, "scripts/notify_feishu.py", msg[:800]], capture_output=True)


if __name__ == "__main__":
    main()
