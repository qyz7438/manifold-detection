"""Plan 2.26: Post-training recipe sweep.

6 recipes (A_2ep, A_5ep, C_2ep, C_5ep, AC_2ep, AC_5ep) x 3 seeds x PF+MobV3.
All start from mid06_5ep checkpoint.

Approach A: weak gate (0.1), AFM-only
Approach C: feature constraint (MSE AFM_in/out), AFM-only
Approach AC: weak gate + feature constraint
"""
import sys, json, subprocess, time
from pathlib import Path
import torch, torch.nn as nn
from tqdm import tqdm

sys.path.insert(0, "E:/CLIproject/RLimage")
from spectral_detection_posttrain.datasets import build_penn_fudan_loaders
from spectral_detection_posttrain.eval.detection_metrics import evaluate_detection_predictions
from spectral_detection_posttrain.models import build_detector
from spectral_detection_posttrain.models.micro_afm import MPLSegAFMBlock
from spectral_detection_posttrain.utils.io import load_checkpoint, save_checkpoint, save_json, ensure_run_dir
from spectral_detection_posttrain.utils.seed import resolve_device, set_seed

GIT = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS = [42, 123, 456]
CKPT = "runs/round216pp_mid06_s42/checkpoint_last.pth"
MAX_RETRIES = 2

RECIPES = [
    ("A_2ep", "afm_only", "mplseg_weak", 2, False),
    ("A_5ep", "afm_only", "mplseg_weak", 5, False),
    ("C_2ep", "afm_only", "mplseg_mid",  2, True),
    ("C_5ep", "afm_only", "mplseg_mid",  5, True),
    ("AC_2ep", "afm_only", "mplseg_weak", 2, True),
    ("AC_5ep", "afm_only", "mplseg_weak", 5, True),
]


def freeze_all(model):
    for p in model.parameters():
        p.requires_grad = False


def build_cfg():
    return {
        "model": {"name": "fasterrcnn_mobilenet_v3_large_320_fpn",
                  "model_name": "fasterrcnn_mobilenet_v3_large_320_fpn",
                  "pretrained": True, "num_classes": 2, "min_size": 320, "max_size": 320},
    }


def build_loaders(seed):
    return build_penn_fudan_loaders({
        "data": {"root": "./data", "max_size": 320, "train_fraction": 0.8, "num_workers": 0},
        "train": {"batch_size": 2},
    })


def build_model(afm_type):
    cfg = build_cfg()
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


def train_one(seed, recipe_name, afm_type, epochs, use_constraint, run_name):
    set_seed(seed)
    model = build_model(afm_type).to(DEV)
    load_checkpoint(model, CKPT, DEV)
    afm = model.roi_heads.box_head.afm

    # Set gate_strength for weak gate
    if "weak" in afm_type:
        afm.gate_strength = 0.1

    freeze_all(model)
    for p in afm.parameters():
        p.requires_grad = True

    # Hooks for feature constraint
    afm_in = {}
    if use_constraint:
        def pre_hook(m, inp):
            afm_in["x"] = inp[0].detach()
        def fwd_hook(m, inp, out):
            afm_in["y"] = out
        h1 = afm.register_forward_pre_hook(pre_hook)
        h2 = afm.register_forward_hook(fwd_hook)

    train_loader, val_loader = build_loaders(seed)
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                          lr=0.001, momentum=0.9, weight_decay=0.0005)
    run_dir = ensure_run_dir(run_name)

    for epoch in range(1, epochs + 1):
        model.train()
        for images, targets in tqdm(train_loader, desc=f"{run_name} e{epoch}"):
            images = [img.to(DEV) for img in images]
            targets = [{k: v.to(DEV) for k, v in t.items()} for t in targets]
            if use_constraint:
                afm_in.clear()
            ld = model(images, targets)
            det_loss = sum(ld.values())
            feat_loss = torch.tensor(0.0, device=DEV)
            if use_constraint:
                x = afm_in.get("x")
                y = afm_in.get("y")
                if x is not None and y is not None:
                    feat_loss = 0.05 * nn.functional.mse_loss(y, x)
            total = det_loss + feat_loss
            opt.zero_grad(set_to_none=True)
            total.backward()
            opt.step()

    if use_constraint:
        h1.remove()
        h2.remove()

    m = evaluate(model, val_loader)
    m.update({"run_name": run_name, "recipe": recipe_name, "epochs": epochs,
              "constraint": use_constraint, "seed": seed, "git_hash": GIT})
    save_json(m, run_dir / "eval_metrics.json")
    save_checkpoint(model, run_dir / "checkpoint_last.pth", metadata=m)
    return m


def main():
    all_r = []
    for recipe, _, afm_type, epochs, use_constraint in RECIPES:
        print(f"\n-- {recipe} (afm={afm_type}, ep={epochs}, constr={use_constraint}) --")
        for seed in SEEDS:
            run_name = f"round226_{recipe}_s{seed}"
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    r = train_one(seed, recipe, afm_type, epochs, use_constraint, run_name)
                    all_r.append(r)
                    print(f"  s{seed}: AP50={r['ap50']:.4f} AP75={r['ap75']:.4f}")
                    break
                except Exception as e:
                    print(f"  RETRY {attempt}/{MAX_RETRIES}: {e}")
                    time.sleep(5)

    lines = ["## Plan 2.26 Recipe Sweep", "",
             "| Run | AP50 | AP75 | Prec | ECE |",
             "|---:|---:|---:|---:|---:|"]
    for r in all_r:
        lines.append(f"| {r['run_name']} | {r['ap50']:.4f} | {r['ap75']:.4f} | {r['precision']:.4f} | {r['ece']:.4f} |")
    msg = "\n".join(lines)
    print(f"\n{msg}")
    subprocess.run([sys.executable, "scripts/notify_feishu.py", msg[:800]], capture_output=True)


if __name__ == "__main__":
    main()
