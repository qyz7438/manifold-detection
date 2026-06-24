"""Plan 2.26b: Scale C_5ep feature-constraint post-train.

Start from round226_C_5ep_s42 checkpoint and continue training with the same
feature constraint for 5/10/20 epochs to see if AP75 improves further or saturates.
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
from spectral_detection_posttrain.utils.io import load_checkpoint, save_json, ensure_run_dir
from spectral_detection_posttrain.utils.seed import resolve_device, set_seed

GIT = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
CKPT = "runs/round226_C_5ep_s42/checkpoint_last.pth"
MAX_RETRIES = 2

CONTINUES = [
    ("C_5ep_plus5", 5),
    ("C_5ep_plus10", 10),
    ("C_5ep_plus20", 20),
]


def freeze_all(model):
    for p in model.parameters():
        p.requires_grad = False


def build_model():
    cfg = {
        "model": {"name": "fasterrcnn_mobilenet_v3_large_320_fpn",
                  "model_name": "fasterrcnn_mobilenet_v3_large_320_fpn",
                  "pretrained": True, "num_classes": 2, "min_size": 320, "max_size": 320,
                  "afm_channels": 256, "afm_type": "mplseg_mid"},
    }
    return build_detector(cfg)


def build_loaders(seed):
    return build_penn_fudan_loaders({
        "data": {"root": "./data", "max_size": 320, "train_fraction": 0.8, "num_workers": 0},
        "train": {"batch_size": 2},
    })


@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    preds, targs = [], []
    for images, targets in val_loader:
        out = model([img.to(DEV) for img in images])
        preds.extend([{k: v.cpu() for k, v in o.items()} for o in out])
        targs.extend([{k: v.cpu() for k, v in t.items()} for t in targets])
    return evaluate_detection_predictions(preds, targs, iou_threshold=0.5, score_threshold=0.05)


def train_one(seed, extra_epochs, run_name):
    set_seed(seed)
    model = build_model().to(DEV)
    load_checkpoint(model, CKPT, DEV)
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

    train_loader, val_loader = build_loaders(seed)
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                          lr=0.001, momentum=0.9, weight_decay=0.0005)
    run_dir = ensure_run_dir(run_name)
    history = []

    for epoch in range(1, extra_epochs + 1):
        model.train()
        epoch_loss = 0.0
        for images, targets in tqdm(train_loader, desc=f"{run_name} e{epoch}"):
            images = [img.to(DEV) for img in images]
            targets = [{k: v.to(DEV) for k, v in t.items()} for t in targets]
            afm_in.clear()
            ld = model(images, targets)
            det_loss = sum(ld.values())
            x = afm_in.get("x")
            y = afm_in.get("y")
            feat_loss = 0.05 * nn.functional.mse_loss(y, x) if x is not None and y is not None else torch.tensor(0.0, device=DEV)
            total = det_loss + feat_loss
            opt.zero_grad(set_to_none=True)
            total.backward()
            opt.step()
            epoch_loss += total.item()

        m = evaluate(model, val_loader)
        m.update({"run_name": run_name, "extra_epochs": extra_epochs,
                  "epoch": epoch, "seed": seed, "git_hash": GIT,
                  "train_loss": epoch_loss / len(train_loader)})
        history.append(m)
        save_json(m, run_dir / f"eval_metrics_epoch_{epoch}.json")

    h1.remove()
    h2.remove()

    final = evaluate(model, val_loader)
    final.update({"run_name": run_name, "extra_epochs": extra_epochs,
                  "seed": seed, "git_hash": GIT, "history": history})
    save_json(final, run_dir / "eval_metrics.json")
    return final


def main():
    all_r = []
    for name, extra_epochs in CONTINUES:
        print(f"\n-- {name} (+{extra_epochs} epochs from C_5ep_s42) --")
        run_name = f"round226b_{name}_s{SEED}"
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                r = train_one(SEED, extra_epochs, run_name)
                all_r.append(r)
                print(f"  AP50={r['ap50']:.4f} AP75={r['ap75']:.4f} ECE={r['ece']:.4f}")
                break
            except Exception as e:
                print(f"  RETRY {attempt}/{MAX_RETRIES}: {e}")
                time.sleep(5)

    lines = ["## Plan 2.26b C_5ep Continuation", "",
             "| Run | AP50 | AP75 | Prec | ECE |",
             "|---:|---:|---:|---:|---:|"]
    for r in all_r:
        lines.append(f"| {r['run_name']} | {r['ap50']:.4f} | {r['ap75']:.4f} | {r['precision']:.4f} | {r['ece']:.4f} |")
    msg = "\n".join(lines)
    print(f"\n{msg}")
    subprocess.run([sys.executable, "scripts/notify_feishu.py", msg[:800]], capture_output=True)


if __name__ == "__main__":
    main()
