"""Plan 2.29: AFM feature refinement.
Load V1 checkpoint (no AFM), embed fresh AFM, freeze all, train AFM only.
"""
import sys, json, subprocess
from pathlib import Path
import torch
from tqdm import tqdm

sys.path.insert(0, "E:/CLIproject/RLimage")
from spectral_detection_posttrain.datasets import build_penn_fudan_loaders
from spectral_detection_posttrain.eval.detection_metrics import evaluate_detection_predictions
from spectral_detection_posttrain.models import build_detector
from spectral_detection_posttrain.models.micro_afm import build_afm_block
from spectral_detection_posttrain.utils.io import load_checkpoint, save_json, ensure_run_dir
from spectral_detection_posttrain.utils.seed import resolve_device, set_seed

GIT = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
DEV = "cuda" if torch.cuda.is_available() else "cpu"
CKPT = "runs/round227_v1_baseline_20ep/checkpoint_best.pth"
SEEDS = [42, 123, 456]
EPOCHS = 5


def build_loaders(seed):
    return build_penn_fudan_loaders({
        "data": {"root": "./data", "max_size": 320, "train_fraction": 0.8, "num_workers": 0},
        "train": {"batch_size": 2},
    })


def build_baseline_model():
    cfg = {"model": {"name": "fasterrcnn_mobilenet_v3_large_320_fpn",
                     "model_name": "fasterrcnn_mobilenet_v3_large_320_fpn",
                     "pretrained": True, "num_classes": 2, "min_size": 320, "max_size": 320}}
    return build_detector(cfg)


def wrap_box_head_with_afm(model, afm_type="mplseg_mid"):
    import torch.nn as nn
    afm = build_afm_block(afm_type=afm_type, channels=256)
    original_box_head = model.roi_heads.box_head

    class AFMThenHead(nn.Module):
        def __init__(self, afm, head):
            super().__init__()
            self.afm = afm
            self.head = head
        def forward(self, x):
            return self.head(self.afm(x))

    model.roi_heads.box_head = AFMThenHead(afm, original_box_head)
    return model.roi_heads.box_head.afm


def freeze_except(module, trainable_modules):
    for p in module.parameters():
        p.requires_grad = False
    for m in trainable_modules:
        for p in m.parameters():
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
    for seed in SEEDS:
        run_name = f"round229_v3_afm_refine_s{seed}"
        set_seed(seed)

        # Build baseline model, load V1 checkpoint
        model = build_baseline_model().to(DEV)
        ckpt = torch.load(CKPT, map_location=DEV)
        model.load_state_dict(ckpt["model"])

        # Wrap box_head with fresh AFM
        afm = wrap_box_head_with_afm(model, "mplseg_mid")
        model = model.to(DEV)  # move new AFM to GPU

        # Freeze all except AFM
        freeze_except(model, [afm])

        train_loader, val_loader = build_loaders(seed)
        opt = torch.optim.SGD(afm.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)
        run_dir = ensure_run_dir(run_name)
        history = []
        best_ap50 = -1.0

        for epoch in range(1, EPOCHS + 1):
            model.train()
            for images, targets in tqdm(train_loader, desc=f"{run_name} e{epoch}"):
                images = [img.to(DEV) for img in images]
                targets = [{k: v.to(DEV) for k, v in t.items()} for t in targets]
                ld = model(images, targets)
                loss = sum(ld.values())
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

            ep_m = evaluate(model, val_loader)
            row = {"epoch": epoch, "val_ap50": ep_m["ap50"], "val_ap75": ep_m["ap75"]}
            history.append(row)
            print(f"  epoch {epoch}: AP50={ep_m['ap50']:.4f} AP75={ep_m['ap75']:.4f}")
            if ep_m["ap50"] > best_ap50:
                best_ap50 = ep_m["ap50"]

        ep_m.update({"run_name": run_name, "afm_type": "mplseg_mid",
                     "trainable_mode": "afm_only", "epochs": EPOCHS, "seed": seed,
                     "best_ap50": best_ap50, "history": history, "git_hash": GIT})
        save_json(ep_m, run_dir / "eval_metrics.json")
        all_r.append(ep_m)
        print(f"  DONE: AP50={ep_m['ap50']:.4f} AP75={ep_m['ap75']:.4f}")

    print("\n## Plan 2.29 Results")
    print(f"| seed | AP50 | AP75 |")
    print(f"|---:|---:|---:|")
    for r in all_r:
        print(f"| {r['seed']} | {r['ap50']:.4f} | {r['ap75']:.4f} |")


if __name__ == "__main__":
    main()
