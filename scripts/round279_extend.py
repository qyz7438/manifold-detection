"""Train det_only_unf for 30 epochs from baseline checkpoint."""
import sys, json, subprocess, math, shutil
from pathlib import Path
import torch, torch.nn as nn
from tqdm import tqdm
sys.path.insert(0, "E:/CLIproject/RLimage")
from spectral_detection_posttrain.datasets import build_penn_fudan_loaders
from spectral_detection_posttrain.eval.detection_metrics import evaluate_detection_predictions
from spectral_detection_posttrain.models import build_detector
from spectral_detection_posttrain.utils.io import save_json, ensure_run_dir
from spectral_detection_posttrain.utils.seed import set_seed

GIT = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
DEV = "cuda"; CKPT = "runs/round227_v1_baseline_20ep/checkpoint_best.pth"
SEEDS = [42]; EPOCHS = 30; HEAD_LR = 0.001; BODY_LR = 0.0001


def unfreeze_rlvr(model):
    for p in model.backbone.body.parameters(): p.requires_grad = False
    if hasattr(model.backbone, 'fpn'):
        for p in model.backbone.fpn.parameters(): p.requires_grad = True
    for p in model.rpn.parameters(): p.requires_grad = True
    for p in model.roi_heads.box_head.parameters(): p.requires_grad = True
    for p in model.roi_heads.box_predictor.parameters(): p.requires_grad = True
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            for p in m.parameters(): p.requires_grad = False


def build_opt(model):
    body_params = []; head_params = []
    for n, p in model.named_parameters():
        if not p.requires_grad: continue
        if 'box_head' in n or 'box_predictor' in n: head_params.append(p)
        else: body_params.append(p)
    return torch.optim.SGD([
        {'params': body_params, 'lr': BODY_LR},
        {'params': head_params, 'lr': HEAD_LR},
    ], lr=HEAD_LR, momentum=0.9, weight_decay=0.0005)


@torch.no_grad()
def ev(model, vl):
    model.eval(); ps, ts = [], []
    for img, tgt in vl:
        out = model([i.to(DEV) for i in img])
        ps.extend([{k: v.cpu() for k, v in o.items()} for o in out])
        ts.extend([{k: v.cpu() for k, v in t.items()} for t in tgt])
    return evaluate_detection_predictions(ps, ts, iou_threshold=0.5, score_threshold=0.05)


def bl():
    return build_penn_fudan_loaders({"data": {"root": "./data", "max_size": 320, "train_fraction": 0.8, "num_workers": 0}, "train": {"batch_size": 4}})


def bm():
    return build_detector({"model": {"name": "fasterrcnn_mobilenet_v3_large_320_fpn", "model_name": "fasterrcnn_mobilenet_v3_large_320_fpn", "pretrained": True, "num_classes": 2, "min_size": 320, "max_size": 320}})


def run_one(seed):
    run_name = f"round279_extend_s{seed}"
    set_seed(seed)
    model = bm().to(DEV)
    ckpt = torch.load(CKPT, map_location=DEV)
    model.load_state_dict(ckpt["model"])
    unfreeze_rlvr(model)
    opt = build_opt(model)
    tl, vl = bl()
    run_dir = ensure_run_dir(run_name)
    h = []; best_ap75 = -1.0

    for ep in range(1, EPOCHS + 1):
        model.train()
        td = 0.0
        for imgs, tgts in tqdm(tl, desc=f"{run_name} e{ep}"):
            imgs_d = [i.to(DEV) for i in imgs]
            tgts_t = [{k: v.to(DEV) for k, v in t.items()} for t in tgts]
            ld = model(imgs_d, tgts_t)
            det = sum(ld.values()) if isinstance(ld, dict) else sum(sum(d.values()) for d in ld if isinstance(d, dict))
            opt.zero_grad(set_to_none=True); det.backward(); opt.step()
            td += det.item()

        em = ev(model, vl)
        row = {"epoch": ep, "val_ap50": em["ap50"], "val_ap75": em["ap75"],
               "precision": em.get("precision", 0), "recall": em.get("recall", 0),
               "ece": em.get("ece", 0), "det_loss": td}
        h.append(row)
        print(f"  e{ep}: AP75={em['ap75']:.4f} AP50={em['ap50']:.4f}")
        if em["ap75"] > best_ap75: best_ap75 = em["ap75"]

    em.update({"run_name": run_name, "seed": seed, "epochs": EPOCHS,
               "best_ap75": best_ap75, "history": h, "git_hash": GIT})
    save_json(em, run_dir / "eval_metrics.json")
    return em


if __name__ == "__main__":
    all_results = []
    for s in SEEDS:
        r = run_one(s)
        all_results.append(r)

    print("\n## Round279 Extend (epoch 9-30)")
    for r in all_results:
        bv = max(x["val_ap75"] for x in r["history"])
        print(f"  seed={r['seed']}: bestAP75={bv:.4f}  finalAP75={r['history'][-1]['val_ap75']:.4f}")
