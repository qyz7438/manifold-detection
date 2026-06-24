"""Hot-start MultiScaleAFM: freeze backbone+box_head, train AFM, then joint FT."""
from __future__ import annotations

import argparse

import torch
from tqdm import tqdm

from spectral_detection_posttrain.datasets import build_penn_fudan_loaders
from spectral_detection_posttrain.eval.detection_metrics import evaluate_detection_predictions
from spectral_detection_posttrain.models import build_detector
from spectral_detection_posttrain.utils.io import ensure_run_dir, save_checkpoint, save_json
from spectral_detection_posttrain.utils.seed import resolve_device, set_seed


def _to_device(targets, device):
    return [{k: v.to(device) if torch.is_tensor(v) else v for k, v in t.items()} for t in targets]


def _eval_model(model, val_loader, device, run_dir):
    model.eval()
    predictions, targets_list = [], []
    for images, batch_targets in val_loader:
        outputs = model([img.to(device) for img in images])
        predictions.extend([{k: v.detach().cpu() for k, v in output.items()} for output in outputs])
        targets_list.extend([{k: v.detach().cpu() if torch.is_tensor(v) else v for k, v in t.items()} for t in batch_targets])
    metrics = evaluate_detection_predictions(
        predictions, targets_list, iou_threshold=0.5, score_threshold=0.05, high_conf_threshold=0.7,
    )
    save_json(metrics, run_dir / "eval_metrics.json")
    print(metrics)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--afm-mode", default="none", choices=["none", "single", "multi"])
    parser.add_argument("--hotstart-epochs", type=int, default=2)
    parser.add_argument("--full-epochs", type=int, default=3)
    parser.add_argument("--run-name", default="round31_hotstart")
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    args = parser.parse_args()

    config = {
        "seed": 42, "device": "cuda" if torch.cuda.is_available() else "cpu",
        "data": {"root": "./data", "download": True, "max_size": 320, "train_fraction": 0.8, "num_workers": 0},
        "model": {"name": "fasterrcnn_mobilenet_v3_large_320_fpn", "pretrained": True,
                  "num_classes": 2, "min_size": 320, "max_size": 320,
                  "afm_channels": 256 if args.afm_mode == "single" else 0,
                  "afm_fpn": args.afm_mode == "multi"},
        "train": {"batch_size": 2, "epochs": 1, "lr": 0.003, "momentum": 0.9, "weight_decay": 0.0005},
        "matching": {"iou_threshold": 0.5, "score_threshold": 0.05},
        "eval": {"batch_size": 2, "high_conf_threshold": 0.7},
    }
    set_seed(config["seed"])
    device = resolve_device(config)
    run_dir = ensure_run_dir(args.run_name)

    train_loader, val_loader = build_penn_fudan_loaders(
        config, limit_train=args.limit_train, limit_val=args.limit_val,
    )
    model = build_detector(config).to(device)

    if args.afm_mode in ("single", "multi"):
        for param in model.parameters():
            param.requires_grad = False
        if hasattr(model, "_multi_afm"):
            for param in model._multi_afm.parameters():
                param.requires_grad = True
        elif hasattr(model.roi_heads.box_head, "afm"):
            for param in model.roi_heads.box_head.afm.parameters():
                param.requires_grad = True

        if args.hotstart_epochs > 0:
            opt_hs = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.0003)
        for epoch in range(1, args.hotstart_epochs + 1):
            model.train()
            total_loss = 0.0
            total_seen = 0
            for images, targets in tqdm(train_loader, desc=f"hotstart epoch {epoch}"):
                images = [img.to(device) for img in images]
                targets = _to_device(targets, device)
                loss_dict = model(images, targets)
                loss = sum(loss_dict.values())
                opt_hs.zero_grad(set_to_none=True)
                loss.backward()
                opt_hs.step()
                total_loss += float(loss.item()) * len(images)
                total_seen += len(images)
            print(f"hotstart epoch {epoch}: loss={total_loss / max(1, total_seen):.4f}")

        for param in model.roi_heads.box_head.parameters():
            param.requires_grad = True
        for param in model.roi_heads.box_predictor.parameters():
            param.requires_grad = True

    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=0.003, momentum=0.9, weight_decay=0.0005,
    )
    for epoch in range(1, args.full_epochs + 1):
        model.train()
        total_loss = 0.0
        total_seen = 0
        for images, targets in tqdm(train_loader, desc=f"joint epoch {epoch}"):
            images = [img.to(device) for img in images]
            targets = _to_device(targets, device)
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(images)
            total_seen += len(images)
        print(f"joint epoch {epoch}: loss={total_loss / max(1, total_seen):.4f}")
        save_checkpoint(model, run_dir / "checkpoint_last.pth", {"epoch": epoch})

    _eval_model(model, val_loader, device, run_dir)
    print(f"run: {args.run_name} mode: {args.afm_mode}")


if __name__ == "__main__":
    main()
