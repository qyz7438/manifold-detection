"""Train and eval one Round 2.8 group with AFM scales logging."""
from __future__ import annotations

import argparse

import torch
from tqdm import tqdm

from spectral_detection_posttrain.datasets import build_penn_fudan_loaders
from spectral_detection_posttrain.eval.detection_metrics import evaluate_detection_predictions
from spectral_detection_posttrain.models import build_detector
from spectral_detection_posttrain.utils.io import ensure_run_dir, save_checkpoint, save_json
from spectral_detection_posttrain.utils.seed import resolve_device, set_seed


def _to_device(targets: list[dict], device: torch.device) -> list[dict]:
    return [{k: v.to(device) if torch.is_tensor(v) else v for k, v in t.items()} for t in targets]


def _set_trainable(model: torch.nn.Module, mode: str) -> None:
    for param in model.parameters():
        param.requires_grad = mode == "full"
    if mode == "full":
        return
    if mode == "box_head_only":
        for param in model.roi_heads.box_head.parameters():
            param.requires_grad = True
        for param in model.roi_heads.box_predictor.parameters():
            param.requires_grad = True
        return
    if mode == "afm_only":
        if hasattr(model.roi_heads.box_head, "afm"):
            for param in model.roi_heads.box_head.afm.parameters():
                param.requires_grad = True
        return
    if mode == "afm_box_head":
        if hasattr(model.roi_heads.box_head, "afm"):
            for param in model.roi_heads.box_head.afm.parameters():
                param.requires_grad = True
        for param in model.roi_heads.box_head.parameters():
            param.requires_grad = True
        for param in model.roi_heads.box_predictor.parameters():
            param.requires_grad = True
        return
    if mode == "rpn_box_head":
        for param in model.rpn.head.parameters():
            param.requires_grad = True
        for param in model.roi_heads.box_head.parameters():
            param.requires_grad = True
        for param in model.roi_heads.box_predictor.parameters():
            param.requires_grad = True
        return
    if mode == "all_except_backbone":
        for param in model.parameters():
            param.requires_grad = True
        for param in model.backbone.parameters():
            param.requires_grad = False
        return
    if mode == "all_except_rpn":
        for param in model.parameters():
            param.requires_grad = True
        for param in model.rpn.parameters():
            param.requires_grad = False
        return
    if mode == "all_except_box":
        for param in model.parameters():
            param.requires_grad = True
        for param in model.roi_heads.box_head.parameters():
            param.requires_grad = False
        for param in model.roi_heads.box_predictor.parameters():
            param.requires_grad = False
        return
    raise ValueError(f"Unknown trainable mode: {mode}")


def _read_afm_scales(model: torch.nn.Module) -> dict:
    if not hasattr(model.roi_heads.box_head, "afm"):
        return {}
    afm = model.roi_heads.box_head.afm
    result = {}
    for key in ["mag_scale", "phase_scale", "residual_scale"]:
        if hasattr(afm, key):
            result[key] = float(getattr(afm, key).item())
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--afm-type", default="none", choices=["none", "old", "identity", "mplseg", "mplseg_weak", "mplseg_mid", "mplseg_frozen", "mplseg_notune", "mplseg_mag_only", "mplseg_phase_only"])
    parser.add_argument("--afm-residual-mode", default="current", choices=["current", "delta", "norm_delta"])
    parser.add_argument("--trainable-mode", default="full", choices=["full", "box_head_only", "afm_only", "afm_box_head", "rpn_box_head", "all_except_backbone", "all_except_rpn", "all_except_box"])
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--edge-mix", action="store_true", default=False)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument("--dataset", default="penn_fudan", choices=["penn_fudan", "voc"])
    parser.add_argument("--voc-full", action="store_true", default=False)
    parser.add_argument("--model-name", default="fasterrcnn_mobilenet_v3_large_320_fpn",
                        choices=["fasterrcnn_mobilenet_v3_large_320_fpn", "fasterrcnn_resnet50_fpn"])
    args = parser.parse_args()

    config = {
        "seed": args.seed, "device": "cuda" if torch.cuda.is_available() else "cpu",
        "data": {"root": "./data", "download": True, "max_size": 320, "train_fraction": 0.8, "num_workers": 0},
        "model": {"name": args.model_name, "pretrained": True,
                  "model_name": args.model_name,
                  "num_classes": 2, "min_size": 320, "max_size": 320,
                  "afm_channels": 256 if args.afm_type != "none" else 0,
                  "afm_type": args.afm_type, "afm_residual_mode": args.afm_residual_mode},
        "train": {"batch_size": 2, "lr": 0.003, "momentum": 0.9, "weight_decay": 0.0005},
        "matching": {"iou_threshold": 0.5, "score_threshold": 0.05},
        "eval": {"batch_size": 2, "high_conf_threshold": 0.7},
    }

    VOC_20 = ["aeroplane","bicycle","bird","boat","bottle","bus","car","cat","chair",
              "cow","diningtable","dog","horse","motorbike","person","pottedplant",
              "sheep","sofa","train","tvmonitor"]
    if args.dataset == "voc":
        classes = VOC_20 if args.voc_full else ["person", "car", "dog"]
        config["data"].update({"root": "E:/pythonProject1", "year": "2012", "download": False,
                               "classes": classes, "max_size": 480,
                               "train_set": "train", "val_set": "val"})
        config["model"]["num_classes"] = len(classes) + 1
        config["model"]["max_size"] = 480

    set_seed(args.seed)
    device = resolve_device(config)
    run_dir = ensure_run_dir(args.run_name)
    if args.dataset == "voc":
        from spectral_detection_posttrain.datasets.voc_detection import build_voc_detection_loaders
        train_loader, val_loader = build_voc_detection_loaders(config, limit_train=args.limit_train, limit_val=args.limit_val)
    else:
        train_loader, val_loader = build_penn_fudan_loaders(config, limit_train=args.limit_train, limit_val=args.limit_val)
    model = build_detector(config).to(device)

    if args.checkpoint:
        from spectral_detection_posttrain.utils.io import load_checkpoint
        load_checkpoint(model, args.checkpoint, device)

    if args.epochs == 0:
        if not args.checkpoint:
            raise ValueError("--epochs 0 requires --checkpoint")
        _eval_model_fn = None
        # direct eval
        model.eval()
        predictions, targets_list = [], []
        for images, batch_targets in val_loader:
            outputs = model([img.to(device) for img in images])
            predictions.extend([{k: v.detach().cpu() for k, v in output.items()} for output in outputs])
            targets_list.extend([{k: v.detach().cpu() if torch.is_tensor(v) else v for k, v in t.items()} for t in batch_targets])
        metrics = evaluate_detection_predictions(
            predictions, targets_list,
            iou_threshold=float(config["matching"]["iou_threshold"]),
            score_threshold=float(config["matching"]["score_threshold"]),
            high_conf_threshold=float(config["eval"]["high_conf_threshold"]),
        )
        metrics.update({"run_name": args.run_name, "afm_type": args.afm_type,
                        "afm_residual_mode": args.afm_residual_mode,
                        "trainable_mode": args.trainable_mode, "epochs": 0, "seed": args.seed,
                        "history": []})
        save_json(metrics, run_dir / "eval_metrics.json")
        print(metrics)
        return

    _set_trainable(model, args.trainable_mode)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError(f"No trainable params for mode={args.trainable_mode}")

    optimizer = torch.optim.SGD(trainable_params, lr=float(config["train"]["lr"]),
                                momentum=float(config["train"]["momentum"]),
                                weight_decay=float(config["train"]["weight_decay"]))

    history = []
    best_ap50 = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_seen = 0
        for images, targets in tqdm(train_loader, desc=f"{args.run_name} epoch {epoch}"):
            images = [img.to(device) for img in images]
            targets = _to_device(targets, device)
            if args.edge_mix:
                import random
                from spectral_detection_posttrain.datasets.patch_transform import add_detection_patch
                for i in range(len(images)):
                    if random.random() < 0.5:
                        images[i] = add_detection_patch(
                            images[i].cpu(), targets[i], placement="edge",
                            patch_type="checkerboard", patch_size=48,
                        ).to(device)
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(images)
            total_seen += len(images)
        avg_loss = total_loss / max(1, total_seen)

        # Per-epoch val eval
        model.eval()
        predictions, targets_list = [], []
        for images, batch_targets in val_loader:
            outputs = model([img.to(device) for img in images])
            predictions.extend([{k: v.detach().cpu() for k, v in output.items()} for output in outputs])
            targets_list.extend([{k: v.detach().cpu() if torch.is_tensor(v) else v for k, v in t.items()} for t in batch_targets])
        ep_metrics = evaluate_detection_predictions(
            predictions, targets_list,
            iou_threshold=float(config["matching"]["iou_threshold"]),
            score_threshold=float(config["matching"]["score_threshold"]),
            high_conf_threshold=float(config["eval"]["high_conf_threshold"]),
        )
        row = {"epoch": epoch, "train_loss": avg_loss,
               "val_ap50": ep_metrics["ap50"], "val_ap75": ep_metrics["ap75"],
               **_read_afm_scales(model)}
        history.append(row)
        print(f"  epoch {epoch}: loss={avg_loss:.4f} AP50={ep_metrics['ap50']:.4f} AP75={ep_metrics['ap75']:.4f}")

        save_checkpoint(model, run_dir / "checkpoint_last.pth", {"epoch": epoch})
        if ep_metrics["ap50"] > best_ap50:
            best_ap50 = ep_metrics["ap50"]
            save_checkpoint(model, run_dir / "checkpoint_best.pth", {"epoch": epoch, "ap50": best_ap50})

    metrics = ep_metrics
    metrics.update({"run_name": args.run_name, "afm_type": args.afm_type,
                    "afm_residual_mode": args.afm_residual_mode,
                    "trainable_mode": args.trainable_mode, "epochs": args.epochs,
                    "seed": args.seed, "best_ap50": best_ap50, "history": history})
    save_json(metrics, run_dir / "eval_metrics.json")
    print(metrics)


if __name__ == "__main__":
    main()
