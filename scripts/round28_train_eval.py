"""Train and eval one Round 2.8 group with AFM scales logging."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running the script directly without setting PYTHONPATH.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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
    if mode == "image_sm_only":
        for param in model.parameters():
            param.requires_grad = False
        if hasattr(model, "_image_spectral_manifold"):
            for param in model._image_spectral_manifold.parameters():
                param.requires_grad = True
        return
    if mode == "roi_sm_only":
        for param in model.parameters():
            param.requires_grad = False
        if (
            hasattr(model.roi_heads.box_head, "roi_sm")
            and model.roi_heads.box_head.roi_sm is not None
        ):
            for param in model.roi_heads.box_head.roi_sm.parameters():
                param.requires_grad = True
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


def _reset_spectral_stats(model: torch.nn.Module) -> None:
    """Reset running diagnostic buffers for any spectral manifold modules."""
    for attr in ["_fpn_spectral_manifold", "_fpn_real_adapter"]:
        module = getattr(model, attr, None)
        if module is not None and hasattr(module, "reset_stats"):
            module.reset_stats()
    if (
        hasattr(model.roi_heads, "box_head")
        and hasattr(model.roi_heads.box_head, "roi_sm")
        and model.roi_heads.box_head.roi_sm is not None
    ):
        model.roi_heads.box_head.roi_sm.reset_stats()


def _epoch_spectral_stats(model: torch.nn.Module) -> dict:
    """Return averaged diagnostic stats for the current epoch."""
    result = {}
    for attr in ["_fpn_spectral_manifold", "_fpn_real_adapter"]:
        module = getattr(model, attr, None)
        if module is not None and hasattr(module, "epoch_stats"):
            result.update(module.epoch_stats())
    if (
        hasattr(model.roi_heads, "box_head")
        and hasattr(model.roi_heads.box_head, "roi_sm")
        and model.roi_heads.box_head.roi_sm is not None
    ):
        result.update(model.roi_heads.box_head.roi_sm.epoch_stats())
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--afm-type", default="none", choices=["none", "old", "identity", "mplseg", "mplseg_weak", "mplseg_mid", "mplseg_frozen", "mplseg_notune", "mplseg_mag_only", "mplseg_phase_only"])
    parser.add_argument("--afm-residual-mode", default="current", choices=["current", "delta", "norm_delta"])
    parser.add_argument("--trainable-mode", default="full", choices=["full", "box_head_only", "afm_only", "afm_box_head", "rpn_box_head", "all_except_backbone", "all_except_rpn", "all_except_box", "image_sm_only", "roi_sm_only"])
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--edge-mix", action="store_true", default=False)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument("--dataset", default="penn_fudan", choices=["penn_fudan", "voc", "nwpu", "coco"])
    parser.add_argument("--voc-full", action="store_true", default=False)
    parser.add_argument("--voc-root", default="./data", help="Root directory containing VOCdevkit")
    parser.add_argument("--coco-root", default="./data/coco", help="Root directory containing COCO 2017 folders")
    parser.add_argument("--min-size", type=int, default=None, help="Model min_size (default: dataset-specific)")
    parser.add_argument("--max-size", type=int, default=None, help="Model max_size (default: dataset-specific)")
    parser.add_argument("--batch-size", type=int, default=None, help="Training/eval batch size (default 16)")
    parser.add_argument("--lr", type=float, default=None, help="SGD learning rate (default dataset-specific)")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader num_workers")
    parser.add_argument("--model-name", default="fasterrcnn_mobilenet_v3_large_320_fpn",
                        choices=["fasterrcnn_mobilenet_v3_large_320_fpn", "fasterrcnn_resnet50_fpn"])
    parser.add_argument("--use-pbg", action="store_true", default=False)
    parser.add_argument("--use-lsg", action="store_true", default=False)
    parser.add_argument("--use-tam", action="store_true", default=False)
    parser.add_argument("--use-pah", action="store_true", default=False)
    parser.add_argument("--fpn-spectral-manifold", action="store_true", default=False)
    parser.add_argument("--fpn-real-adapter", action="store_true", default=False)
    parser.add_argument("--image-spectral-manifold", action="store_true", default=False)
    parser.add_argument("--roi-spectral-manifold", action="store_true", default=False)
    parser.add_argument("--pbg-alpha-init", type=float, default=1e-2)
    parser.add_argument("--pbg-phase-mask", default="soft", choices=["none", "hard", "soft"])
    parser.add_argument("--lsg-alpha-init", type=float, default=0.1)
    parser.add_argument("--lsg-use-radius", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--roi-align-size", type=int, default=7)
    parser.add_argument("--tam-latent-dim", type=int, default=256)
    parser.add_argument("--tam-spectral-quality", action="store_true", default=False)
    parser.add_argument("--tam-contrastive", action="store_true", default=False)
    parser.add_argument("--pah-temperature", type=float, default=0.3)
    parser.add_argument("--pah-learnable-temp", action="store_true", default=False)
    parser.add_argument("--pah-num-bg", type=int, default=4)
    parser.add_argument("--pah-gamma-init", type=float, default=0.0)
    parser.add_argument("--fpn-sm-channels", type=int, default=256)
    parser.add_argument("--fpn-sm-levels", type=int, default=4)
    parser.add_argument("--fpn-sm-latent-dim", type=int, default=None)
    parser.add_argument("--fpn-sm-hidden-dim", type=int, default=None)
    parser.add_argument("--fpn-sm-use-freq-coords", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fpn-sm-use-level-coords", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fpn-sm-gate-activation", default="sigmoid", choices=["sigmoid", "sigmoid2", "tanh"])
    parser.add_argument("--fpn-sm-suppress-dc", action="store_true", default=False)
    parser.add_argument("--fpn-sm-init-alpha", type=float, default=1e-3)
    parser.add_argument("--fpn-real-channels", type=int, default=256)
    parser.add_argument("--fpn-real-levels", type=int, default=4)
    parser.add_argument("--fpn-real-latent-dim", type=int, default=None)
    parser.add_argument("--fpn-real-init-alpha", type=float, default=1e-3)
    parser.add_argument("--fpn-attention-type", default="none",
                        choices=["none", "se", "fcanet", "eca"],
                        help="FPN-level channel attention baseline")
    parser.add_argument("--fpn-attention-reduction", type=int, default=16,
                        help="Reduction ratio for SE/FcaNet attention")
    parser.add_argument("--img-sm-channels", type=int, default=3)
    parser.add_argument("--img-sm-latent-dim", type=int, default=None)
    parser.add_argument("--roi-sm-channels", type=int, default=256)
    parser.add_argument("--roi-sm-latent-dim", type=int, default=None)
    parser.add_argument("--roi-sm-hidden-dim", type=int, default=None)
    parser.add_argument("--roi-sm-use-freq-coords", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--roi-sm-gate-activation", default="sigmoid", choices=["sigmoid", "sigmoid2", "tanh"])
    parser.add_argument("--roi-sm-suppress-dc", action="store_true", default=False)
    parser.add_argument("--roi-sm-init-alpha", type=float, default=1e-3)
    args = parser.parse_args()

    config = {
        "seed": args.seed, "device": "cuda" if torch.cuda.is_available() else "cpu",
        "data": {"root": "./data", "dataset": args.dataset, "download": True, "max_size": 320, "train_fraction": 0.8, "num_workers": 0},
        "model": {"name": args.model_name, "pretrained": True,
                  "model_name": args.model_name,
                  "num_classes": 2, "min_size": 320, "max_size": 320,
                  "afm_channels": 256 if args.afm_type != "none" else 0,
                  "afm_type": args.afm_type, "afm_residual_mode": args.afm_residual_mode,
                  "use_pbg": args.use_pbg,
                  "use_lsg": args.use_lsg,
                  "use_tam": args.use_tam,
                  "use_pah": args.use_pah,
                  "fpn_spectral_manifold": args.fpn_spectral_manifold,
                  "fpn_real_adapter": args.fpn_real_adapter,
                  "image_spectral_manifold": args.image_spectral_manifold,
                  "roi_spectral_manifold": args.roi_spectral_manifold,
                  "pbg_alpha_init": args.pbg_alpha_init,
                  "pbg_phase_mask": args.pbg_phase_mask,
                  "lsg_alpha_init": args.lsg_alpha_init,
                  "lsg_use_radius": args.lsg_use_radius,
                  "roi_align_size": args.roi_align_size,
                  "tam_latent_dim": args.tam_latent_dim,
                  "tam_spectral_quality": args.tam_spectral_quality,
                  "tam_contrastive": args.tam_contrastive,
                  "pah_temperature": args.pah_temperature,
                  "pah_learnable_temp": args.pah_learnable_temp,
                  "pah_num_bg": args.pah_num_bg,
                  "pah_gamma_init": args.pah_gamma_init,
                  "fpn_sm_channels": args.fpn_sm_channels,
                  "fpn_sm_levels": args.fpn_sm_levels,
                  "fpn_sm_latent_dim": args.fpn_sm_latent_dim,
                  "fpn_sm_hidden_dim": args.fpn_sm_hidden_dim,
                  "fpn_sm_use_freq_coords": args.fpn_sm_use_freq_coords,
                  "fpn_sm_use_level_coords": args.fpn_sm_use_level_coords,
                  "fpn_sm_gate_activation": args.fpn_sm_gate_activation,
                  "fpn_sm_suppress_dc": args.fpn_sm_suppress_dc,
                  "fpn_sm_init_alpha": args.fpn_sm_init_alpha,
                  "fpn_real_channels": args.fpn_real_channels,
                  "fpn_real_levels": args.fpn_real_levels,
                  "fpn_real_latent_dim": args.fpn_real_latent_dim,
                  "fpn_real_init_alpha": args.fpn_real_init_alpha,
                  "fpn_attention_type": args.fpn_attention_type,
                  "fpn_attention_reduction": args.fpn_attention_reduction,
                  "img_sm_channels": args.img_sm_channels,
                  "img_sm_latent_dim": args.img_sm_latent_dim,
                  "roi_sm_channels": args.roi_sm_channels,
                  "roi_sm_latent_dim": args.roi_sm_latent_dim,
                  "roi_sm_hidden_dim": args.roi_sm_hidden_dim,
                  "roi_sm_use_freq_coords": args.roi_sm_use_freq_coords,
                  "roi_sm_gate_activation": args.roi_sm_gate_activation,
                  "roi_sm_suppress_dc": args.roi_sm_suppress_dc,
                  "roi_sm_init_alpha": args.roi_sm_init_alpha},
        "train": {"batch_size": args.batch_size if args.batch_size is not None else 16, "lr": args.lr if args.lr is not None else 0.024, "momentum": 0.9, "weight_decay": 0.0005},
        "matching": {"iou_threshold": 0.5, "score_threshold": 0.05},
        "eval": {"batch_size": 16, "high_conf_threshold": 0.7},
    }

    VOC_20 = ["aeroplane","bicycle","bird","boat","bottle","bus","car","cat","chair",
              "cow","diningtable","dog","horse","motorbike","person","pottedplant",
              "sheep","sofa","train","tvmonitor"]
    if args.dataset == "voc":
        classes = VOC_20 if args.voc_full else ["person", "car", "dog"]
        config["data"].update({"root": args.voc_root, "download": False,
                               "classes": classes, "max_size": 480,
                               "num_workers": args.num_workers,
                               "train_years": [{"year": "2007", "image_set": "trainval"},
                                               {"year": "2012", "image_set": "trainval"}],
                               "val_years": [{"year": "2012", "image_set": "val"}]})
        config["model"]["num_classes"] = len(classes) + 1
        config["model"]["max_size"] = 480
    elif args.dataset == "coco":
        config["data"].update({"root": args.coco_root,
                               "download": False,
                               "max_size": 800,
                               "num_workers": args.num_workers})
        config["model"]["num_classes"] = 91  # 80 COCO classes + background
        config["model"]["min_size"] = 800
        config["model"]["max_size"] = 1333
    elif args.dataset == "nwpu":
        config["data"].update({
            "root": "./data/NWPU VHR-10 dataset",
            "annotation": "./data/NWPU_VHR10_coco.json",
            "train_fraction": 0.7,
            "max_size": 480,
        })
        config["model"]["num_classes"] = 11
        config["model"]["min_size"] = 480
        config["model"]["max_size"] = 480

    if args.min_size is not None:
        config["model"]["min_size"] = args.min_size
    if args.max_size is not None:
        config["model"]["max_size"] = args.max_size

    set_seed(args.seed)
    device = resolve_device(config)
    run_dir = ensure_run_dir(args.run_name)
    save_json(config, run_dir / "config.json")
    if args.dataset == "voc":
        from spectral_detection_posttrain.datasets.voc_detection import build_voc_detection_loaders
        train_loader, val_loader = build_voc_detection_loaders(config, limit_train=args.limit_train, limit_val=args.limit_val)
    elif args.dataset == "nwpu":
        from spectral_detection_posttrain.datasets.nwpu_vhr10 import build_nwpu_vhr10_loaders
        train_loader, val_loader = build_nwpu_vhr10_loaders(config, limit_train=args.limit_train, limit_val=args.limit_val)
    elif args.dataset == "coco":
        from spectral_detection_posttrain.datasets.coco_detection import build_coco_detection_loaders
        train_loader, val_loader = build_coco_detection_loaders(config, limit_train=args.limit_train, limit_val=args.limit_val)
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
        _reset_spectral_stats(model)
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

            # Optional auxiliary losses from TAM (spectral quality / contrastive).
            tam_aux = 0.0
            if (
                hasattr(model.roi_heads.box_head, "tam")
                and model.roi_heads.box_head.tam is not None
            ):
                tam_aux = model.roi_heads.box_head.tam.get_aux_loss()
                if isinstance(tam_aux, torch.Tensor) and tam_aux.item() != 0.0:
                    loss = loss + tam_aux

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
        spectral_stats = _epoch_spectral_stats(model)
        row = {"epoch": epoch, "train_loss": avg_loss,
               "val_ap50": ep_metrics["ap50"], "val_ap75": ep_metrics["ap75"],
               **_read_afm_scales(model), **spectral_stats}
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
