"""Round 2.11 stress evaluation — clean, edge, texture, near-object patch scenes."""
from __future__ import annotations

import argparse
import yaml

import torch

from spectral_detection_posttrain.datasets.patch_transform import add_detection_patch
from spectral_detection_posttrain.datasets.voc_detection import build_voc_detection_loaders
from spectral_detection_posttrain.eval.detection_metrics import evaluate_detection_predictions
from spectral_detection_posttrain.models import build_detector
from spectral_detection_posttrain.utils.io import load_checkpoint, save_json
from spectral_detection_posttrain.utils.seed import resolve_device, set_seed


SCENES = [
    ("clean", "none", "none"),
    ("object_edge", "object_edge", "checkerboard"),
    ("background_texture", "random", "random"),
    ("near_object", "near_object", "checkerboard"),
]

GROUPS = [
    "round211_voc_v1_baseline_eval",
    "round211_voc_v2_posttrain_detection_only",
    "round211_voc_v3_posttrain_spatial",
    "round211_voc_v4_posttrain_spatial_spectral_loggate",
    "round211_voc_v5_posttrain_spatial_shuffled_spectral",
]


@torch.no_grad()
def eval_scene(model, val_loader, device, config, placement, patch_type, patch_size):
    model.eval()
    predictions, targets_list = [], []
    for images, batch_targets in val_loader:
        stressed = []
        for img, tgt in zip(images, batch_targets):
            if placement != "none":
                img = add_detection_patch(img, tgt, placement=placement,
                                          patch_type=patch_type, patch_size=patch_size)
            stressed.append(img.to(device))
        outputs = model(stressed)
        predictions.extend([{k: v.detach().cpu() for k, v in output.items()} for output in outputs])
        targets_list.extend([{k: v.detach().cpu() if torch.is_tensor(v) else v for k, v in t.items()} for t in batch_targets])
    eval_cfg = config.get("eval", {})
    return evaluate_detection_predictions(
        predictions, targets_list,
        iou_threshold=float(eval_cfg.get("score_threshold", 0.05)),
        score_threshold=float(eval_cfg.get("score_threshold", 0.05)),
        high_conf_threshold=float(eval_cfg.get("high_conf_threshold", 0.7)),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument("--groups", nargs="*", default=None,
                        help="Specific groups to eval (default: all 5 VOC groups)")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config["seed"] = args.seed

    set_seed(args.seed)
    device = resolve_device(config)
    patch_cfg = config.get("patch", {})
    patch_size = int(patch_cfg.get("patch_size", 64))

    batch_size = int(config["train"].get("batch_size", 2))
    _, val_loader = build_voc_detection_loaders(
        config, limit_val=args.limit_val, batch_size=batch_size)

    groups = args.groups if args.groups else GROUPS

    for group_name in groups:
        ckpt_path = f"runs/{group_name}/checkpoint_last.pth"
        model = build_detector(config).to(device)
        try:
            load_checkpoint(model, ckpt_path, device)
        except FileNotFoundError:
            print(f"SKIP {group_name}: checkpoint not found at {ckpt_path}")
            continue

        for scene_name, placement, patch_type in SCENES:
            metrics = eval_scene(model, val_loader, device, config, placement, patch_type, patch_size)
            metrics["group"] = group_name
            metrics["scene"] = scene_name
            save_json(metrics, f"runs/{group_name}/{scene_name}_eval_metrics.json")
            ap50 = metrics.get("ap50", metrics.get("AP50", "N/A"))
            print(f"{group_name}/{scene_name}  AP50={ap50}")


if __name__ == "__main__":
    main()
