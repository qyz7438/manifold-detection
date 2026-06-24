"""Verify identity AFM is detector-level no-op with frozen weights."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from spectral_detection_posttrain.datasets import build_penn_fudan_loaders
from spectral_detection_posttrain.models import build_detector
from spectral_detection_posttrain.utils.io import ensure_run_dir, save_json
from spectral_detection_posttrain.utils.seed import resolve_device, set_seed


def _config(afm_type: str = "none") -> dict:
    return {
        "seed": 42, "device": "cuda" if torch.cuda.is_available() else "cpu",
        "data": {"root": "./data", "download": True, "max_size": 320, "train_fraction": 0.8, "num_workers": 0},
        "model": {"name": "fasterrcnn_mobilenet_v3_large_320_fpn", "pretrained": True,
                  "num_classes": 2, "min_size": 320, "max_size": 320,
                  "afm_channels": 256 if afm_type != "none" else 0,
                  "afm_type": afm_type, "afm_residual_mode": "current"},
        "eval": {"batch_size": 1},
    }


def _strip_afm_state(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k: v for k, v in state_dict.items() if ".afm." not in k}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="round28_frozen_parity")
    parser.add_argument("--limit-val", type=int, default=8)
    args = parser.parse_args()

    set_seed(42)
    cfg_no = _config("none")
    device = resolve_device(cfg_no)
    run_dir = ensure_run_dir(args.run_name)

    model_no = build_detector(cfg_no).to(device).eval()
    cfg_afm = _config("identity")
    model_afm = build_detector(cfg_afm).to(device).eval()
    missing, unexpected = model_afm.load_state_dict(_strip_afm_state(model_no.state_dict()), strict=False)

    _, val_loader = build_penn_fudan_loaders(cfg_no, limit_val=args.limit_val, batch_size=1)
    max_box_diff = 0.0
    max_score_diff = 0.0
    max_count_diff = 0
    per_image = []

    with torch.no_grad():
        for image_idx, (images, _) in enumerate(val_loader):
            images = [img.to(device) for img in images]
            pred_no = model_no(images)[0]
            pred_afm = model_afm(images)[0]
            count_diff = abs(len(pred_no["scores"]) - len(pred_afm["scores"]))
            max_count_diff = max(max_count_diff, count_diff)
            common = min(len(pred_no["scores"]), len(pred_afm["scores"]))
            box_diff = 0.0
            score_diff = 0.0
            if common > 0:
                box_diff = float((pred_no["boxes"][:common] - pred_afm["boxes"][:common]).abs().max().item())
                score_diff = float((pred_no["scores"][:common] - pred_afm["scores"][:common]).abs().max().item())
            max_box_diff = max(max_box_diff, box_diff)
            max_score_diff = max(max_score_diff, score_diff)
            per_image.append({"image_idx": image_idx, "box_diff": box_diff, "score_diff": score_diff,
                              "count_no_afm": int(len(pred_no["scores"])), "count_afm": int(len(pred_afm["scores"]))})

    metrics = {"max_box_diff": max_box_diff, "max_score_diff": max_score_diff, "max_count_diff": max_count_diff,
               "missing_keys": list(missing), "unexpected_keys": list(unexpected), "per_image": per_image,
               "pass": max_box_diff <= 1e-2 and max_score_diff <= 1e-2 and max_count_diff == 0}
    save_json(metrics, Path(run_dir) / "parity_metrics.json")
    print(metrics)


if __name__ == "__main__":
    main()
