"""Light-weight efficiency profiling for detector variants.

Reports:
- total / trainable parameters
- backbone + full-model CPU latency on a single dummy image
- peak CPU memory allocated (if CUDA is used, also GPU memory)

Usage example:
    python scripts/profile_model.py --model-name fasterrcnn_mobilenet_v3_large_320_fpn --fpn-spectral-manifold
    python scripts/profile_model.py --model-name fasterrcnn_mobilenet_v3_large_320_fpn --fpn-attention-type eca
"""
from __future__ import annotations

import argparse
import time

import torch

from spectral_detection_posttrain.core.models.build_detector import build_detector


def _build_config(args: argparse.Namespace) -> dict:
    return {
        "seed": 42,
        "device": "cpu",
        "data": {"root": "./data", "dataset": "voc", "download": False, "max_size": args.max_size},
        "model": {
            "name": args.model_name,
            "model_name": args.model_name,
            "pretrained": False,
            "num_classes": args.num_classes,
            "min_size": args.min_size,
            "max_size": args.max_size,
            "afm_channels": 0,
            "afm_type": "none",
            "afm_residual_mode": "current",
            "use_pbg": False,
            "use_lsg": False,
            "use_tam": False,
            "use_pah": False,
            "fpn_spectral_manifold": args.fpn_spectral_manifold,
            "fpn_real_adapter": args.fpn_real_adapter,
            "image_spectral_manifold": False,
            "roi_spectral_manifold": False,
            "fpn_attention_type": args.fpn_attention_type,
            "fpn_attention_reduction": args.fpn_attention_reduction,
            "fpn_sm_channels": 256,
            "fpn_sm_levels": 4,
            "fpn_sm_latent_dim": args.fpn_sm_latent_dim,
            "fpn_sm_hidden_dim": args.fpn_sm_hidden_dim,
            "fpn_sm_use_freq_coords": not args.no_fpn_sm_freq_coords,
            "fpn_sm_use_level_coords": not args.no_fpn_sm_level_coords,
            "fpn_sm_gate_activation": "sigmoid",
            "fpn_sm_suppress_dc": False,
            "fpn_sm_init_alpha": args.fpn_sm_init_alpha,
            "fpn_real_channels": 256,
            "fpn_real_levels": 4,
            "fpn_real_latent_dim": None,
            "fpn_real_init_alpha": 1e-3,
            "img_sm_channels": 3,
            "img_sm_latent_dim": None,
            "roi_sm_channels": 256,
            "roi_sm_latent_dim": None,
            "roi_sm_hidden_dim": None,
            "roi_sm_use_freq_coords": True,
            "roi_sm_gate_activation": "sigmoid",
            "roi_sm_suppress_dc": False,
            "roi_sm_init_alpha": 1e-3,
        },
        "train": {"batch_size": 1, "lr": 0.024, "momentum": 0.9, "weight_decay": 0.0005},
        "matching": {"iou_threshold": 0.5, "score_threshold": 0.05},
        "eval": {"batch_size": 1, "high_conf_threshold": 0.7},
    }


def _timeit(fn, x, warmup: int = 3, repeats: int = 10):
    for _ in range(warmup):
        fn(x)
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn(x)
    t1 = time.perf_counter()
    return (t1 - t0) / repeats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="fasterrcnn_mobilenet_v3_large_320_fpn")
    parser.add_argument("--num-classes", type=int, default=21)
    parser.add_argument("--min-size", type=int, default=320)
    parser.add_argument("--max-size", type=int, default=480)
    parser.add_argument("--fpn-spectral-manifold", action="store_true")
    parser.add_argument("--fpn-real-adapter", action="store_true")
    parser.add_argument("--fpn-attention-type", default="none", choices=["none", "se", "fcanet", "eca"])
    parser.add_argument("--fpn-attention-reduction", type=int, default=16)
    parser.add_argument("--fpn-sm-latent-dim", type=int, default=None)
    parser.add_argument("--fpn-sm-hidden-dim", type=int, default=None)
    parser.add_argument("--fpn-sm-init-alpha", type=float, default=1e-3)
    parser.add_argument("--no-fpn-sm-freq-coords", action="store_true")
    parser.add_argument("--no-fpn-sm-level-coords", action="store_true")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()

    config = _build_config(args)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model = build_detector(config).to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    added_params = 0
    if hasattr(model, "_fpn_spectral_manifold") and model._fpn_spectral_manifold is not None:
        added_params = sum(p.numel() for p in model._fpn_spectral_manifold.parameters())
    elif hasattr(model, "_fpn_attention") and model._fpn_attention is not None:
        added_params = sum(p.numel() for p in model._fpn_attention.parameters())

    dummy = torch.zeros(1, 3, args.min_size, args.max_size, device=device)
    from torch.utils.flop_counter import FlopCounterMode
    with torch.no_grad():
        # FLOPs estimate.
        with FlopCounterMode(model, display=False) as fcm:
            model(dummy)
            total_flops = fcm.get_total_flops()
        # BackBone / FPN latency.
        backbone_time = _timeit(lambda x: model.backbone(x), dummy, repeats=args.repeats)
        # Full-model eval latency (RPN + ROI are stochastic, but on dummy zero input
        # the number of proposals may be small; this is a lower-bound estimate).
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        full_time = _timeit(lambda x: model(x), dummy, repeats=args.repeats)
        peak_mem_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2) if device.type == "cuda" else 0.0

    print(f"model: {args.model_name}")
    print(f"input: {tuple(dummy.shape)}")
    print(f"total_params: {total_params:,}")
    print(f"trainable_params: {trainable_params:,}")
    print(f"added_module_params: {added_params:,}")
    print(f"relative_overhead: {added_params / max(1, total_params - added_params):.4%}")
    print(f"total_flops: {total_flops:.3e}")
    print(f"backbone_latency_ms: {backbone_time * 1000:.2f}")
    print(f"full_model_latency_ms: {full_time * 1000:.2f}")
    print(f"full_model_fps: {1.0 / full_time:.2f}")
    if device.type == "cuda":
        print(f"peak_gpu_memory_mb: {peak_mem_mb:.2f}")


if __name__ == "__main__":
    main()
