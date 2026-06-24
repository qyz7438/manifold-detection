from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("runs/.matplotlib").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from spectral_detection_posttrain.core.models.spectral_quality_head import (
    SpectralQualityHead,
    build_quality_features,
    normalize_r_amp,
)
from spectral_detection_posttrain.signals.fft.roi_spectral_dataset import load_candidate_cache
from spectral_detection_posttrain.utils.config import load_config, save_config
from spectral_detection_posttrain.utils.io import ensure_run_dir
from spectral_detection_posttrain.utils.seed import resolve_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize spectral quality distributions and amplitude profiles.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--quality-checkpoint", default=None)
    return parser.parse_args()


def _load_head(path: str, device: torch.device) -> tuple[SpectralQualityHead, dict]:
    checkpoint = torch.load(path, map_location=device)
    metadata = checkpoint["metadata"]
    head = SpectralQualityHead(
        input_dim=int(metadata["input_dim"]),
        hidden_dim=int(metadata.get("quality_head", {}).get("hidden_dim", 256)),
        dropout=float(metadata.get("quality_head", {}).get("dropout", 0.1)),
    ).to(device)
    head.load_state_dict(checkpoint["model"])
    head.eval()
    return head, metadata


@torch.no_grad()
def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config.get("seed", 42)))
    run_dir = ensure_run_dir(args.run_name)
    save_config(config, run_dir / "config.yaml")
    payload = load_candidate_cache(args.candidates)

    device = resolve_device(config)
    head = None
    metadata = None
    if args.quality_checkpoint:
        head, metadata = _load_head(args.quality_checkpoint, device)

    q_tp = []
    q_fp = []
    amp_tp = []
    amp_fp = []
    raw_values = []
    for sample in payload["samples"]:
        if len(sample["scores"]) == 0:
            continue
        is_tp = sample["is_tp"].bool()
        amp_tp.append(sample["amp_profiles"][is_tp])
        amp_fp.append(sample["amp_profiles"][~is_tp])
        if head is not None and metadata is not None:
            features = build_quality_features(sample, metadata["feature_mode"]).to(device)
            q_spec = head.predict_quality(features).detach().cpu()
        else:
            stats = {"mode": "minmax", "min": float(sample["raw_r_amp"].min().item()), "max": float(sample["raw_r_amp"].max().item())}
            q_spec = normalize_r_amp(sample["raw_r_amp"].float(), stats)
        q_tp.extend(q_spec[is_tp].tolist())
        q_fp.extend(q_spec[~is_tp].tolist())
        raw_values.extend(sample["raw_r_amp"].tolist())

    fig, ax = plt.subplots(figsize=(6, 4))
    if q_tp:
        ax.hist(q_tp, bins=12, alpha=0.65, label="TP q")
    if q_fp:
        ax.hist(q_fp, bins=12, alpha=0.65, label="FP q")
    ax.set_xlabel("quality score")
    ax.set_ylabel("count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "q_spec_distribution.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    if amp_tp and torch.cat(amp_tp).numel() > 0:
        ax.plot(torch.cat(amp_tp).mean(dim=0), label="TP amp profile")
    if amp_fp and torch.cat(amp_fp).numel() > 0:
        ax.plot(torch.cat(amp_fp).mean(dim=0), label="FP amp profile")
    ax.set_xlabel("radial frequency bin")
    ax.set_ylabel("normalized amplitude")
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "amp_profile_tp_fp.png", dpi=160)
    plt.close(fig)

    print(
        {
            "q_spec_distribution": str(run_dir / "q_spec_distribution.png"),
            "amp_profile_tp_fp": str(run_dir / "amp_profile_tp_fp.png"),
            "num_q_tp": len(q_tp),
            "num_q_fp": len(q_fp),
        }
    )


if __name__ == "__main__":
    main()
