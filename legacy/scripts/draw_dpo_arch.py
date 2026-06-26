"""Draw a high-level DPO detection post-training architecture diagram."""
from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path


def add_box(ax, x, y, w, h, text, color="white", edge="black", fontsize=10, bold=False):
    box = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.15",
        facecolor=color, edgecolor=edge, linewidth=1.5,
    )
    ax.add_patch(box)
    weight = "bold" if bold else "normal"
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, weight=weight, wrap=True)
    return box


def arrow(ax, x1, y1, x2, y2, color="black", style="->"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=1.2,
                                connectionstyle="arc3,rad=0"))


def main() -> None:
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # Title
    ax.text(7, 9.6, "DPO Detection Post-Training Architecture", fontsize=16, weight="bold", ha="center")

    # Colors
    c_input = "#E8F4F8"
    c_model = "#DEE8F7"
    c_action = "#FFF2CC"
    c_verifier = "#E1F5E1"
    c_loss = "#FCE4EC"
    c_update = "#F3E5F5"

    # Row 0: input
    add_box(ax, 7, 8.8, 2.0, 0.6, "Input Image", c_input)

    # Row 1: two detectors
    add_box(ax, 3.5, 7.6, 3.0, 0.8, "Frozen Baseline Detector\n(Faster R-CNN)", c_model, bold=True)
    add_box(ax, 10.5, 7.6, 3.0, 0.8, "Current Detector\n(trainable predictor)", c_model, bold=True)

    # proposals / roi outputs
    add_box(ax, 3.5, 6.3, 2.4, 0.7, "Top-Scoring Proposals", "#FFFFFF")
    add_box(ax, 10.5, 6.3, 2.6, 0.7, "ROI Head Outputs\n(logits + box deltas)", "#FFFFFF")

    # Row 2: action sampler
    add_box(ax, 7.0, 5.0, 4.2, 0.9,
            "Action Sampler\nmu  ->  Gaussian perturbations  ->  decoded boxes\nbaseline_mu  ->  reference log probs",
            c_action, bold=True)

    # Row 3: verifier
    add_box(ax, 7.0, 3.7, 4.4, 0.8,
            "Verifiers\nIoU  +  FFT / Manifold / raw_ifft_hd_fusion  ->  quality",
            c_verifier, bold=True)

    # Row 4: DPO pair builder
    add_box(ax, 7.0, 2.6, 3.6, 0.6, "DPO Pair Builder  (chosen vs rejected)", "#FFFFFF", bold=True)

    # Row 5: losses
    loss_y = 1.5
    add_box(ax, 2.0, loss_y, 2.0, 0.6, "DPO Loss\n(logsigmoid margin)", c_loss)
    add_box(ax, 4.5, loss_y, 1.8, 0.6, "KL Loss\n(logits + boxes)", c_loss)
    add_box(ax, 6.7, loss_y, 1.6, 0.6, "Det Loss\n(Faster R-CNN)", c_loss)
    add_box(ax, 8.8, loss_y, 1.8, 0.6, "Confidence\nCorrection", c_loss)
    add_box(ax, 11.0, loss_y, 1.8, 0.6, "Rescue Loss\n(LC-HI score)", c_loss)

    # Row 6: update
    add_box(ax, 7.0, 0.5, 3.0, 0.6, "Update Current Predictor", c_update, bold=True)

    # Arrows
    arrow(ax, 7.0, 8.5, 3.5, 8.0)
    arrow(ax, 7.0, 8.5, 10.5, 8.0)

    arrow(ax, 3.5, 7.2, 3.5, 6.65)
    arrow(ax, 10.5, 7.2, 10.5, 6.65)

    arrow(ax, 3.5, 5.95, 6.0, 5.45)   # proposals -> sampler
    arrow(ax, 10.5, 5.95, 8.0, 5.45)  # roi outputs -> sampler

    arrow(ax, 7.0, 4.55, 7.0, 4.1)    # sampler -> verifier
    arrow(ax, 7.0, 3.3, 7.0, 2.9)     # verifier -> pair builder

    arrow(ax, 5.3, 2.6, 3.0, 1.8)     # pair -> DPO loss
    arrow(ax, 8.7, 2.6, 11.0, 1.8)    # pair -> rescue (simplified)

    # losses -> update
    arrow(ax, 2.0, 1.2, 5.5, 0.8)
    arrow(ax, 4.5, 1.2, 5.8, 0.8)
    arrow(ax, 6.7, 1.2, 6.5, 0.8)
    arrow(ax, 8.8, 1.2, 7.5, 0.8)
    arrow(ax, 11.0, 1.2, 8.5, 0.8)

    # Legend / note
    ax.text(0.5, 0.15,
            "Note: Baseline provides proposals and reference log-probabilities.\n"
            "Current model is updated only via predictor/adapter parameters.",
            fontsize=9, va="bottom", color="#555555")

    plt.tight_layout()
    out = Path("output/dpo_architecture.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
