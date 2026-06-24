# Refined MFVPT Method Analysis

## Feasibility

The refinement is feasible and improves the method framing.

The previous implementation already used Fourier transforms mainly to build `x_low` and `x_high`, but the naming still made the method look broader than necessary. The refined design is cleaner:

- Fourier is only the frequency-view generator.
- The four training views are `x`, `x_low`, `x_high`, and `x_patch`.
- The post-training loss is one unified multi-view consistency objective.
- Frequency robustness remains in evaluation because low/high frequency response is part of the hypothesis.

This keeps Fourier visible as the core source of interpretable frequency views while avoiding a pile-up of separate Fourier-specific losses or penalties.

## Implemented Design

Training still has two stages:

1. Load an ImageNet pretrained ViT/DeiT from `timm` and fine-tune it on CIFAR-100.
2. Load the baseline checkpoint and run MFVPT post-training.

During post-training each batch builds:

```text
x
x_low    = low-pass frequency view from torch.fft.fft2
x_high   = high-frequency perturbed view from torch.fft.fft2
x_patch  = local meaningless patch view
```

The loss is:

```text
L = L_ce + lambda_view_consistency * L_view_consistency + lambda_confidence * L_confidence
```

Where:

- `L_ce` is averaged over the original and three perturbed views.
- `L_view_consistency` uses the original prediction distribution as a detached teacher and constrains `x_low`, `x_high`, and `x_patch`.
- `L_confidence` penalizes high-confidence mistakes on perturbed views, currently focused on patch views in the first version.

## Code Changes

- Added `mfvpt/losses/view_consistency.py`.
- Renamed training metric output from `loss_consistency` to `loss_view_consistency`.
- Renamed config weight from `lambda_consistency` to `lambda_view_consistency`.
- Kept backward compatibility for old configs that still contain `lambda_consistency`.
- Updated README and plan wording to describe Fourier as a view generator, not a separate loss family.

## Remaining Interpretation

Evaluation still reports `low_acc`, `high_acc`, and related consistency metrics. This is intentional: Fourier is narrowed in method design, but frequency robustness remains part of what the experiment is testing.
