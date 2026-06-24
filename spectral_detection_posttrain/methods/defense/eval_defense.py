r"""Evaluation metrics for adversarial defense.

The functions work with scalar scores returned by a detector or any mock loss.
Higher scores are assumed to mean better detection performance.
"""

from __future__ import annotations

import torch


def _to_tensor(value: float | torch.Tensor, device: torch.device | None = None) -> torch.Tensor:
    """Convert a scalar or tensor to a floating-point tensor."""
    if isinstance(value, torch.Tensor):
        return value.float().to(device) if device is not None else value.float()
    return torch.tensor(float(value), dtype=torch.float32, device=device)


def defense_success_rate(
    clean_score: float | torch.Tensor,
    adversarial_score: float | torch.Tensor,
    defended_score: float | torch.Tensor,
) -> torch.Tensor:
    r"""Measure how much of the adversarial degradation is recovered.

    The recovery ratio is

    .. math::
        r = \frac{s_{\mathrm{defended}} - s_{\mathrm{adv}}}
                 {s_{\mathrm{clean}} - s_{\mathrm{adv}}}

    and is clipped to :math:`[0, 1]`.  A value of ``1.0`` means full recovery,
    ``0.0`` means no recovery.

    Args:
        clean_score: detector score on clean inputs.
        adversarial_score: detector score on adversarial inputs.
        defended_score: detector score on defended adversarial inputs.

    Returns:
        Scalar recovery ratio.
    """
    clean = _to_tensor(clean_score)
    adv = _to_tensor(adversarial_score, device=clean.device)
    def_ = _to_tensor(defended_score, device=clean.device)
    denom = clean - adv
    # If there is no degradation, any defended score at least as good as clean
    # counts as full success.
    if abs(denom.item()) < 1e-8:
        return torch.ones_like(clean)
    recovery = (def_ - adv) / denom
    return torch.clamp(recovery, 0.0, 1.0)


def clean_accuracy_drop(
    clean_score: float | torch.Tensor,
    defended_clean_score: float | torch.Tensor,
) -> torch.Tensor:
    r"""Relative drop in clean-sample performance caused by the defense.

    Computes

    .. math::
        d = \frac{s_{\mathrm{clean}} - s_{\mathrm{defended}}}
                 {s_{\mathrm{clean}}}

    Args:
        clean_score: detector score on clean inputs.
        defended_clean_score: detector score on clean inputs after defense.

    Returns:
        Scalar relative drop.  Negative values indicate improvement.
    """
    clean = _to_tensor(clean_score)
    defended = _to_tensor(defended_clean_score, device=clean.device)
    denom = clean.abs().clamp_min(1e-8)
    return (clean - defended) / denom


def robust_accuracy(
    defended_score: float | torch.Tensor,
    threshold: float | torch.Tensor,
) -> torch.Tensor:
    r"""Binary robust accuracy: fraction of defended inputs above threshold.

    Args:
        defended_score: detector score(s) after defense.
        threshold: decision threshold.

    Returns:
        Scalar accuracy in :math:`[0, 1]`.
    """
    defended = _to_tensor(defended_score)
    thresh = _to_tensor(threshold, device=defended.device)
    return (defended >= thresh).float().mean()
