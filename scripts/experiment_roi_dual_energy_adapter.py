"""Offline final-head adapter experiment with ROI dual-energy regularization.

The cache does not contain final boxes/NMS state, so this is not a detector AP
experiment.  It tests a narrower question: can a small trainable final-head
adapter/scorer improve AP75 candidate classification when regularized by the
intra/inter dual-energy objective?
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spectral_detection_posttrain.methods.energy_transport import roi_dual_energy  # noqa: E402
from spectral_detection_posttrain.methods.energy_transport.cone_projection import (  # noqa: E402
    compute_class_prototypes,
    normalize_l2,
)


@dataclass(frozen=True)
class VariantConfig:
    name: str
    energy: str
    mask: str
    lambda_energy: float


@dataclass(frozen=True)
class VariantResult:
    seed: int
    variant: str
    energy: str
    mask: str
    lambda_energy: float
    train_loss: float
    train_bce: float
    train_energy: float
    train_auc: float
    train_ap: float
    train_brier: float
    train_ece: float
    train_precision_at_pos: float
    val_auc: float
    val_ap: float
    val_brier: float
    val_ece: float
    val_precision_at_pos: float
    val_recall_at_score_050: float
    val_pred_count_050: int
    val_energy_intra_all: float
    val_energy_inter_all: float
    val_energy_dual_all: float
    val_energy_intra_pos: float | None
    val_energy_inter_pos: float | None
    val_energy_dual_pos: float | None
    residual_logit_scale: float
    embed_gate: float
    train_abs_logit_delta_mean: float
    train_abs_prob_delta_mean: float
    train_embedding_step_mean: float
    val_abs_logit_delta_mean: float
    val_abs_prob_delta_mean: float
    val_prob_delta_mean: float
    val_embedding_step_mean: float
    val_score_iou_corr: float
    val_prior_iou_corr: float
    val_delta_iou_corr: float
    train_threshold_metrics: dict[str, dict[str, float | int | None]]
    train_topk_metrics: dict[str, dict[str, float | int | None]]
    val_threshold_metrics: dict[str, dict[str, float | int | None]]
    val_topk_metrics: dict[str, dict[str, float | int | None]]
    val_energy_groups: dict[str, dict[str, float | int | None]]
    val_group_metrics: dict[str, dict[str, float | int | None]]
    history: list[dict[str, float]]


def bounded_logit(value: float, eps: float = 1e-4) -> torch.Tensor:
    clipped = min(max(float(value), eps), 1.0 - eps)
    return torch.tensor(math.log(clipped / (1.0 - clipped)), dtype=torch.float32)


class ResidualAdapterScorer(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden: int = 64,
        max_step: float = 0.25,
        residual_scale_init: float = 0.05,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.adapter = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )
        self.gate_logit = nn.Parameter(torch.tensor(-4.0))
        self.max_step = float(max_step)
        self.scorer = nn.Linear(dim, 1)
        self.residual_logit_scale = nn.Parameter(bounded_logit(residual_scale_init))

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        delta = torch.tanh(self.adapter(self.norm(x)))
        gate = self.max_step * torch.sigmoid(self.gate_logit)
        return normalize_l2(x + gate * delta)

    def forward(self, x: torch.Tensor, prior_logit: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.embed(x)
        residual = torch.sigmoid(self.residual_logit_scale) * self.scorer(z).squeeze(-1)
        if prior_logit is not None:
            return prior_logit + residual, z
        return residual, z


def labels_zero_based(class_ids: np.ndarray) -> np.ndarray:
    class_ids = class_ids.astype(np.int64)
    if class_ids.size and class_ids.min() >= 1:
        return class_ids - 1
    return class_ids


def auc_score(y_true: np.ndarray, scores: np.ndarray) -> float:
    y = y_true.astype(bool)
    s = scores.astype(float)
    pos_count = int(y.sum())
    neg_count = int((~y).sum())
    if pos_count == 0 or neg_count == 0:
        return float("nan")
    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1, dtype=np.float64)
    sorted_scores = s[order]
    start = 0
    while start < len(sorted_scores):
        end = start + 1
        while end < len(sorted_scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        if end - start > 1:
            ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    rank_sum = ranks[y].sum()
    return float((rank_sum - pos_count * (pos_count + 1) / 2.0) / (pos_count * neg_count))


def average_precision(y_true: np.ndarray, scores: np.ndarray) -> float:
    y = y_true.astype(bool)
    if int(y.sum()) == 0:
        return float("nan")
    order = np.argsort(-scores.astype(float))
    ordered = y[order]
    tp = np.cumsum(ordered)
    precision = tp / (np.arange(len(ordered)) + 1)
    return float((precision * ordered).sum() / max(1, int(ordered.sum())))


def brier_score(y_true: np.ndarray, probs: np.ndarray) -> float:
    y = y_true.astype(np.float64)
    p = probs.astype(np.float64)
    return float(np.mean((p - y) ** 2))


def ece_score(y_true: np.ndarray, probs: np.ndarray, bins: int = 10) -> float:
    y = y_true.astype(np.float64)
    p = probs.astype(np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(y)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi == 1.0:
            mask = (p >= lo) & (p <= hi)
        else:
            mask = (p >= lo) & (p < hi)
        if not mask.any():
            continue
        ece += float(mask.mean()) * abs(float(p[mask].mean()) - float(y[mask].mean()))
    return float(ece if total else 0.0)


def precision_at_pos_count(y_true: np.ndarray, scores: np.ndarray) -> float:
    y = y_true.astype(bool)
    k = int(y.sum())
    if k <= 0:
        return float("nan")
    order = np.argsort(-scores.astype(float))[:k]
    return float(y[order].mean())


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    if a.size < 2 or float(np.std(a)) < 1e-12 or float(np.std(b)) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def safe_mean(values: np.ndarray) -> float | None:
    if values.size == 0:
        return None
    return float(np.mean(values))


def safe_rate(numer: int, denom: int) -> float | None:
    if denom <= 0:
        return None
    return float(numer / denom)


def threshold_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    iou: np.ndarray,
    *,
    thresholds: tuple[float, ...] = (0.05, 0.10, 0.20, 0.30, 0.40, 0.50),
) -> dict[str, dict[str, float | int | None]]:
    y = y_true.astype(bool)
    rows: dict[str, dict[str, float | int | None]] = {}
    for threshold in thresholds:
        pred = scores >= float(threshold)
        tp = int((pred & y).sum())
        fp = int((pred & ~y).sum())
        fn = int((~pred & y).sum())
        pred_count = int(pred.sum())
        rows[f"score_ge_{threshold:.2f}"] = {
            "threshold": float(threshold),
            "pred_count": pred_count,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": safe_rate(tp, pred_count),
            "recall": safe_rate(tp, int(y.sum())),
            "mean_iou_pred": safe_mean(iou[pred]),
        }
    return rows


def topk_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    iou: np.ndarray,
) -> dict[str, dict[str, float | int | None]]:
    y = y_true.astype(bool)
    pos_count = int(y.sum())
    candidates = {
        "top_pos_count": pos_count,
        "top_2x_pos_count": 2 * pos_count,
        "top_50": 50,
        "top_100": 100,
    }
    order = np.argsort(-scores.astype(float))
    rows: dict[str, dict[str, float | int | None]] = {}
    for name, raw_k in candidates.items():
        k = min(max(int(raw_k), 0), len(scores))
        if k <= 0:
            rows[name] = {"k": k, "tp": 0, "precision": None, "recall": None, "mean_iou": None}
            continue
        selected = order[:k]
        tp = int(y[selected].sum())
        rows[name] = {
            "k": k,
            "tp": tp,
            "fp": int(k - tp),
            "precision": float(tp / k),
            "recall": safe_rate(tp, pos_count),
            "mean_iou": float(iou[selected].mean()),
        }
    return rows


def group_masks(
    y_true: np.ndarray,
    iou: np.ndarray,
    prior_probs: np.ndarray | None,
    scores: np.ndarray,
) -> dict[str, np.ndarray]:
    ranking_score = prior_probs if prior_probs is not None else scores
    y = y_true.astype(bool)
    return {
        "all": np.ones_like(y, dtype=bool),
        "ap75_positive": y,
        "ap75_negative": ~y,
        "low_conf_high_iou": (ranking_score < 0.50) & (iou >= 0.75),
        "very_low_conf_high_iou": (ranking_score < 0.30) & (iou >= 0.75),
        "rescue_band_030_050_high_iou": (ranking_score >= 0.30) & (ranking_score < 0.50) & (iou >= 0.75),
        "risky_score_ge_030_low_iou": (ranking_score >= 0.30) & (iou < 0.50),
        "risky_score_ge_020_low_iou": (ranking_score >= 0.20) & (iou < 0.50),
        "high_conf_low_iou": (ranking_score >= 0.50) & (iou < 0.50),
        "borderline_iou": (iou >= 0.50) & (iou < 0.75),
    }


def group_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    iou: np.ndarray,
    *,
    prior_probs: np.ndarray | None,
    prob_delta: np.ndarray,
) -> dict[str, dict[str, float | int | None]]:
    rows: dict[str, dict[str, float | int | None]] = {}
    for name, mask in group_masks(y_true, iou, prior_probs, scores).items():
        count = int(mask.sum())
        if count == 0:
            rows[name] = {"count": 0}
            continue
        y_group = y_true[mask].astype(bool)
        scores_group = scores[mask]
        pred_050 = scores_group >= 0.50
        tp_050 = int((pred_050 & y_group).sum())
        pred_count_050 = int(pred_050.sum())
        rows[name] = {
            "count": count,
            "ap75_positive_count": int(y_group.sum()),
            "ap75_positive_rate": float(y_group.mean()),
            "mean_iou": float(iou[mask].mean()),
            "mean_score": float(scores_group.mean()),
            "mean_prior": safe_mean(prior_probs[mask]) if prior_probs is not None else None,
            "mean_prob_delta": float(prob_delta[mask].mean()),
            "mean_abs_prob_delta": float(np.abs(prob_delta[mask]).mean()),
            "pred_count_050": pred_count_050,
            "tp_050": tp_050,
            "fp_050": int((pred_050 & ~y_group).sum()),
            "precision_050": safe_rate(tp_050, pred_count_050),
            "recall_050": safe_rate(tp_050, int(y_group.sum())),
        }
    return rows


def build_reference_prototypes(
    features: torch.Tensor,
    class_labels: torch.Tensor,
    quality_labels: torch.Tensor,
    *,
    mode: str,
    num_classes: int,
) -> torch.Tensor:
    if mode not in {"all", "positive_fallback"}:
        raise ValueError("reference mode must be 'all' or 'positive_fallback'")
    if mode == "all":
        return compute_class_prototypes(features, class_labels, num_classes=num_classes)

    prototypes = []
    for class_idx in range(num_classes):
        class_mask = class_labels == class_idx
        pos_mask = class_mask & quality_labels.bool()
        use_mask = pos_mask if bool(pos_mask.any()) else class_mask
        if bool(use_mask.any()):
            proto = normalize_l2(features[use_mask]).mean(dim=0)
            prototypes.append(normalize_l2(proto))
        else:
            prototypes.append(torch.zeros(features.shape[1], device=features.device, dtype=features.dtype))
    return torch.stack(prototypes, dim=0)


def mask_for_variant(quality_labels: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "all":
        return torch.ones_like(quality_labels, dtype=torch.bool)
    if mode == "positive":
        return quality_labels.bool()
    raise ValueError(f"Unknown mask mode: {mode}")


def remap_for_mask(
    features: torch.Tensor,
    class_labels: torch.Tensor,
    mask: torch.Tensor,
    reference: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    present = torch.unique(class_labels[mask]).sort().values
    if int(present.numel()) < 2:
        return None
    mapping = torch.full((reference.shape[0],), -1, dtype=torch.long, device=features.device)
    mapping[present] = torch.arange(present.numel(), device=features.device)
    local_labels = mapping[class_labels[mask]]
    return features[mask], local_labels, reference[present]


def dual_energy_loss(
    features: torch.Tensor,
    class_labels: torch.Tensor,
    quality_labels: torch.Tensor,
    reference: torch.Tensor,
    *,
    mode: str,
    energy_kind: str,
) -> torch.Tensor:
    if energy_kind == "none":
        return features.new_tensor(0.0)
    mask = mask_for_variant(quality_labels, mode)
    remapped = remap_for_mask(features, class_labels, mask, reference)
    if remapped is None:
        return features.new_tensor(0.0)
    selected_features, selected_labels, selected_reference = remapped
    energy = roi_dual_energy(
        selected_features,
        selected_labels,
        reference_prototypes=selected_reference,
        perturb_radius=0.0,
        num_perturbations=0,
        anchor_weight=0.25,
        separation_weight=0.10,
    )
    if energy_kind == "intra":
        return energy.intra_energy
    if energy_kind == "dual":
        return energy.dual_energy
    raise ValueError(f"Unknown energy kind: {energy_kind}")


@torch.no_grad()
def eval_energy(
    features: torch.Tensor,
    class_labels: torch.Tensor,
    quality_labels: torch.Tensor,
    reference: torch.Tensor,
    *,
    mask_mode: str,
) -> tuple[float | None, float | None, float | None]:
    remapped = remap_for_mask(features, class_labels, mask_for_variant(quality_labels, mask_mode), reference)
    if remapped is None:
        return None, None, None
    selected_features, selected_labels, selected_reference = remapped
    energy = roi_dual_energy(
        selected_features,
        selected_labels,
        reference_prototypes=selected_reference,
        perturb_radius=0.0,
        num_perturbations=0,
        anchor_weight=0.25,
        separation_weight=0.10,
    )
    return (
        float(energy.intra_energy.item()),
        float(energy.inter_energy.item()),
        float(energy.dual_energy.item()),
    )


@torch.no_grad()
def eval_energy_components_for_mask(
    features: torch.Tensor,
    class_labels: torch.Tensor,
    mask: torch.Tensor,
    reference: torch.Tensor,
) -> dict[str, float | int | None]:
    out: dict[str, float | int | None] = {
        "count": int(mask.sum().item()),
        "class_count": int(torch.unique(class_labels[mask]).numel()) if bool(mask.any()) else 0,
    }
    remapped = remap_for_mask(features, class_labels, mask, reference)
    if remapped is None:
        out.update(
            {
                "e_compact": None,
                "e_basin": None,
                "e_intra": None,
                "e_inter": None,
                "dual_energy": None,
            }
        )
        return out
    selected_features, selected_labels, selected_reference = remapped
    energy = roi_dual_energy(
        selected_features,
        selected_labels,
        reference_prototypes=selected_reference,
        perturb_radius=0.0,
        num_perturbations=0,
        anchor_weight=0.25,
        separation_weight=0.10,
    )
    for key, value in energy.components.items():
        out[key] = float(value.detach().item())
    return out


@torch.no_grad()
def energy_groups(
    features: torch.Tensor,
    class_labels: torch.Tensor,
    quality_labels: torch.Tensor,
    iou: torch.Tensor,
    prior_probs: torch.Tensor | None,
    scores: torch.Tensor,
    reference: torch.Tensor,
) -> dict[str, dict[str, float | int | None]]:
    y_np = quality_labels.detach().cpu().numpy().astype(bool)
    iou_np = iou.detach().cpu().numpy()
    score_np = scores.detach().cpu().numpy()
    prior_np = prior_probs.detach().cpu().numpy() if prior_probs is not None else None
    groups = group_masks(y_np, iou_np, prior_np, score_np)
    result: dict[str, dict[str, float | int | None]] = {}
    for name, mask_np in groups.items():
        mask = torch.from_numpy(mask_np).to(device=features.device, dtype=torch.bool)
        result[name] = eval_energy_components_for_mask(features, class_labels, mask, reference)
    return result


@torch.no_grad()
def split_diagnostics(
    *,
    x: torch.Tensor,
    z: torch.Tensor,
    logits: torch.Tensor,
    prior_logits: torch.Tensor | None,
    labels: torch.Tensor,
    iou: torch.Tensor,
    class_labels: torch.Tensor,
    reference: torch.Tensor,
) -> dict[str, object]:
    probs_t = torch.sigmoid(logits)
    prior_probs_t = torch.sigmoid(prior_logits) if prior_logits is not None else None
    logit_delta_t = logits - prior_logits if prior_logits is not None else logits
    prob_delta_t = probs_t - prior_probs_t if prior_probs_t is not None else probs_t

    probs = probs_t.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy().astype(bool)
    iou_np = iou.detach().cpu().numpy()
    prior_probs = prior_probs_t.detach().cpu().numpy() if prior_probs_t is not None else None
    logit_delta = logit_delta_t.detach().cpu().numpy()
    prob_delta = prob_delta_t.detach().cpu().numpy()
    original = normalize_l2(x)
    embedding_step = torch.linalg.vector_norm(z - original, dim=1).detach().cpu().numpy()

    return {
        "auc": auc_score(labels_np, probs),
        "ap": average_precision(labels_np, probs),
        "brier": brier_score(labels_np, probs),
        "ece": ece_score(labels_np, probs),
        "precision_at_pos": precision_at_pos_count(labels_np, probs),
        "score_iou_corr": safe_corr(probs, iou_np),
        "prior_iou_corr": safe_corr(prior_probs, iou_np) if prior_probs is not None else float("nan"),
        "delta_iou_corr": safe_corr(prob_delta, iou_np),
        "abs_logit_delta_mean": float(np.abs(logit_delta).mean()),
        "abs_logit_delta_p95": float(np.quantile(np.abs(logit_delta), 0.95)),
        "prob_delta_mean": float(prob_delta.mean()),
        "abs_prob_delta_mean": float(np.abs(prob_delta).mean()),
        "abs_prob_delta_p95": float(np.quantile(np.abs(prob_delta), 0.95)),
        "embedding_step_mean": float(embedding_step.mean()),
        "embedding_step_p95": float(np.quantile(embedding_step, 0.95)),
        "thresholds": threshold_metrics(labels_np, probs, iou_np),
        "topk": topk_metrics(labels_np, probs, iou_np),
        "groups": group_metrics(labels_np, probs, iou_np, prior_probs=prior_probs, prob_delta=prob_delta),
        "energy_groups": energy_groups(
            z,
            class_labels,
            labels,
            iou,
            prior_probs_t,
            probs_t,
            reference,
        ),
    }


def train_variant(
    config: VariantConfig,
    *,
    seed: int,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    train_class: torch.Tensor,
    train_iou: torch.Tensor,
    train_prior: torch.Tensor | None,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    val_class: torch.Tensor,
    val_iou: torch.Tensor,
    val_prior: torch.Tensor | None,
    reference: torch.Tensor,
    epochs: int,
    lr: float,
    hidden: int,
    max_step: float,
    residual_scale_init: float,
    weight_decay: float,
    pos_weight_mode: str,
    log_every: int,
) -> VariantResult:
    torch.manual_seed(seed)
    model = ResidualAdapterScorer(
        train_x.shape[1],
        hidden=hidden,
        max_step=max_step,
        residual_scale_init=residual_scale_init,
    ).to(train_x.device)
    if pos_weight_mode == "balanced":
        pos = float(train_y.sum().item())
        neg = float(train_y.numel() - train_y.sum().item())
        pos_weight = torch.tensor([neg / max(pos, 1.0)], device=train_x.device)
    elif pos_weight_mode == "none":
        pos_weight = None
    else:
        raise ValueError("pos_weight_mode must be 'balanced' or 'none'")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    final_loss = final_bce = final_energy = 0.0
    history: list[dict[str, float]] = []
    total_epochs = int(epochs)
    for epoch in range(total_epochs):
        logits, z = model(train_x, train_prior)
        bce = F.binary_cross_entropy_with_logits(logits, train_y.float(), pos_weight=pos_weight)
        energy_loss = dual_energy_loss(
            z,
            train_class,
            train_y,
            reference,
            mode=config.mask,
            energy_kind=config.energy,
        )
        loss = bce + float(config.lambda_energy) * energy_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        final_loss = float(loss.detach().item())
        final_bce = float(bce.detach().item())
        final_energy = float(energy_loss.detach().item())
        should_log = log_every > 0 and (
            epoch == 0 or epoch + 1 == total_epochs or (epoch + 1) % int(log_every) == 0
        )
        if should_log:
            with torch.no_grad():
                train_probs = torch.sigmoid(logits).detach().cpu().numpy()
                train_y_np = train_y.detach().cpu().numpy().astype(bool)
                val_logits_probe, _ = model(val_x, val_prior)
                val_probs_probe = torch.sigmoid(val_logits_probe).detach().cpu().numpy()
                val_y_np_probe = val_y.detach().cpu().numpy().astype(bool)
                residual_t = logits - train_prior if train_prior is not None else logits
                history.append(
                    {
                        "epoch": float(epoch + 1),
                        "loss": final_loss,
                        "bce": final_bce,
                        "energy": final_energy,
                        "train_auc": auc_score(train_y_np, train_probs),
                        "train_ap": average_precision(train_y_np, train_probs),
                        "val_auc": auc_score(val_y_np_probe, val_probs_probe),
                        "val_ap": average_precision(val_y_np_probe, val_probs_probe),
                        "abs_logit_delta_mean": float(residual_t.abs().mean().item()),
                        "embed_gate": float((model.max_step * torch.sigmoid(model.gate_logit)).item()),
                        "residual_logit_scale": float(torch.sigmoid(model.residual_logit_scale).item()),
                    }
                )

    model.eval()
    with torch.no_grad():
        train_logits, train_z = model(train_x, train_prior)
        val_logits, val_z = model(val_x, val_prior)
        train_diag = split_diagnostics(
            x=train_x,
            z=train_z,
            logits=train_logits,
            prior_logits=train_prior,
            labels=train_y,
            iou=train_iou,
            class_labels=train_class,
            reference=reference,
        )
        val_diag = split_diagnostics(
            x=val_x,
            z=val_z,
            logits=val_logits,
            prior_logits=val_prior,
            labels=val_y,
            iou=val_iou,
            class_labels=val_class,
            reference=reference,
        )
    val_energy_groups = val_diag["energy_groups"]  # type: ignore[index]
    val_group_metrics = val_diag["groups"]  # type: ignore[index]
    all_energy = val_energy_groups["all"]  # type: ignore[index]
    pos_energy = val_energy_groups["ap75_positive"]  # type: ignore[index]
    val_threshold_050 = val_diag["thresholds"]["score_ge_0.50"]  # type: ignore[index]
    return VariantResult(
        seed=seed,
        variant=config.name,
        energy=config.energy,
        mask=config.mask,
        lambda_energy=float(config.lambda_energy),
        train_loss=final_loss,
        train_bce=final_bce,
        train_energy=final_energy,
        train_auc=float(train_diag["auc"]),
        train_ap=float(train_diag["ap"]),
        train_brier=float(train_diag["brier"]),
        train_ece=float(train_diag["ece"]),
        train_precision_at_pos=float(train_diag["precision_at_pos"]),
        val_auc=float(val_diag["auc"]),
        val_ap=float(val_diag["ap"]),
        val_brier=float(val_diag["brier"]),
        val_ece=float(val_diag["ece"]),
        val_precision_at_pos=float(val_diag["precision_at_pos"]),
        val_recall_at_score_050=float(val_threshold_050["recall"] or 0.0),
        val_pred_count_050=int(val_threshold_050["pred_count"]),
        val_energy_intra_all=float(all_energy["e_intra"]) if all_energy["e_intra"] is not None else math.nan,
        val_energy_inter_all=float(all_energy["e_inter"]) if all_energy["e_inter"] is not None else math.nan,
        val_energy_dual_all=float(all_energy["dual_energy"]) if all_energy["dual_energy"] is not None else math.nan,
        val_energy_intra_pos=pos_energy["e_intra"],
        val_energy_inter_pos=pos_energy["e_inter"],
        val_energy_dual_pos=pos_energy["dual_energy"],
        residual_logit_scale=float(torch.sigmoid(model.residual_logit_scale).item()),
        embed_gate=float((model.max_step * torch.sigmoid(model.gate_logit)).item()),
        train_abs_logit_delta_mean=float(train_diag["abs_logit_delta_mean"]),
        train_abs_prob_delta_mean=float(train_diag["abs_prob_delta_mean"]),
        train_embedding_step_mean=float(train_diag["embedding_step_mean"]),
        val_abs_logit_delta_mean=float(val_diag["abs_logit_delta_mean"]),
        val_abs_prob_delta_mean=float(val_diag["abs_prob_delta_mean"]),
        val_prob_delta_mean=float(val_diag["prob_delta_mean"]),
        val_embedding_step_mean=float(val_diag["embedding_step_mean"]),
        val_score_iou_corr=float(val_diag["score_iou_corr"]),
        val_prior_iou_corr=float(val_diag["prior_iou_corr"]),
        val_delta_iou_corr=float(val_diag["delta_iou_corr"]),
        train_threshold_metrics=train_diag["thresholds"],  # type: ignore[arg-type]
        train_topk_metrics=train_diag["topk"],  # type: ignore[arg-type]
        val_threshold_metrics=val_diag["thresholds"],  # type: ignore[arg-type]
        val_topk_metrics=val_diag["topk"],  # type: ignore[arg-type]
        val_energy_groups=val_energy_groups,  # type: ignore[arg-type]
        val_group_metrics=val_group_metrics,  # type: ignore[arg-type]
        history=history,
    )


def load_cache(cache: Path, feature_key: str, device: torch.device) -> dict[str, torch.Tensor]:
    data = np.load(cache, allow_pickle=False)
    result = {
        "train_x": torch.from_numpy(data[f"train_{feature_key}"].astype(np.float32)).to(device),
        "val_x": torch.from_numpy(data[f"val_{feature_key}"].astype(np.float32)).to(device),
        "train_y": torch.from_numpy(data["train_labels"].astype(np.float32)).to(device),
        "val_y": torch.from_numpy(data["val_labels"].astype(np.float32)).to(device),
        "train_iou": torch.from_numpy(data["train_best_iou"].astype(np.float32)).to(device),
        "val_iou": torch.from_numpy(data["val_best_iou"].astype(np.float32)).to(device),
        "train_class": torch.from_numpy(labels_zero_based(data["train_class_ids"])).long().to(device),
        "val_class": torch.from_numpy(labels_zero_based(data["val_class_ids"])).long().to(device),
        "train_label_prob": torch.from_numpy(data["train_label_probs"].astype(np.float32)).to(device),
        "val_label_prob": torch.from_numpy(data["val_label_probs"].astype(np.float32)).to(device),
    }
    return result


def prior_logits_from_probs(probs: torch.Tensor) -> torch.Tensor:
    probs = probs.clamp(1e-5, 1.0 - 1e-5)
    return torch.logit(probs)


def default_variants() -> list[VariantConfig]:
    return [
        VariantConfig("bce_only", "none", "all", 0.0),
        VariantConfig("intra_all_l005", "intra", "all", 0.05),
        VariantConfig("dual_all_l005", "dual", "all", 0.05),
        VariantConfig("intra_pos_l005", "intra", "positive", 0.05),
        VariantConfig("dual_pos_l005", "dual", "positive", 0.05),
        VariantConfig("dual_pos_l020", "dual", "positive", 0.20),
    ]


def summarize(results: list[VariantResult]) -> list[dict]:
    names = list(dict.fromkeys(r.variant for r in results))
    fields = [
        "train_auc",
        "train_ap",
        "val_auc",
        "val_ap",
        "val_brier",
        "val_ece",
        "val_precision_at_pos",
        "val_energy_dual_all",
        "val_abs_prob_delta_mean",
        "val_embedding_step_mean",
        "val_score_iou_corr",
        "val_delta_iou_corr",
        "val_pred_count_050",
    ]
    rows = []
    for name in names:
        sub = [r for r in results if r.variant == name]
        row = {"variant": name, "n": len(sub)}
        for field in fields:
            values = np.array([getattr(r, field) for r in sub], dtype=np.float64)
            row[f"{field}_mean"] = float(np.nanmean(values))
            row[f"{field}_std"] = float(np.nanstd(values))
        for group_name in ("low_conf_high_iou", "rescue_band_030_050_high_iou", "risky_score_ge_030_low_iou"):
            for metric_name in ("count", "mean_score", "mean_prob_delta", "pred_count_050", "tp_050", "fp_050"):
                values = []
                for result in sub:
                    value = result.val_group_metrics.get(group_name, {}).get(metric_name)
                    if value is not None:
                        values.append(float(value))
                arr = np.array(values, dtype=np.float64)
                key = f"{group_name}_{metric_name}"
                row[f"{key}_mean"] = float(np.nanmean(arr)) if arr.size else math.nan
                row[f"{key}_std"] = float(np.nanstd(arr)) if arr.size else math.nan
        rows.append(row)
    return rows


def print_summary(rows: list[dict]) -> None:
    print("variant | val_auc | val_ap | p@pos | pred50 | val_ece | dprob | step | corr(score,iou) | corr(delta,iou) | lowHIoU dprob | band030-050 dprob | risky030 dprob | val_dual")
    print("---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:")
    for row in rows:
        mean = lambda field: row[f"{field}_mean"]  # noqa: E731
        std = lambda field: row[f"{field}_std"]  # noqa: E731
        print(
            f"{row['variant']} | "
            f"{mean('val_auc'):.4f} +/- {std('val_auc'):.4f} | "
            f"{mean('val_ap'):.4f} +/- {std('val_ap'):.4f} | "
            f"{mean('val_precision_at_pos'):.4f} | "
            f"{mean('val_pred_count_050'):.1f} | "
            f"{mean('val_ece'):.4f} +/- {std('val_ece'):.4f} | "
            f"{mean('val_abs_prob_delta_mean'):.4f} | "
            f"{mean('val_embedding_step_mean'):.4f} | "
            f"{mean('val_score_iou_corr'):.4f} | "
            f"{mean('val_delta_iou_corr'):.4f} | "
            f"{row['low_conf_high_iou_mean_prob_delta_mean']:.4f} | "
            f"{row['rescue_band_030_050_high_iou_mean_prob_delta_mean']:.4f} | "
            f"{row['risky_score_ge_030_low_iou_mean_prob_delta_mean']:.4f} | "
            f"{mean('val_energy_dual_all'):.4f} +/- {std('val_energy_dual_all'):.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--feature-key", default="final_head_l2")
    parser.add_argument("--reference-mode", default="positive_fallback", choices=["all", "positive_fallback"])
    parser.add_argument("--seeds", default="1,2,3,42,999")
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--max-step", type=float, default=0.25)
    parser.add_argument("--residual-scale-init", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--score-prior", default="label_prob", choices=["none", "label_prob"])
    parser.add_argument("--pos-weight-mode", default="none", choices=["none", "balanced"])
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=Path("output/roi_dual_energy_adapter_results.json"))
    args = parser.parse_args()

    device = torch.device(args.device)
    tensors = load_cache(args.cache, args.feature_key, device)
    num_classes = int(tensors["train_class"].max().item()) + 1
    reference = build_reference_prototypes(
        tensors["train_x"],
        tensors["train_class"],
        tensors["train_y"],
        mode=args.reference_mode,
        num_classes=num_classes,
    ).detach()
    train_prior = None
    val_prior = None
    if args.score_prior == "label_prob":
        train_prior = prior_logits_from_probs(tensors["train_label_prob"])
        val_prior = prior_logits_from_probs(tensors["val_label_prob"])

    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    results = []
    for seed in seeds:
        for variant in default_variants():
            result = train_variant(
                variant,
                seed=seed,
                train_x=tensors["train_x"],
                train_y=tensors["train_y"],
                train_class=tensors["train_class"],
                train_iou=tensors["train_iou"],
                train_prior=train_prior,
                val_x=tensors["val_x"],
                val_y=tensors["val_y"],
                val_class=tensors["val_class"],
                val_iou=tensors["val_iou"],
                val_prior=val_prior,
                reference=reference,
                epochs=args.epochs,
                lr=args.lr,
                hidden=args.hidden,
                max_step=args.max_step,
                residual_scale_init=args.residual_scale_init,
                weight_decay=args.weight_decay,
                pos_weight_mode=args.pos_weight_mode,
                log_every=args.log_every,
            )
            results.append(result)

    train_prob = tensors["train_label_prob"].detach().cpu().numpy()
    val_prob = tensors["val_label_prob"].detach().cpu().numpy()
    train_y = tensors["train_y"].detach().cpu().numpy().astype(bool)
    val_y = tensors["val_y"].detach().cpu().numpy().astype(bool)
    val_iou = tensors["val_iou"].detach().cpu().numpy()
    val_delta_zero = np.zeros_like(val_prob)
    baseline = {
        "variant": "detector_label_prob",
        "train_auc": auc_score(train_y, train_prob),
        "train_ap": average_precision(train_y, train_prob),
        "val_auc": auc_score(val_y, val_prob),
        "val_ap": average_precision(val_y, val_prob),
        "val_brier": brier_score(val_y, val_prob),
        "val_ece": ece_score(val_y, val_prob),
        "val_precision_at_pos": precision_at_pos_count(val_y, val_prob),
        "val_score_iou_corr": safe_corr(val_prob, val_iou),
        "val_thresholds": threshold_metrics(val_y, val_prob, val_iou),
        "val_topk": topk_metrics(val_y, val_prob, val_iou),
        "val_group_metrics": group_metrics(
            val_y,
            val_prob,
            val_iou,
            prior_probs=val_prob,
            prob_delta=val_delta_zero,
        ),
    }
    summary = summarize(results)
    print_summary(summary)
    print("\nDetector label_prob baseline:")
    print(json.dumps(baseline, indent=2))

    payload = {
        "config": {
            "cache": str(args.cache),
            "feature_key": args.feature_key,
            "reference_mode": args.reference_mode,
            "seeds": seeds,
            "epochs": args.epochs,
            "lr": args.lr,
            "hidden": args.hidden,
            "max_step": args.max_step,
            "residual_scale_init": args.residual_scale_init,
            "weight_decay": args.weight_decay,
            "score_prior": args.score_prior,
            "pos_weight_mode": args.pos_weight_mode,
            "log_every": args.log_every,
            "device": str(device),
        },
        "baseline": baseline,
        "summary": summary,
        "results": [asdict(result) for result in results],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved {args.output}")


if __name__ == "__main__":
    main()
