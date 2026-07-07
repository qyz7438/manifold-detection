"""Check whether ROI structure signals add information beyond detector scores.

This is an offline cache diagnostic, not a detector AP experiment.  It asks a
more basic question before adding any new loss: do compactness/basin prototype
features provide incremental AP75 information after the detector's own
label-probability score is already known?
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
except ImportError as exc:  # pragma: no cover - exercised only on missing local env deps.
    raise SystemExit(
        "This diagnostic requires scikit-learn. The maintained remote test set "
        "does not require it, but local information-gain analysis does."
    ) from exc

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@dataclass(frozen=True)
class FitResult:
    name: str
    feature_space: str
    feature_kind: str
    class_weight: str
    control: str
    feature_dim: int
    selected_c: float | None
    cv_ap_mean: float | None
    val_auc: float
    val_ap: float
    val_logloss: float
    val_brier: float
    val_ece: float
    val_precision_at_pos: float
    val_score_iou_corr: float
    group_metrics: dict[str, dict[str, float | int | None]]


@dataclass(frozen=True)
class IncrementalResult:
    feature_space: str
    class_weight: str
    comparison: str
    base_model: str
    test_model: str
    delta_auc: float
    delta_ap: float
    delta_logloss_improvement: float
    delta_brier_improvement: float
    bootstrap_ap_ci95: tuple[float, float]
    bootstrap_auc_ci95: tuple[float, float]
    global_shuffle_delta_ap_mean: float | None
    global_shuffle_delta_ap_std: float | None
    class_shuffle_delta_ap_mean: float | None
    class_shuffle_delta_ap_std: float | None
    verdict: str


def labels_zero_based(class_ids: np.ndarray) -> np.ndarray:
    class_ids = class_ids.astype(np.int64)
    if class_ids.size and class_ids.min() >= 1:
        return class_ids - 1
    return class_ids


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x.astype(np.float64), -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x))


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p.astype(np.float64), 1e-5, 1.0 - 1e-5)
    return np.log(p / (1.0 - p))


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)


def class_prototypes(
    features: np.ndarray,
    class_ids: np.ndarray,
    quality_labels: np.ndarray,
    *,
    num_classes: int,
    reference_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    z = l2_normalize(features.astype(np.float64))
    y = quality_labels.astype(bool)
    prototypes = np.zeros((num_classes, z.shape[1]), dtype=np.float64)
    counts = np.zeros(num_classes, dtype=np.int64)
    for class_idx in range(num_classes):
        mask = class_ids == class_idx
        if reference_mode == "positive_fallback":
            pos_mask = mask & y
            if pos_mask.any():
                mask = pos_mask
        elif reference_mode != "all":
            raise ValueError("reference_mode must be 'all' or 'positive_fallback'")
        counts[class_idx] = int(mask.sum())
        if mask.any():
            proto = z[mask].mean(axis=0)
            prototypes[class_idx] = proto / (np.linalg.norm(proto) + 1e-12)
    return prototypes, counts


def structure_features(
    train_features: np.ndarray,
    train_class_ids: np.ndarray,
    train_quality: np.ndarray,
    split_features: np.ndarray,
    split_class_ids: np.ndarray,
    *,
    reference_mode: str,
    temperature: float,
) -> tuple[np.ndarray, list[str]]:
    num_classes = int(max(train_class_ids.max(), split_class_ids.max())) + 1
    prototypes, counts = class_prototypes(
        train_features,
        train_class_ids,
        train_quality,
        num_classes=num_classes,
        reference_mode=reference_mode,
    )
    z = l2_normalize(split_features.astype(np.float64))
    sim = z @ prototypes.T
    row = np.arange(z.shape[0])
    true_sim = sim[row, split_class_ids]
    other_sim = sim.copy()
    other_sim[row, split_class_ids] = -np.inf
    best_other_sim = np.max(other_sim, axis=1)
    margin = true_sim - best_other_sim

    target_proto = prototypes[split_class_ids]
    dist_to_proto = np.sum((z - target_proto) ** 2, axis=1)
    present = np.linalg.norm(prototypes, axis=1) > 1e-8
    if int(present.sum()) >= 2:
        p = prototypes[present]
        sq = np.sum((p[:, None, :] - p[None, :, :]) ** 2, axis=2)
        spacing = float(np.mean(sq[~np.eye(p.shape[0], dtype=bool)]))
    else:
        spacing = 1.0
    compact = dist_to_proto / max(spacing, 1e-8)
    basin_energy = np.logaddexp(0.0, (-margin) / float(temperature))

    true_rank = 1 + np.sum(sim > true_sim[:, None], axis=1)
    class_support = counts[split_class_ids].astype(np.float64)
    class_support_log = np.log1p(class_support)
    class_support_inv = 1.0 / np.maximum(class_support, 1.0)

    features = np.stack(
        [
            true_sim,
            best_other_sim,
            margin,
            dist_to_proto,
            compact,
            basin_energy,
            true_rank.astype(np.float64),
            class_support_log,
            class_support_inv,
        ],
        axis=1,
    )
    names = [
        "proto_true_cos",
        "proto_best_other_cos",
        "proto_margin",
        "proto_sqdist",
        "compact_norm",
        "basin_energy",
        "proto_true_rank",
        "class_support_log1p",
        "class_support_inv",
    ]
    return features, names


def safe_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    if np.unique(y_true).size < 2:
        return math.nan
    return float(roc_auc_score(y_true, scores))


def safe_ap(y_true: np.ndarray, scores: np.ndarray) -> float:
    if int(np.sum(y_true)) == 0:
        return math.nan
    return float(average_precision_score(y_true, scores))


def ece_score(y_true: np.ndarray, probs: np.ndarray, bins: int = 10) -> float:
    y = y_true.astype(np.float64)
    p = np.clip(probs.astype(np.float64), 0.0, 1.0)
    ece = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi >= 1.0:
            mask = (p >= lo) & (p <= hi)
        else:
            mask = (p >= lo) & (p < hi)
        if mask.any():
            ece += float(mask.mean()) * abs(float(p[mask].mean()) - float(y[mask].mean()))
    return float(ece)


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    if a.size < 2 or float(np.std(a)) < 1e-12 or float(np.std(b)) < 1e-12:
        return math.nan
    return float(np.corrcoef(a, b)[0, 1])


def precision_at_pos_count(y_true: np.ndarray, scores: np.ndarray) -> float:
    y = y_true.astype(bool)
    k = int(y.sum())
    if k <= 0:
        return math.nan
    order = np.argsort(-scores.astype(np.float64))[:k]
    return float(y[order].mean())


def safe_rate(numer: int, denom: int) -> float | None:
    if denom <= 0:
        return None
    return float(numer / denom)


def group_masks(y_true: np.ndarray, iou: np.ndarray, prior: np.ndarray) -> dict[str, np.ndarray]:
    y = y_true.astype(bool)
    return {
        "all": np.ones_like(y, dtype=bool),
        "ap75_positive": y,
        "ap75_negative": ~y,
        "low_conf_high_iou": (prior < 0.50) & (iou >= 0.75),
        "very_low_conf_high_iou": (prior < 0.30) & (iou >= 0.75),
        "rescue_band_030_050_high_iou": (prior >= 0.30) & (prior < 0.50) & (iou >= 0.75),
        "risky_score_ge_030_low_iou": (prior >= 0.30) & (iou < 0.50),
        "risky_score_ge_020_low_iou": (prior >= 0.20) & (iou < 0.50),
    }


def group_metrics(
    y_true: np.ndarray,
    probs: np.ndarray,
    iou: np.ndarray,
    prior: np.ndarray,
) -> dict[str, dict[str, float | int | None]]:
    rows: dict[str, dict[str, float | int | None]] = {}
    delta = probs - prior
    for name, mask in group_masks(y_true, iou, prior).items():
        count = int(mask.sum())
        if count == 0:
            rows[name] = {"count": 0}
            continue
        y_group = y_true[mask].astype(bool)
        pred_050 = probs[mask] >= 0.50
        tp = int((pred_050 & y_group).sum())
        pred_count = int(pred_050.sum())
        rows[name] = {
            "count": count,
            "target_positive_count": int(y_group.sum()),
            "target_positive_rate": float(y_group.mean()),
            "mean_iou": float(iou[mask].mean()),
            "mean_prior": float(prior[mask].mean()),
            "mean_score": float(probs[mask].mean()),
            "mean_delta_vs_prior": float(delta[mask].mean()),
            "mean_abs_delta_vs_prior": float(np.abs(delta[mask]).mean()),
            "pred_count_050": pred_count,
            "tp_050": tp,
            "fp_050": int((pred_050 & ~y_group).sum()),
            "precision_050": safe_rate(tp, pred_count),
            "recall_050": safe_rate(tp, int(y_group.sum())),
        }
    return rows


def rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(values.shape[0], dtype=np.float64)
    return ranks


def target_labels(
    target: str,
    labels: np.ndarray,
    iou: np.ndarray,
    prob: np.ndarray,
) -> np.ndarray:
    if target == "ap75":
        return labels.astype(np.int64)
    if target == "rescue075":
        return ((prob < 0.50) & (iou >= 0.75)).astype(np.int64)
    if target == "very_low_conf_high_iou":
        return ((prob < 0.30) & (iou >= 0.75)).astype(np.int64)
    if target == "risky030_low_iou":
        return ((prob >= 0.30) & (iou < 0.50)).astype(np.int64)
    if target == "risky020_low_iou":
        return ((prob >= 0.20) & (iou < 0.50)).astype(np.int64)
    if target == "underestimated_rank":
        return (rank(iou) > rank(prob)).astype(np.int64)
    raise ValueError(f"Unknown target: {target}")


def metric_bundle(y_true: np.ndarray, probs: np.ndarray, iou: np.ndarray, prior: np.ndarray) -> dict[str, object]:
    clipped = np.clip(probs.astype(np.float64), 1e-6, 1.0 - 1e-6)
    return {
        "val_auc": safe_auc(y_true, clipped),
        "val_ap": safe_ap(y_true, clipped),
        "val_logloss": float(log_loss(y_true, clipped)),
        "val_brier": float(brier_score_loss(y_true, clipped)),
        "val_ece": ece_score(y_true, clipped),
        "val_precision_at_pos": precision_at_pos_count(y_true, clipped),
        "val_score_iou_corr": safe_corr(clipped, iou),
        "group_metrics": group_metrics(y_true, clipped, iou, prior),
    }


def make_estimator(c: float, class_weight: str | None) -> object:
    weight = None if class_weight in {None, "none"} else class_weight
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=float(c),
            class_weight=weight,
            max_iter=5000,
            solver="lbfgs",
        ),
    )


def cv_select_c(
    x: np.ndarray,
    y: np.ndarray,
    *,
    c_grid: Iterable[float],
    class_weight: str,
    seed: int,
) -> tuple[float, float]:
    pos = int(y.sum())
    neg = int(y.shape[0] - pos)
    folds = min(5, pos, neg)
    c_values = [float(c) for c in c_grid]
    if folds < 2:
        return c_values[0], math.nan

    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    best_c = c_values[0]
    best_score = -math.inf
    for c in c_values:
        scores = []
        for train_idx, test_idx in splitter.split(x, y):
            estimator = make_estimator(c, class_weight)
            estimator.fit(x[train_idx], y[train_idx])
            probs = estimator.predict_proba(x[test_idx])[:, 1]
            scores.append(safe_ap(y[test_idx], probs))
        score = float(np.nanmean(scores))
        if score > best_score:
            best_score = score
            best_c = c
    return best_c, best_score


def fit_predict(
    *,
    name: str,
    feature_space: str,
    feature_kind: str,
    control: str,
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    val_iou: np.ndarray,
    val_prior: np.ndarray,
    class_weight: str,
    c_grid: Iterable[float],
    seed: int,
    fixed_c: float | None = None,
) -> tuple[FitResult, np.ndarray]:
    if fixed_c is None:
        selected_c, cv_ap = cv_select_c(train_x, train_y, c_grid=c_grid, class_weight=class_weight, seed=seed)
    else:
        selected_c, cv_ap = float(fixed_c), None
    estimator = make_estimator(selected_c, class_weight)
    estimator.fit(train_x, train_y)
    probs = estimator.predict_proba(val_x)[:, 1]
    metrics = metric_bundle(val_y, probs, val_iou, val_prior)
    result = FitResult(
        name=name,
        feature_space=feature_space,
        feature_kind=feature_kind,
        class_weight=class_weight,
        control=control,
        feature_dim=int(train_x.shape[1]),
        selected_c=selected_c,
        cv_ap_mean=cv_ap,
        val_auc=float(metrics["val_auc"]),
        val_ap=float(metrics["val_ap"]),
        val_logloss=float(metrics["val_logloss"]),
        val_brier=float(metrics["val_brier"]),
        val_ece=float(metrics["val_ece"]),
        val_precision_at_pos=float(metrics["val_precision_at_pos"]),
        val_score_iou_corr=float(metrics["val_score_iou_corr"]),
        group_metrics=metrics["group_metrics"],  # type: ignore[arg-type]
    )
    return result, probs


def raw_prior_result(val_y: np.ndarray, val_prob: np.ndarray, val_iou: np.ndarray) -> FitResult:
    metrics = metric_bundle(val_y, val_prob, val_iou, val_prob)
    return FitResult(
        name="detector_label_prob_raw",
        feature_space="detector",
        feature_kind="prior",
        class_weight="raw",
        control="none",
        feature_dim=1,
        selected_c=None,
        cv_ap_mean=None,
        val_auc=float(metrics["val_auc"]),
        val_ap=float(metrics["val_ap"]),
        val_logloss=float(metrics["val_logloss"]),
        val_brier=float(metrics["val_brier"]),
        val_ece=float(metrics["val_ece"]),
        val_precision_at_pos=float(metrics["val_precision_at_pos"]),
        val_score_iou_corr=float(metrics["val_score_iou_corr"]),
        group_metrics=metrics["group_metrics"],  # type: ignore[arg-type]
    )


def shuffle_rows(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return x[rng.permutation(x.shape[0])]


def shuffle_rows_within_class(x: np.ndarray, class_ids: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = x.copy()
    for class_idx in np.unique(class_ids):
        mask = np.flatnonzero(class_ids == class_idx)
        if mask.size > 1:
            out[mask] = out[rng.permutation(mask)]
    return out


def bootstrap_delta_ci(
    y: np.ndarray,
    base_probs: np.ndarray,
    test_probs: np.ndarray,
    *,
    metric: str,
    seed: int,
    reps: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    deltas = []
    n = y.shape[0]
    for _ in range(int(reps)):
        idx = rng.integers(0, n, size=n)
        if np.unique(y[idx]).size < 2:
            continue
        if metric == "ap":
            delta = safe_ap(y[idx], test_probs[idx]) - safe_ap(y[idx], base_probs[idx])
        elif metric == "auc":
            delta = safe_auc(y[idx], test_probs[idx]) - safe_auc(y[idx], base_probs[idx])
        else:
            raise ValueError(f"Unsupported bootstrap metric: {metric}")
        if math.isfinite(delta):
            deltas.append(delta)
    if not deltas:
        return math.nan, math.nan
    arr = np.array(deltas, dtype=np.float64)
    return float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))


def summarize_shuffle_deltas(rows: list[FitResult], base: FitResult, *, metric: str) -> tuple[float | None, float | None]:
    if not rows:
        return None, None
    if metric == "ap":
        deltas = np.array([row.val_ap - base.val_ap for row in rows], dtype=np.float64)
    elif metric == "auc":
        deltas = np.array([row.val_auc - base.val_auc for row in rows], dtype=np.float64)
    else:
        raise ValueError(f"Unsupported metric: {metric}")
    return float(deltas.mean()), float(deltas.std())


def verdict_for_delta(
    real_delta_ap: float,
    global_mean: float | None,
    global_std: float | None,
    class_mean: float | None,
    class_std: float | None,
) -> str:
    if real_delta_ap <= 0.0:
        return "no_gain"
    global_pass = global_mean is None or real_delta_ap > global_mean + 2.0 * max(global_std or 0.0, 1e-12)
    class_pass = class_mean is None or real_delta_ap > class_mean + 2.0 * max(class_std or 0.0, 1e-12)
    if global_pass and class_pass:
        return "incremental_signal"
    if global_pass:
        return "class_prior_or_marginal_only"
    return "not_above_shuffle"


def build_incremental_result(
    *,
    feature_space: str,
    class_weight: str,
    comparison: str,
    base: FitResult,
    test: FitResult,
    base_probs: np.ndarray,
    test_probs: np.ndarray,
    global_shuffle_rows: list[FitResult],
    class_shuffle_rows: list[FitResult],
    val_y: np.ndarray,
    seed: int,
    bootstrap_reps: int,
) -> IncrementalResult:
    global_mean, global_std = summarize_shuffle_deltas(global_shuffle_rows, base, metric="ap")
    class_mean, class_std = summarize_shuffle_deltas(class_shuffle_rows, base, metric="ap")
    delta_ap = float(test.val_ap - base.val_ap)
    return IncrementalResult(
        feature_space=feature_space,
        class_weight=class_weight,
        comparison=comparison,
        base_model=base.name,
        test_model=test.name,
        delta_auc=float(test.val_auc - base.val_auc),
        delta_ap=delta_ap,
        delta_logloss_improvement=float(base.val_logloss - test.val_logloss),
        delta_brier_improvement=float(base.val_brier - test.val_brier),
        bootstrap_ap_ci95=bootstrap_delta_ci(
            val_y,
            base_probs,
            test_probs,
            metric="ap",
            seed=seed,
            reps=bootstrap_reps,
        ),
        bootstrap_auc_ci95=bootstrap_delta_ci(
            val_y,
            base_probs,
            test_probs,
            metric="auc",
            seed=seed + 17,
            reps=bootstrap_reps,
        ),
        global_shuffle_delta_ap_mean=global_mean,
        global_shuffle_delta_ap_std=global_std,
        class_shuffle_delta_ap_mean=class_mean,
        class_shuffle_delta_ap_std=class_std,
        verdict=verdict_for_delta(delta_ap, global_mean, global_std, class_mean, class_std),
    )


def json_ready(value: object) -> object:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, tuple):
        return [json_ready(v) for v in value]
    if isinstance(value, list):
        return [json_ready(v) for v in value]
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    return value


def write_markdown(payload: dict, path: Path) -> None:
    lines = [
        "# ROI Structure Information-Gain Check",
        "",
        "This offline diagnostic tests whether ROI structure features add target information beyond detector label probability.",
        "",
        f"Target: `{payload['config'].get('target', 'ap75')}`.",
        "",
        "## Incremental Results",
        "",
        "| feature | class weight | comparison | dAUC | dAP | AP CI95 | global shuffle dAP | class shuffle dAP | verdict |",
        "|---|---|---|---:|---:|---|---:|---:|---|",
    ]
    for row in payload["incremental"]:
        gap = row["global_shuffle_delta_ap_mean"]
        csp = row["class_shuffle_delta_ap_mean"]
        ci = row["bootstrap_ap_ci95"]
        lines.append(
            "| {feature_space} | {class_weight} | {comparison} | {delta_auc:.4f} | {delta_ap:.4f} | "
            "[{ci0:.4f}, {ci1:.4f}] | {global_gap} | {class_gap} | {verdict} |".format(
                feature_space=row["feature_space"],
                class_weight=row["class_weight"],
                comparison=row["comparison"],
                delta_auc=row["delta_auc"],
                delta_ap=row["delta_ap"],
                ci0=ci[0] if ci[0] is not None else math.nan,
                ci1=ci[1] if ci[1] is not None else math.nan,
                global_gap="-" if gap is None else f"{gap:.4f}",
                class_gap="-" if csp is None else f"{csp:.4f}",
                verdict=row["verdict"],
            )
        )
    lines.extend(
        [
            "",
            "## Model Results",
            "",
            "| model | feature | kind | cw | control | C | AUC | AP | logloss | brier | ECE | p@pos |",
            "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["results"]:
        if row["control"] != "none" and not row["name"].endswith("_rep0"):
            continue
        row_for_format = dict(row)
        row_for_format["selected_c"] = "-" if row["selected_c"] is None else f"{row['selected_c']:.4g}"
        lines.append(
            "| {name} | {feature_space} | {feature_kind} | {class_weight} | {control} | {selected_c} | "
            "{val_auc:.4f} | {val_ap:.4f} | {val_logloss:.4f} | {val_brier:.4f} | "
            "{val_ece:.4f} | {val_precision_at_pos:.4f} |".format(**row_for_format)
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_analysis(args: argparse.Namespace) -> dict:
    rng = np.random.default_rng(int(args.seed))
    data = np.load(args.cache, allow_pickle=False)
    train_quality = data["train_labels"].astype(np.int64)
    val_quality = data["val_labels"].astype(np.int64)
    train_class = labels_zero_based(data["train_class_ids"])
    val_class = labels_zero_based(data["val_class_ids"])
    train_prob = data["train_label_probs"].astype(np.float64)
    val_prob = data["val_label_probs"].astype(np.float64)
    train_iou = data["train_best_iou"].astype(np.float64)
    val_iou = data["val_best_iou"].astype(np.float64)
    train_y = target_labels(args.target, train_quality, train_iou, train_prob)
    val_y = target_labels(args.target, val_quality, val_iou, val_prob)
    train_prior = logit(train_prob)[:, None]
    val_prior = logit(val_prob)[:, None]
    c_grid = [float(item) for item in parse_csv(args.c_grid)]
    class_weight_modes = parse_csv(args.class_weight_modes)

    results: list[FitResult] = [raw_prior_result(val_y, val_prob, val_iou)]
    predictions: dict[str, np.ndarray] = {"detector_label_prob_raw": val_prob}
    incremental: list[IncrementalResult] = []

    prior_results: dict[str, FitResult] = {}
    for class_weight in class_weight_modes:
        result, probs = fit_predict(
            name=f"prior_logit_calibrated_cw_{class_weight}",
            feature_space="detector",
            feature_kind="prior_logit",
            control="none",
            train_x=train_prior,
            train_y=train_y,
            val_x=val_prior,
            val_y=val_y,
            val_iou=val_iou,
            val_prior=val_prob,
            class_weight=class_weight,
            c_grid=c_grid,
            seed=int(args.seed),
        )
        results.append(result)
        predictions[result.name] = probs
        prior_results[class_weight] = result

    for feature_key in parse_csv(args.feature_keys):
        train_key = f"train_{feature_key}"
        val_key = f"val_{feature_key}"
        if train_key not in data.files or val_key not in data.files:
            print(f"[skip] missing feature key: {feature_key}")
            continue
        train_features = data[train_key].astype(np.float64)
        val_features = data[val_key].astype(np.float64)
        train_structure, structure_names = structure_features(
            train_features,
            train_class,
            train_quality,
            train_features,
            train_class,
            reference_mode=args.reference_mode,
            temperature=float(args.basin_temperature),
        )
        val_structure, _ = structure_features(
            train_features,
            train_class,
            train_quality,
            val_features,
            val_class,
            reference_mode=args.reference_mode,
            temperature=float(args.basin_temperature),
        )
        for class_weight in class_weight_modes:
            prior_result = prior_results[class_weight]
            prior_probs = predictions[prior_result.name]

            structure_result, structure_probs = fit_predict(
                name=f"{feature_key}_structure_only_cw_{class_weight}",
                feature_space=feature_key,
                feature_kind="structure",
                control="none",
                train_x=train_structure,
                train_y=train_y,
                val_x=val_structure,
                val_y=val_y,
                val_iou=val_iou,
                val_prior=val_prob,
                class_weight=class_weight,
                c_grid=c_grid,
                seed=int(args.seed),
            )
            results.append(structure_result)
            predictions[structure_result.name] = structure_probs

            combined_train = np.concatenate([train_prior, train_structure], axis=1)
            combined_val = np.concatenate([val_prior, val_structure], axis=1)
            combined_result, combined_probs = fit_predict(
                name=f"{feature_key}_prior_plus_structure_cw_{class_weight}",
                feature_space=feature_key,
                feature_kind="prior_plus_structure",
                control="none",
                train_x=combined_train,
                train_y=train_y,
                val_x=combined_val,
                val_y=val_y,
                val_iou=val_iou,
                val_prior=val_prob,
                class_weight=class_weight,
                c_grid=c_grid,
                seed=int(args.seed),
            )
            results.append(combined_result)
            predictions[combined_result.name] = combined_probs

            global_shuffle_rows: list[FitResult] = []
            class_shuffle_rows: list[FitResult] = []
            for rep in range(int(args.shuffle_reps)):
                shuffled_train = shuffle_rows(train_structure, rng)
                shuffled_val = shuffle_rows(val_structure, rng)
                shuffled_result, shuffled_probs = fit_predict(
                    name=f"{feature_key}_prior_plus_global_shuffle_cw_{class_weight}_rep{rep}",
                    feature_space=feature_key,
                    feature_kind="prior_plus_structure",
                    control="global_shuffle",
                    train_x=np.concatenate([train_prior, shuffled_train], axis=1),
                    train_y=train_y,
                    val_x=np.concatenate([val_prior, shuffled_val], axis=1),
                    val_y=val_y,
                    val_iou=val_iou,
                    val_prior=val_prob,
                    class_weight=class_weight,
                    c_grid=c_grid,
                    seed=int(args.seed) + rep + 101,
                    fixed_c=combined_result.selected_c,
                )
                results.append(shuffled_result)
                predictions[shuffled_result.name] = shuffled_probs
                global_shuffle_rows.append(shuffled_result)

                class_shuffled_train = shuffle_rows_within_class(train_structure, train_class, rng)
                class_shuffled_val = shuffle_rows_within_class(val_structure, val_class, rng)
                class_shuffled_result, class_shuffled_probs = fit_predict(
                    name=f"{feature_key}_prior_plus_class_shuffle_cw_{class_weight}_rep{rep}",
                    feature_space=feature_key,
                    feature_kind="prior_plus_structure",
                    control="class_shuffle",
                    train_x=np.concatenate([train_prior, class_shuffled_train], axis=1),
                    train_y=train_y,
                    val_x=np.concatenate([val_prior, class_shuffled_val], axis=1),
                    val_y=val_y,
                    val_iou=val_iou,
                    val_prior=val_prob,
                    class_weight=class_weight,
                    c_grid=c_grid,
                    seed=int(args.seed) + rep + 701,
                    fixed_c=combined_result.selected_c,
                )
                results.append(class_shuffled_result)
                predictions[class_shuffled_result.name] = class_shuffled_probs
                class_shuffle_rows.append(class_shuffled_result)

            incremental.append(
                build_incremental_result(
                    feature_space=feature_key,
                    class_weight=class_weight,
                    comparison="prior_plus_structure_vs_prior",
                    base=prior_result,
                    test=combined_result,
                    base_probs=prior_probs,
                    test_probs=combined_probs,
                    global_shuffle_rows=global_shuffle_rows,
                    class_shuffle_rows=class_shuffle_rows,
                    val_y=val_y,
                    seed=int(args.seed) + 3001,
                    bootstrap_reps=int(args.bootstrap_reps),
                )
            )

        if not args.skip_raw_feature_controls:
            for class_weight in class_weight_modes:
                prior_result = prior_results[class_weight]
                raw_combined_train = np.concatenate([train_prior, train_features], axis=1)
                raw_combined_val = np.concatenate([val_prior, val_features], axis=1)
                raw_result, raw_probs = fit_predict(
                    name=f"{feature_key}_prior_plus_raw_feature_cw_{class_weight}",
                    feature_space=feature_key,
                    feature_kind="prior_plus_raw_feature",
                    control="none",
                    train_x=raw_combined_train,
                    train_y=train_y,
                    val_x=raw_combined_val,
                    val_y=val_y,
                    val_iou=val_iou,
                    val_prior=val_prob,
                    class_weight=class_weight,
                    c_grid=c_grid,
                    seed=int(args.seed),
                )
                results.append(raw_result)
                predictions[raw_result.name] = raw_probs
                incremental.append(
                    build_incremental_result(
                        feature_space=feature_key,
                        class_weight=class_weight,
                        comparison="prior_plus_raw_feature_vs_prior",
                        base=prior_result,
                        test=raw_result,
                        base_probs=predictions[prior_result.name],
                        test_probs=raw_probs,
                        global_shuffle_rows=[],
                        class_shuffle_rows=[],
                        val_y=val_y,
                        seed=int(args.seed) + 9001,
                        bootstrap_reps=int(args.bootstrap_reps),
                    )
                )

    payload = {
        "config": {
            "cache": str(args.cache),
            "target": args.target,
            "feature_keys": parse_csv(args.feature_keys),
            "reference_mode": args.reference_mode,
            "basin_temperature": args.basin_temperature,
            "class_weight_modes": class_weight_modes,
            "c_grid": c_grid,
            "shuffle_reps": args.shuffle_reps,
            "bootstrap_reps": args.bootstrap_reps,
            "skip_raw_feature_controls": args.skip_raw_feature_controls,
            "seed": args.seed,
            "structure_feature_names": structure_names if "structure_names" in locals() else [],
            "train_count": int(train_y.shape[0]),
            "train_positive_count": int(train_y.sum()),
            "val_count": int(val_y.shape[0]),
            "val_positive_count": int(val_y.sum()),
            "prototype_quality_label": "ap75",
        },
        "baseline": asdict(results[0]),
        "results": [asdict(row) for row in results],
        "incremental": [asdict(row) for row in incremental],
    }
    return json_ready(payload)  # type: ignore[return-value]


def print_incremental(payload: dict) -> None:
    print("feature | cw | comparison | dAUC | dAP | global_shuffle_dAP | class_shuffle_dAP | verdict")
    print("---|---|---|---:|---:|---:|---:|---")
    for row in payload["incremental"]:
        global_gap = row["global_shuffle_delta_ap_mean"]
        class_gap = row["class_shuffle_delta_ap_mean"]
        print(
            f"{row['feature_space']} | {row['class_weight']} | {row['comparison']} | "
            f"{row['delta_auc']:.4f} | {row['delta_ap']:.4f} | "
            f"{'-' if global_gap is None else f'{global_gap:.4f}'} | "
            f"{'-' if class_gap is None else f'{class_gap:.4f}'} | "
            f"{row['verdict']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument(
        "--target",
        choices=[
            "ap75",
            "rescue075",
            "very_low_conf_high_iou",
            "risky030_low_iou",
            "risky020_low_iou",
            "underestimated_rank",
        ],
        default="ap75",
    )
    parser.add_argument("--feature-keys", default="features_l2,final_head_l2")
    parser.add_argument("--reference-mode", choices=["all", "positive_fallback"], default="positive_fallback")
    parser.add_argument("--basin-temperature", type=float, default=0.05)
    parser.add_argument("--class-weight-modes", default="none,balanced")
    parser.add_argument("--c-grid", default="0.003,0.01,0.03,0.1,0.3,1.0")
    parser.add_argument("--shuffle-reps", type=int, default=30)
    parser.add_argument("--bootstrap-reps", type=int, default=1000)
    parser.add_argument("--skip-raw-feature-controls", action="store_true")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output", type=Path, default=Path("output/roi_structure_information_gain.json"))
    parser.add_argument("--markdown-output", type=Path, default=Path("output/roi_structure_information_gain.md"))
    args = parser.parse_args()

    payload = run_analysis(args)
    print_incremental(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_markdown(payload, args.markdown_output)
    print(f"\nSaved {args.output}")
    print(f"Saved {args.markdown_output}")


if __name__ == "__main__":
    main()
