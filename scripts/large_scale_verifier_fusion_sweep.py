from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spectral_detection_posttrain.analysis.dimensionality import binary_ranking_metrics  # noqa: E402
from spectral_detection_posttrain.analysis.raw_ifft_verifier import (  # noqa: E402
    LEGACY_IFFT_FEATURE_NAMES,
    calibrate_precision_threshold,
    fit_train_effect_scorer,
    threshold_metrics,
)
from spectral_detection_posttrain.utils.io import save_json  # noqa: E402


class Scorer(Protocol):
    def score(self, features: np.ndarray, class_ids: np.ndarray | None = None) -> np.ndarray: ...


@dataclass
class CenterScorer:
    positive_center: np.ndarray
    negative_center: np.ndarray

    def score(self, features: np.ndarray, class_ids: np.ndarray | None = None) -> np.ndarray:
        pos_dist = np.linalg.norm(features - self.positive_center, axis=1)
        neg_dist = np.linalg.norm(features - self.negative_center, axis=1)
        return neg_dist - pos_dist


@dataclass
class KNNScorer:
    positive: np.ndarray
    negative: np.ndarray
    k: int

    def score(self, features: np.ndarray, class_ids: np.ndarray | None = None) -> np.ndarray:
        def mean_distance(reference: np.ndarray) -> np.ndarray:
            k_eff = min(max(1, int(self.k)), reference.shape[0])
            nn = NearestNeighbors(n_neighbors=k_eff, metric="euclidean")
            nn.fit(reference)
            distances, _ = nn.kneighbors(features, return_distance=True)
            return distances.mean(axis=1)

        return mean_distance(self.negative) - mean_distance(self.positive)


@dataclass
class LogisticScorer:
    model: LogisticRegression

    def score(self, features: np.ndarray, class_ids: np.ndarray | None = None) -> np.ndarray:
        return self.model.decision_function(features)


@dataclass
class ClasswiseScorer:
    global_scorer: Scorer
    scorers: dict[int, Scorer]

    def score(self, features: np.ndarray, class_ids: np.ndarray | None = None) -> np.ndarray:
        output = self.global_scorer.score(features)
        if class_ids is None:
            return output
        class_ids = np.asarray(class_ids, dtype=np.int64)
        for class_id, scorer in self.scorers.items():
            mask = class_ids == int(class_id)
            if mask.any():
                output[mask] = scorer.score(features[mask])
        return output


@dataclass
class Transform:
    name: str
    scaler: StandardScaler | None = None
    pca: PCA | None = None

    def apply(self, features: np.ndarray) -> np.ndarray:
        output = np.asarray(features, dtype=np.float64)
        if self.scaler is not None:
            output = self.scaler.transform(output)
        if self.pca is not None:
            output = self.pca.transform(output)
        return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Large-scale box-feature/raw-iFFT verifier fusion sweep.")
    parser.add_argument(
        "--full-cache",
        default="runs/round2199_box_feature_classwise_iou_bucket_manifold/iou_bucket_box_features.npz",
    )
    parser.add_argument(
        "--raw-cache",
        default="runs/round2150_raw_ifft_legacy_full_ap75/candidate_raw_ifft_features.npz",
    )
    parser.add_argument("--run-name", default="round2200_large_scale_verifier_fusion_sweep")
    parser.add_argument("--components", type=int, nargs="+", default=[32, 48, 56, 58, 64, 96, 128])
    parser.add_argument("--knn-k", type=int, default=5)
    parser.add_argument("--low-conf-max", type=float, default=0.5)
    parser.add_argument("--high-iou-min", type=float, default=0.75)
    parser.add_argument("--low-iou-max", type=float, default=0.3)
    parser.add_argument("--classwise-min-pos", type=int, default=5)
    parser.add_argument("--classwise-min-neg", type=int, default=5)
    parser.add_argument(
        "--raw-features",
        nargs="+",
        default=["fft_edge_truncation@64", "phase_edge@64", "phase_abs_high@11"],
    )
    return parser.parse_args()


def high_low_mask(ious: np.ndarray, classes: np.ndarray, *, high: float, low: float, matched_only: bool) -> np.ndarray:
    valid = (ious >= float(high)) | (ious <= float(low))
    if matched_only:
        valid &= classes > 0
    return valid


def high_labels(ious: np.ndarray, *, high: float) -> np.ndarray:
    return np.asarray(ious >= float(high), dtype=bool)


def zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1, 1)
    return StandardScaler().fit_transform(values).reshape(-1)


def metrics_fixed_thresholds(train_scores: np.ndarray, train_labels: np.ndarray, val_scores: np.ndarray, val_labels: np.ndarray) -> dict[str, object]:
    train_scores = np.asarray(train_scores, dtype=np.float64)
    val_scores = np.asarray(val_scores, dtype=np.float64)
    train_labels = np.asarray(train_labels, dtype=bool)
    val_labels = np.asarray(val_labels, dtype=bool)
    output: dict[str, object] = {
        "ranking": binary_ranking_metrics(val_scores, val_labels),
        "train_ranking": binary_ranking_metrics(train_scores, train_labels),
        "fixed_thresholds": {},
    }
    for target in (0.7, 0.8, 0.9):
        calibration = calibrate_precision_threshold(train_scores, train_labels, target_precision=target)
        selected = val_scores >= calibration.threshold
        fixed = threshold_metrics(selected, val_labels)
        fixed["threshold"] = float(calibration.threshold)
        fixed["train_selected_prefix"] = int(calibration.selected_prefix)
        fixed["train_tp_prefix"] = int(calibration.tp_prefix)
        fixed["train_fp_prefix"] = int(calibration.fp_prefix)
        fixed["train_precision_prefix"] = float(calibration.precision_prefix)
        fixed["train_recall_prefix"] = float(calibration.recall_prefix)
        fixed["reason"] = str(calibration.reason)
        output["fixed_thresholds"][f"p{target:g}"] = fixed  # type: ignore[index]
    return output


def leaderboard_row(name: str, result: dict[str, object], *, group: str, extra: dict[str, object] | None = None) -> dict[str, object]:
    fixed = result["fixed_thresholds"]["p0.7"]  # type: ignore[index]
    ranking = result["ranking"]  # type: ignore[index]
    row = {
        "name": name,
        "group": group,
        "auc": float(ranking["auc"]),  # type: ignore[index]
        "average_precision": float(ranking["average_precision"]),  # type: ignore[index]
        "rank_recall_at_precision_0.7": float(ranking["recall_at_precision_0.7"]),  # type: ignore[index]
        "fixed_p0.7_precision": float(fixed["precision"]),  # type: ignore[index]
        "fixed_p0.7_recall": float(fixed["recall"]),  # type: ignore[index]
        "fixed_p0.7_selected": int(fixed["selected"]),  # type: ignore[index]
        "fixed_p0.7_tp": int(fixed["tp"]),  # type: ignore[index]
        "fixed_p0.7_fp": int(fixed["fp"]),  # type: ignore[index]
    }
    for target in ("p0.8", "p0.9"):
        item = result["fixed_thresholds"][target]  # type: ignore[index]
        row[f"fixed_{target}_precision"] = float(item["precision"])
        row[f"fixed_{target}_recall"] = float(item["recall"])
        row[f"fixed_{target}_selected"] = int(item["selected"])
        row[f"fixed_{target}_tp"] = int(item["tp"])
        row[f"fixed_{target}_fp"] = int(item["fp"])
    if extra:
        row.update(extra)
    return row


def build_transform(name: str, train_features: np.ndarray, components: int | None) -> Transform:
    if name == "l2":
        return Transform(name="l2")
    scaler = StandardScaler().fit(train_features)
    if components is None:
        return Transform(name="l2_z", scaler=scaler)
    train_z = scaler.transform(train_features)
    n_components = min(int(components), train_z.shape[0] - 1, train_z.shape[1])
    pca = PCA(n_components=n_components, whiten=True, random_state=42).fit(train_z)
    return Transform(name=f"l2_z_pca{n_components}", scaler=scaler, pca=pca)


def fit_center(features: np.ndarray, labels: np.ndarray) -> CenterScorer:
    positive = features[labels]
    negative = features[~labels]
    return CenterScorer(positive.mean(axis=0, keepdims=True), negative.mean(axis=0, keepdims=True))


def fit_knn(features: np.ndarray, labels: np.ndarray, *, k: int) -> KNNScorer:
    return KNNScorer(features[labels], features[~labels], k=int(k))


def fit_logistic(features: np.ndarray, labels: np.ndarray) -> LogisticScorer:
    model = LogisticRegression(C=0.1, class_weight="balanced", solver="liblinear", max_iter=1000, random_state=42)
    model.fit(features, labels.astype(np.int32))
    return LogisticScorer(model)


def fit_classwise(
    fit_fn,
    global_scorer: Scorer,
    features: np.ndarray,
    labels: np.ndarray,
    classes: np.ndarray,
    *,
    min_pos: int,
    min_neg: int,
) -> tuple[ClasswiseScorer, dict[str, object]]:
    scorers: dict[int, Scorer] = {}
    diagnostics: dict[str, object] = {"used_classes": [], "fallback_classes": []}
    for class_id in sorted(set(np.asarray(classes, dtype=np.int64).tolist())):
        mask = classes == int(class_id)
        pos_count = int((mask & labels).sum())
        neg_count = int((mask & (~labels)).sum())
        if pos_count < int(min_pos) or neg_count < int(min_neg):
            diagnostics["fallback_classes"].append(  # type: ignore[union-attr]
                {"class_id": int(class_id), "positive_count": pos_count, "negative_count": neg_count}
            )
            continue
        scorers[int(class_id)] = fit_fn(features[mask], labels[mask])
        diagnostics["used_classes"].append(  # type: ignore[union-attr]
            {"class_id": int(class_id), "positive_count": pos_count, "negative_count": neg_count}
        )
    return ClasswiseScorer(global_scorer=global_scorer, scorers=scorers), diagnostics


def load_raw_matrix(data: np.lib.npyio.NpzFile, split: str, specs: list[str]) -> np.ndarray:
    columns = []
    for spec in specs:
        feature_name, crop_text = spec.split("@", maxsplit=1)
        crop = int(crop_text)
        index = LEGACY_IFFT_FEATURE_NAMES.index(feature_name)
        columns.append(np.asarray(data[f"{split}_legacy_ifft_{crop}"][:, index], dtype=np.float64))
    return np.stack(columns, axis=1)


def fit_raw_scorer(raw: np.lib.npyio.NpzFile, specs: list[str]) -> tuple[np.ndarray, np.ndarray, object]:
    train_features = load_raw_matrix(raw, "train", specs)
    val_features = load_raw_matrix(raw, "val", specs)
    train_labels = np.asarray(raw["train_labels"], dtype=bool)
    scorer = fit_train_effect_scorer(train_features, train_labels, method="train_effect_sum")
    return scorer.score(train_features), scorer.score(val_features), scorer


def score_lc_with_scorer(
    transform: Transform,
    scorer: Scorer,
    raw: np.lib.npyio.NpzFile,
) -> tuple[np.ndarray, np.ndarray]:
    train_features = transform.apply(np.asarray(raw["train_roi_l2"], dtype=np.float64))
    val_features = transform.apply(np.asarray(raw["val_roi_l2"], dtype=np.float64))
    train_classes = np.asarray(raw["train_class_ids"], dtype=np.int64)
    val_classes = np.asarray(raw["val_class_ids"], dtype=np.int64)
    return scorer.score(train_features, train_classes), scorer.score(val_features, val_classes)


def fusion_scores(train_columns: list[np.ndarray], val_columns: list[np.ndarray], train_labels: np.ndarray, *, method: str) -> tuple[np.ndarray, np.ndarray]:
    train_matrix = np.stack([zscore(col) for col in train_columns], axis=1)
    val_matrix = np.stack(
        [
            (np.asarray(col, dtype=np.float64) - float(np.asarray(train_col, dtype=np.float64).mean()))
            / max(float(np.asarray(train_col, dtype=np.float64).std()), 1e-6)
            for col, train_col in zip(val_columns, train_columns)
        ],
        axis=1,
    )
    if method == "train_effect":
        scorer = fit_train_effect_scorer(train_matrix, train_labels, method="train_effect_sum")
        return scorer.score(train_matrix), scorer.score(val_matrix)
    if method == "logistic":
        scorer = fit_logistic(train_matrix, train_labels)
        return scorer.score(train_matrix), scorer.score(val_matrix)
    raise ValueError(f"Unknown fusion method: {method}")


def main() -> None:
    args = parse_args()
    run_dir = Path("runs") / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    save_json(vars(args), run_dir / "config.json")

    full = np.load(args.full_cache)
    raw = np.load(args.raw_cache)
    train_full_l2 = np.asarray(full["train_features_l2"], dtype=np.float64)
    val_full_l2 = np.asarray(full["val_features_l2"], dtype=np.float64)
    train_ious = np.asarray(full["train_best_iou"], dtype=np.float64)
    val_ious = np.asarray(full["val_best_iou"], dtype=np.float64)
    train_classes = np.asarray(full["train_class_id"], dtype=np.int64)
    val_classes = np.asarray(full["val_class_id"], dtype=np.int64)
    train_probs = np.asarray(full["train_matched_prob"], dtype=np.float64)
    val_probs = np.asarray(full["val_matched_prob"], dtype=np.float64)

    lc_train_labels = np.asarray(raw["train_labels"], dtype=bool)
    lc_val_labels = np.asarray(raw["val_labels"], dtype=bool)
    raw_train_scores, raw_val_scores, raw_scorer = fit_raw_scorer(raw, list(args.raw_features))

    report: dict[str, object] = {
        "full_cache": str(args.full_cache),
        "raw_cache": str(args.raw_cache),
        "raw_features": list(args.raw_features),
        "full_counts": {
            "train": int(train_ious.shape[0]),
            "val": int(val_ious.shape[0]),
        },
        "lc_counts": {
            "train": int(lc_train_labels.shape[0]),
            "train_positive": int(lc_train_labels.sum()),
            "val": int(lc_val_labels.shape[0]),
            "val_positive": int(lc_val_labels.sum()),
        },
        "raw_scorer": {
            "weights": raw_scorer.weights.astype(float).tolist(),
            "mean": raw_scorer.scaler.mean_.astype(float).tolist(),
            "scale": raw_scorer.scaler.scale_.astype(float).tolist(),
        },
        "experiments": {},
        "leaderboard": [],
    }

    raw_result = metrics_fixed_thresholds(raw_train_scores, lc_train_labels, raw_val_scores, lc_val_labels)
    report["experiments"]["raw_only"] = raw_result  # type: ignore[index]
    leaderboard = [leaderboard_row("raw_only", raw_result, group="raw")]

    regimes = {
        "full_allneg": {
            "train_mask": high_low_mask(train_ious, train_classes, high=args.high_iou_min, low=args.low_iou_max, matched_only=False),
            "val_mask": high_low_mask(val_ious, val_classes, high=args.high_iou_min, low=args.low_iou_max, matched_only=False),
        },
        "full_matchedneg": {
            "train_mask": high_low_mask(train_ious, train_classes, high=args.high_iou_min, low=args.low_iou_max, matched_only=True),
            "val_mask": high_low_mask(val_ious, val_classes, high=args.high_iou_min, low=args.low_iou_max, matched_only=True),
        },
        "lowconf_allneg": {
            "train_mask": high_low_mask(train_ious, train_classes, high=args.high_iou_min, low=args.low_iou_max, matched_only=False)
            & (train_probs <= float(args.low_conf_max)),
            "val_mask": high_low_mask(val_ious, val_classes, high=args.high_iou_min, low=args.low_iou_max, matched_only=False)
            & (val_probs <= float(args.low_conf_max)),
        },
        "lowconf_matchedneg": {
            "train_mask": high_low_mask(train_ious, train_classes, high=args.high_iou_min, low=args.low_iou_max, matched_only=True)
            & (train_probs <= float(args.low_conf_max)),
            "val_mask": high_low_mask(val_ious, val_classes, high=args.high_iou_min, low=args.low_iou_max, matched_only=True)
            & (val_probs <= float(args.low_conf_max)),
        },
    }

    transform_specs: list[tuple[str, int | None]] = [("l2", None), ("l2_z", None)]
    transform_specs.extend(("l2_z_pca", int(component)) for component in args.components)

    for regime_name, regime in regimes.items():
        train_mask = np.asarray(regime["train_mask"], dtype=bool)
        val_mask = np.asarray(regime["val_mask"], dtype=bool)
        train_labels = high_labels(train_ious[train_mask], high=args.high_iou_min)
        val_labels = high_labels(val_ious[val_mask], high=args.high_iou_min)
        if int(train_labels.sum()) == 0 or int((~train_labels).sum()) == 0:
            continue
        train_regime_features = train_full_l2[train_mask]
        val_regime_features = val_full_l2[val_mask]
        train_regime_classes = train_classes[train_mask]
        val_regime_classes = val_classes[val_mask]
        for transform_kind, components in transform_specs:
            transform = build_transform("l2" if transform_kind == "l2" else "l2_z", train_regime_features, components)
            train_x = transform.apply(train_regime_features)
            val_x = transform.apply(val_regime_features)
            fitters = {
                "center": fit_center,
                "knn": lambda x, y: fit_knn(x, y, k=int(args.knn_k)),
                "logistic": fit_logistic,
            }
            for scorer_name, fitter in fitters.items():
                global_scorer = fitter(train_x, train_labels)
                eval_scores = global_scorer.score(val_x, val_regime_classes)
                train_scores = global_scorer.score(train_x, train_regime_classes)
                full_eval = metrics_fixed_thresholds(train_scores, train_labels, eval_scores, val_labels)

                lc_train_hd, lc_val_hd = score_lc_with_scorer(transform, global_scorer, raw)
                lc_eval = metrics_fixed_thresholds(lc_train_hd, lc_train_labels, lc_val_hd, lc_val_labels)
                name = f"{regime_name}/{transform.name}/{scorer_name}"
                report["experiments"][name] = {  # type: ignore[index]
                    "full_eval": full_eval,
                    "lc_eval": lc_eval,
                    "train_count": int(train_labels.shape[0]),
                    "train_positive": int(train_labels.sum()),
                    "val_count": int(val_labels.shape[0]),
                    "val_positive": int(val_labels.sum()),
                }
                leaderboard.append(leaderboard_row(name, lc_eval, group="hd_lc", extra={"full_auc": full_eval["ranking"]["auc"]}))  # type: ignore[index]

                for fusion_method in ("train_effect", "logistic"):
                    f_train, f_val = fusion_scores(
                        [raw_train_scores, lc_train_hd],
                        [raw_val_scores, lc_val_hd],
                        lc_train_labels,
                        method=fusion_method,
                    )
                    fusion_eval = metrics_fixed_thresholds(f_train, lc_train_labels, f_val, lc_val_labels)
                    fusion_name = f"fusion_raw_hd/{regime_name}/{transform.name}/{scorer_name}/{fusion_method}"
                    report["experiments"][fusion_name] = fusion_eval  # type: ignore[index]
                    leaderboard.append(leaderboard_row(fusion_name, fusion_eval, group="fusion2"))

                    f3_train, f3_val = fusion_scores(
                        [raw_train_scores, lc_train_hd, np.asarray(raw["train_label_probs"], dtype=np.float64)],
                        [raw_val_scores, lc_val_hd, np.asarray(raw["val_label_probs"], dtype=np.float64)],
                        lc_train_labels,
                        method=fusion_method,
                    )
                    fusion3_eval = metrics_fixed_thresholds(f3_train, lc_train_labels, f3_val, lc_val_labels)
                    fusion3_name = f"fusion_raw_hd_prob/{regime_name}/{transform.name}/{scorer_name}/{fusion_method}"
                    report["experiments"][fusion3_name] = fusion3_eval  # type: ignore[index]
                    leaderboard.append(leaderboard_row(fusion3_name, fusion3_eval, group="fusion3"))

                if scorer_name in {"center", "knn"}:
                    class_scorer, diagnostics = fit_classwise(
                        fitter,
                        global_scorer,
                        train_x,
                        train_labels,
                        train_regime_classes,
                        min_pos=int(args.classwise_min_pos),
                        min_neg=int(args.classwise_min_neg),
                    )
                    class_train_scores = class_scorer.score(train_x, train_regime_classes)
                    class_val_scores = class_scorer.score(val_x, val_regime_classes)
                    class_full_eval = metrics_fixed_thresholds(class_train_scores, train_labels, class_val_scores, val_labels)
                    lc_train_class_hd, lc_val_class_hd = score_lc_with_scorer(transform, class_scorer, raw)
                    class_lc_eval = metrics_fixed_thresholds(lc_train_class_hd, lc_train_labels, lc_val_class_hd, lc_val_labels)
                    class_name = f"{regime_name}/{transform.name}/classwise_{scorer_name}"
                    report["experiments"][class_name] = {  # type: ignore[index]
                        "full_eval": class_full_eval,
                        "lc_eval": class_lc_eval,
                        "diagnostics": diagnostics,
                    }
                    leaderboard.append(leaderboard_row(class_name, class_lc_eval, group="hd_lc_classwise", extra={"full_auc": class_full_eval["ranking"]["auc"]}))  # type: ignore[index]

                    for fusion_method in ("train_effect", "logistic"):
                        f_train, f_val = fusion_scores(
                            [raw_train_scores, lc_train_class_hd],
                            [raw_val_scores, lc_val_class_hd],
                            lc_train_labels,
                            method=fusion_method,
                        )
                        fusion_eval = metrics_fixed_thresholds(f_train, lc_train_labels, f_val, lc_val_labels)
                        fusion_name = f"fusion_raw_hd/{regime_name}/{transform.name}/classwise_{scorer_name}/{fusion_method}"
                        report["experiments"][fusion_name] = fusion_eval  # type: ignore[index]
                        leaderboard.append(leaderboard_row(fusion_name, fusion_eval, group="fusion2_classwise"))

                        f3_train, f3_val = fusion_scores(
                            [raw_train_scores, lc_train_class_hd, np.asarray(raw["train_label_probs"], dtype=np.float64)],
                            [raw_val_scores, lc_val_class_hd, np.asarray(raw["val_label_probs"], dtype=np.float64)],
                            lc_train_labels,
                            method=fusion_method,
                        )
                        fusion3_eval = metrics_fixed_thresholds(f3_train, lc_train_labels, f3_val, lc_val_labels)
                        fusion3_name = f"fusion_raw_hd_prob/{regime_name}/{transform.name}/classwise_{scorer_name}/{fusion_method}"
                        report["experiments"][fusion3_name] = fusion3_eval  # type: ignore[index]
                        leaderboard.append(leaderboard_row(fusion3_name, fusion3_eval, group="fusion3_classwise"))

    leaderboard = sorted(
        leaderboard,
        key=lambda row: (
            float(row["fixed_p0.7_recall"]),
            float(row["fixed_p0.7_precision"]),
            float(row["average_precision"]),
            -float(row["fixed_p0.7_fp"]),
        ),
        reverse=True,
    )
    report["leaderboard"] = leaderboard[:200]
    save_json(report, run_dir / "large_scale_verifier_fusion_report.json")
    print(json.dumps(leaderboard[:40], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
