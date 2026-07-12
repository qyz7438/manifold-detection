"""Post-hoc family and target-margin audit for the frozen local Delta-Q learner."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CONFIG_PATH = (
    ROOT
    / "spectral_detection_posttrain"
    / "configs"
    / "versions"
    / "det.energy.dense_local_delta_family_audit.001.json"
)
CONFIG_SHA256 = "cb412b9c6f6ffffae2a2b681f5472defac93f0c588fff73656688333cc3cf490"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pair_score(prediction_delta: torch.Tensor, target_delta: torch.Tensor) -> float:
    product = float(prediction_delta * target_delta)
    if product > 0:
        return 1.0
    if product == 0:
        return 0.5
    return 0.0


def _summarize_pair_groups(
    groups: dict[str, dict[int, list[tuple[float | None, float]]]]
) -> dict[str, dict[str, float | int]]:
    result: dict[str, dict[str, float | int]] = {}
    for key in sorted(groups):
        per_image = groups[key]
        image_scores = {
            image_id: [score for score, _ in values if score is not None]
            for image_id, values in per_image.items()
        }
        image_accuracies = [sum(values) / len(values) for values in image_scores.values() if values]
        flat = [item for values in per_image.values() for item in values]
        eligible = [score for score, _ in flat if score is not None]
        if not flat:
            continue
        result[key] = {
            "images": len(image_accuracies),
            "candidate_pairs": len(flat),
            "eligible_pairs": len(eligible),
            "zero_margin_pairs": sum(margin <= 1e-6 for _, margin in flat),
            "image_equal_accuracy": sum(image_accuracies) / len(image_accuracies) if image_accuracies else None,
            "candidate_weighted_accuracy": sum(eligible) / len(eligible) if eligible else None,
            "mean_abs_target_margin": sum(margin for _, margin in flat) / len(flat),
            "per_image_accuracy": {
                str(image_id): sum(values) / len(values)
                for image_id, values in sorted(image_scores.items())
                if values
            },
        }
    return result


def _iter_within_image_pairs(
    prediction: torch.Tensor,
    target: torch.Tensor,
    image_ids: torch.Tensor,
):
    prediction = torch.as_tensor(prediction).flatten().double().cpu()
    target = torch.as_tensor(target).flatten().double().cpu()
    image_ids = torch.as_tensor(image_ids).flatten().long().cpu()
    if prediction.numel() != target.numel() or target.numel() != image_ids.numel():
        raise ValueError("prediction, target, and image_ids must have equal length")
    for image_id in torch.unique(image_ids, sorted=True):
        rows = torch.nonzero(image_ids.eq(image_id), as_tuple=False).flatten()
        indices = torch.triu_indices(rows.numel(), rows.numel(), offset=1)
        for left, right in zip(rows[indices[0]].tolist(), rows[indices[1]].tolist()):
            target_delta = target[left] - target[right]
            prediction_delta = prediction[left] - prediction[right]
            yield int(image_id), left, right, prediction_delta, target_delta, abs(float(target_delta))


def family_pair_breakdown(
    prediction: torch.Tensor,
    target: torch.Tensor,
    image_ids: torch.Tensor,
    families: Sequence[str],
) -> dict[str, dict[str, float | int]]:
    if len(families) != torch.as_tensor(target).numel():
        raise ValueError("families must match target length")
    groups: dict[str, dict[int, list[tuple[float | None, float]]]] = defaultdict(lambda: defaultdict(list))
    for image_id, left, right, predicted_delta, target_delta, margin in _iter_within_image_pairs(prediction, target, image_ids):
        key = "|".join(sorted((str(families[left]), str(families[right]))))
        score = None if margin <= 1e-6 else _pair_score(predicted_delta, target_delta)
        groups[key][image_id].append((score, margin))
    return _summarize_pair_groups(groups)


def margin_breakdown(
    prediction: torch.Tensor,
    target: torch.Tensor,
    image_ids: torch.Tensor,
    boundaries: Sequence[float],
) -> dict[str, dict[str, float | int]]:
    edges = [float(value) for value in boundaries]
    if len(edges) != 3 or edges[0] <= 0.0 or any(a >= b for a, b in zip(edges, edges[1:])):
        raise ValueError("margin boundaries must contain epsilon, near-zero, and beta cutoffs")
    epsilon, near_zero, beta = edges
    labels = (f"[0,{epsilon:g}]", f"({epsilon:g},{near_zero:g}]", f"({near_zero:g},{beta:g}]", f"({beta:g},inf)")
    groups: dict[str, dict[int, list[tuple[float | None, float]]]] = defaultdict(lambda: defaultdict(list))
    for image_id, _, _, predicted_delta, target_delta, margin in _iter_within_image_pairs(prediction, target, image_ids):
        if margin <= epsilon:
            key, score = labels[0], None
        elif margin <= near_zero:
            key, score = labels[1], _pair_score(predicted_delta, target_delta)
        elif margin <= beta:
            key, score = labels[2], _pair_score(predicted_delta, target_delta)
        else:
            key, score = labels[3], _pair_score(predicted_delta, target_delta)
        groups[key][image_id].append((score, margin))
    return _summarize_pair_groups(groups)


def _paired_bootstrap(values: Sequence[float], *, seed: int = 42, samples: int = 10000) -> dict[str, float | int]:
    tensor = torch.as_tensor(values, dtype=torch.float64)
    if tensor.numel() == 0:
        return {"images": 0, "mean": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan")}
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randint(tensor.numel(), (samples, tensor.numel()), generator=generator)
    means = tensor[indices].mean(dim=1)
    interval = torch.quantile(means, torch.tensor([0.025, 0.975], dtype=torch.float64))
    return {
        "images": int(tensor.numel()),
        "mean": float(tensor.mean().item()),
        "ci95_low": float(interval[0].item()),
        "ci95_high": float(interval[1].item()),
    }


def _paired_control_differences(
    full: dict[str, dict[str, Any]], control: dict[str, dict[str, Any]]
) -> dict[str, dict[str, float | int]]:
    output = {}
    for key in sorted(set(full) & set(control)):
        full_images = full[key]["per_image_accuracy"]
        control_images = control[key]["per_image_accuracy"]
        common = sorted(set(full_images) & set(control_images), key=int)
        output[key] = _paired_bootstrap(
            [float(full_images[image_id]) - float(control_images[image_id]) for image_id in common]
        )
    return output


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


@torch.inference_mode()
def run(args: argparse.Namespace) -> dict[str, Any]:
    from scripts.train_dense_local_delta_learner import (
        _new_model,
        _predict,
        imagewise_pairwise_accuracy,
        load_config as load_learner_config,
        split_image_ids,
    )

    if sha256_file(CONFIG_PATH) != CONFIG_SHA256:
        raise ValueError("canonical family audit config SHA256 mismatch")
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    learner_config = load_learner_config()
    run_dir = (args.run_dir or ROOT / config["source"]["run_dir"]).resolve()
    source_metrics_path = run_dir / "eval_metrics.json"
    cache_path = run_dir / "pair_cache.pt"
    if sha256_file(source_metrics_path) != config["source"]["eval_metrics_sha256"]:
        raise ValueError("source eval metrics SHA256 mismatch")
    if sha256_file(cache_path) != config["source"]["pair_cache_sha256"]:
        raise ValueError("pair cache SHA256 mismatch")
    source_metrics = json.loads(source_metrics_path.read_text(encoding="utf-8"))
    if source_metrics.get("scientific_status") != config["source"]["required_status"]:
        raise ValueError("source learner must remain frozen")
    cache = torch.load(cache_path, map_location="cpu")
    if cache.get("format") != "dense_local_delta_pairs_v1" or cache.get("config_sha256") != source_metrics["config_sha256"]:
        raise ValueError("pair cache format or learner config SHA256 mismatch")
    records = cache["records"]
    source_ids = sorted({int(row["image_id"]) for row in records})
    _, tune_ids = split_image_ids(source_ids, learner_config)
    tune_set = set(tune_ids)
    all_tune_records = [row for row in records if int(row["image_id"]) in tune_set]
    tune_records = [row for row in all_tune_records if not bool(row["identity"])]
    target = torch.tensor([row["target"] for row in tune_records], dtype=torch.float32)
    image_ids = torch.tensor([row["image_id"] for row in tune_records], dtype=torch.long)
    families = [str(row["family"]) for row in tune_records]
    device = torch.device(args.device)
    arms: dict[str, Any] = {}
    reference_stats = None
    for arm in config["arms"]:
        checkpoint_path = run_dir / arm / "endpoint.pth"
        if sha256_file(checkpoint_path) != config["source"]["checkpoint_sha256"][arm]:
            raise ValueError(f"{arm} checkpoint SHA256 mismatch")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if checkpoint.get("arm") != arm or checkpoint.get("config_sha256") != source_metrics["config_sha256"]:
            raise ValueError(f"{arm} checkpoint metadata mismatch")
        if reference_stats is None:
            reference_stats = checkpoint["feature_stats"]
        elif any(
            not torch.equal(reference_stats[key], checkpoint["feature_stats"][key])
            for key in reference_stats
        ):
            raise ValueError("checkpoint feature statistics differ across arms")
        model = _new_model(learner_config, device)
        model.load_state_dict(checkpoint["model"], strict=True)
        prediction = _predict(model, tune_records, arm, checkpoint["feature_stats"], device)
        prediction_all = _predict(model, all_tune_records, arm, checkpoint["feature_stats"], device)
        identity_mask = torch.tensor([bool(row["identity"]) for row in all_tune_records])
        family_pairs = family_pair_breakdown(prediction, target, image_ids, families)
        overall = family_pair_breakdown(prediction, target, image_ids, ["all"] * len(families))["all|all"]
        arms[arm] = {
            "original_pooled_pairwise_accuracy": imagewise_pairwise_accuracy(
                prediction, target, image_ids
            ),
            "protocol_overall": overall,
            "identity_max_abs_prediction": float(prediction_all[identity_mask].abs().max().item()),
            "family_pairs": family_pairs,
            "margin_bins": margin_breakdown(prediction, target, image_ids, config["margin_boundaries"]),
        }
    control_differences = {
        control: {
            "overall": _paired_control_differences(
                {"all": arms["local_full"]["protocol_overall"]},
                {"all": arms[control]["protocol_overall"]},
            )["all"],
            "family_pairs": _paired_control_differences(
                arms["local_full"]["family_pairs"], arms[control]["family_pairs"]
            ),
        }
        for control in ("strong_geometry_shuffle", "utility_shuffle")
    }
    zero_prediction = torch.zeros_like(target)
    zero_overall = family_pair_breakdown(zero_prediction, target, image_ids, ["all"] * len(families))["all|all"]
    minimum_images = int(config["reporting"]["minimum_descriptive_support_images"])
    minimum_pairs = int(config["reporting"]["minimum_descriptive_support_pairs"])
    supported = [
        (key, value)
        for key, value in arms["local_full"]["family_pairs"].items()
        if int(value["images"]) >= minimum_images and int(value["eligible_pairs"]) >= minimum_pairs
    ]
    supported.sort(key=lambda item: float(item[1]["image_equal_accuracy"]))
    result = {
        "completed": True,
        "scientific_status": "posthoc_family_diagnostic_complete",
        "version_id": config["version_id"],
        "config_sha256": CONFIG_SHA256,
        "source_eval_metrics_sha256": config["source"]["eval_metrics_sha256"],
        "pair_cache_sha256": config["source"]["pair_cache_sha256"],
        "new_training": False,
        "detector_inference": False,
        "detector_validation_read": False,
        "tune_images": len(set(image_ids.tolist())),
        "tune_rows": int(target.numel()),
        "arms": arms,
        "zero_predictor": {"protocol_overall": zero_overall},
        "paired_control_differences": control_differences,
        "descriptive_extremes": {
            "support_rule": {"images": minimum_images, "eligible_pairs": minimum_pairs},
            "lowest_five": [{"family_pair": key, **value} for key, value in supported[:5]],
            "highest_five": [{"family_pair": key, **value} for key, value in supported[-5:]],
        },
        "decision_rule": config["decision_rule"],
        "claim_boundary": config["claim_boundary"],
        "reopens_frozen_learner": False,
    }
    output = (args.output or run_dir / "family_audit_metrics.json").resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    result = run(_parse_args())
    print(json.dumps({"status": result["scientific_status"], "tune_images": result["tune_images"], "tune_rows": result["tune_rows"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
