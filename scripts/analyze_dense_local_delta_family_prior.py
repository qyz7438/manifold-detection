"""Compare the frozen local Delta-Q learner with a fit-only perturbation-family prior."""

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
    / "det.energy.dense_local_delta_family_prior.001.json"
)
CONFIG_SHA256 = "d2723fe0fba3526035c35c32aa3206c27863f2a5edda0f2aa72c1362ecd489c5"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fit_family_means(families: Sequence[str], targets: torch.Tensor) -> dict[str, Any]:
    values = torch.as_tensor(targets, dtype=torch.float64).flatten()
    if len(families) != values.numel():
        raise ValueError("families and targets must match")
    grouped: dict[str, list[float]] = defaultdict(list)
    for family, target in zip(families, values.tolist()):
        grouped[str(family)].append(float(target))
    return {
        "global_mean": float(values.mean().item()),
        "family_means": {
            family: sum(family_values) / len(family_values)
            for family, family_values in sorted(grouped.items())
        },
        "family_rows": {family: len(family_values) for family, family_values in sorted(grouped.items())},
    }


def predict_family_means(families: Sequence[str], fitted: dict[str, Any]) -> torch.Tensor:
    means = fitted["family_means"]
    fallback = float(fitted["global_mean"])
    return torch.tensor([float(means.get(str(family), fallback)) for family in families], dtype=torch.float32)


def _pearson(left: torch.Tensor, right: torch.Tensor) -> float | None:
    left = torch.as_tensor(left).double().flatten()
    right = torch.as_tensor(right).double().flatten()
    left = left - left.mean()
    right = right - right.mean()
    denominator = left.square().sum().sqrt() * right.square().sum().sqrt()
    return float((left * right).sum().div(denominator).item()) if float(denominator) > 0 else None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


@torch.inference_mode()
def run(args: argparse.Namespace) -> dict[str, Any]:
    from scripts.train_dense_local_delta_learner import (
        _metrics,
        _new_model,
        _predict,
        load_config as load_learner_config,
        split_image_ids,
    )

    if sha256_file(CONFIG_PATH) != CONFIG_SHA256:
        raise ValueError("canonical family-prior config SHA256 mismatch")
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    learner_config = load_learner_config()
    run_dir = (args.run_dir or ROOT / config["source"]["run_dir"]).resolve()
    metrics_path = run_dir / "eval_metrics.json"
    cache_path = run_dir / "pair_cache.pt"
    checkpoint_path = run_dir / "local_full" / "endpoint.pth"
    for path, expected in (
        (metrics_path, config["source"]["eval_metrics_sha256"]),
        (cache_path, config["source"]["pair_cache_sha256"]),
        (checkpoint_path, config["source"]["local_full_checkpoint_sha256"]),
    ):
        if sha256_file(path) != expected:
            raise ValueError(f"source SHA256 mismatch: {path.name}")
    source_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if source_metrics.get("scientific_status") != config["source"]["required_status"]:
        raise ValueError("source learner must remain frozen")
    cache = torch.load(cache_path, map_location="cpu")
    if cache.get("format") != "dense_local_delta_pairs_v1" or cache.get("config_sha256") != source_metrics["config_sha256"]:
        raise ValueError("pair cache metadata mismatch")
    records = cache["records"]
    source_ids = sorted({int(row["image_id"]) for row in records})
    fit_ids, tune_ids = split_image_ids(source_ids, learner_config)
    fit_set, tune_set = set(fit_ids), set(tune_ids)
    fit_records = [row for row in records if int(row["image_id"]) in fit_set and not bool(row["identity"])]
    tune_records = [row for row in records if int(row["image_id"]) in tune_set]
    fitted = fit_family_means(
        [str(row["family"]) for row in fit_records],
        torch.tensor([row["target"] for row in fit_records]),
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if checkpoint.get("arm") != "local_full" or checkpoint.get("config_sha256") != source_metrics["config_sha256"]:
        raise ValueError("local-full checkpoint metadata mismatch")
    device = torch.device(args.device)
    model = _new_model(learner_config, device)
    model.load_state_dict(checkpoint["model"], strict=True)
    neural = _predict(model, tune_records, "local_full", checkpoint["feature_stats"], device)
    targets = torch.tensor([row["target"] for row in tune_records])
    image_ids = torch.tensor([row["image_id"] for row in tune_records])
    identity = torch.tensor([bool(row["identity"]) for row in tune_records])
    family_prior = predict_family_means([str(row["family"]) for row in tune_records], fitted)
    family_prior[identity] = 0.0
    global_prior = torch.full_like(targets, float(fitted["global_mean"]))
    global_prior[identity] = 0.0
    zero = torch.zeros_like(targets)
    evaluations = {
        "local_full": _metrics(neural, targets, image_ids, identity),
        "family_fit_mean": _metrics(family_prior, targets, image_ids, identity),
        "global_fit_mean": _metrics(global_prior, targets, image_ids, identity),
        "zero": _metrics(zero, targets, image_ids, identity),
    }
    selected = ~identity
    result = {
        "completed": True,
        "scientific_status": "posthoc_family_prior_diagnostic_complete",
        "version_id": config["version_id"],
        "config_sha256": CONFIG_SHA256,
        "source_eval_metrics_sha256": config["source"]["eval_metrics_sha256"],
        "new_training": False,
        "detector_inference": False,
        "detector_validation_read": False,
        "fit_images": len(fit_ids),
        "tune_images": len(tune_ids),
        "fitted_prior": fitted,
        "evaluations": evaluations,
        "neural_minus_family": {
            "pairwise_accuracy": float(evaluations["local_full"]["imagewise_pairwise_accuracy"]) - float(evaluations["family_fit_mean"]["imagewise_pairwise_accuracy"]),
            "sign_auroc": float(evaluations["local_full"]["sign_auroc"]) - float(evaluations["family_fit_mean"]["sign_auroc"]),
            "mae_improvement": float(evaluations["family_fit_mean"]["mae"]) - float(evaluations["local_full"]["mae"]),
            "prediction_pearson": _pearson(neural[selected], family_prior[selected]),
        },
        "decision_rule": config["decision_rule"],
        "claim_boundary": config["claim_boundary"],
        "reopens_frozen_learner": False,
    }
    output = (args.output or run_dir / "family_prior_metrics.json").resolve()
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    result = run(_parse_args())
    print(json.dumps({"status": result["scientific_status"], "neural_minus_family": result["neural_minus_family"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
