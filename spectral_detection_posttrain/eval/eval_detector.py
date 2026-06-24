from __future__ import annotations

import argparse

import torch
from tqdm import tqdm

from spectral_detection_posttrain.datasets import build_penn_fudan_loaders
from spectral_detection_posttrain.datasets.patch_transform import add_detection_patch
from spectral_detection_posttrain.eval.detection_metrics import evaluate_detection_predictions, summarize_iou_diagnostics
from spectral_detection_posttrain.experiments.canonical_runner import build_experiment_model, prepare_experiment
from spectral_detection_posttrain.core.matching.pred_gt_matcher import match_predictions_to_gt
from spectral_detection_posttrain.utils.io import save_json
from spectral_detection_posttrain.utils.seed import resolve_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate detector on clean or patched Penn-Fudan.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument("--patch-mode", default="none", choices=["none", "background", "object", "edge", "random", "object_inside", "object_edge", "near_object"])
    parser.add_argument("--patch-type", default=None, choices=["random", "checkerboard", "qr", "qr_like", "qr-like"])
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    context = prepare_experiment(args.config, args.run_name, phase="eval", checkpoint_path=args.checkpoint)
    config = context.config
    set_seed(int(config.get("seed", 42)))
    run_dir = context.run_dir

    _, val_loader = build_penn_fudan_loaders(
        config,
        limit_train=1,
        limit_val=args.limit_val,
        batch_size=int(config["eval"].get("batch_size", 2)),
    )
    device = resolve_device(config)
    model = build_experiment_model(context, checkpoint_path=args.checkpoint, device=device, pretrained=False)
    model.eval()

    predictions = []
    targets = []
    patch_cfg = config.get("patch", {})
    patch_type = args.patch_type or str(patch_cfg.get("patch_type", "random"))
    for images, batch_targets in tqdm(val_loader, desc=args.run_name):
        if args.patch_mode != "none":
            images = [
                add_detection_patch(
                    image,
                    target,
                    placement=args.patch_mode,
                    patch_type=patch_type,
                    patch_size=int(patch_cfg.get("patch_size", 48)),
                )
                for image, target in zip(images, batch_targets)
            ]
        outputs = model([image.to(device) for image in images])
        predictions.extend([{k: v.detach().cpu() for k, v in output.items()} for output in outputs])
        targets.extend([{k: v.detach().cpu() if torch.is_tensor(v) else v for k, v in target.items()} for target in batch_targets])

    metrics = evaluate_detection_predictions(
        predictions,
        targets,
        iou_threshold=float(config["matching"].get("iou_threshold", 0.5)),
        score_threshold=float(config["matching"].get("score_threshold", 0.05)),
        high_conf_threshold=float(config["eval"].get("high_conf_threshold", 0.7)),
    )
    matched_ious: list[float] = []
    matched_scores: list[float] = []
    for prediction, target in zip(predictions, targets):
        pred_cpu = {k: v.detach().cpu() if torch.is_tensor(v) else v for k, v in prediction.items()}
        tgt_cpu = {k: v.detach().cpu() if torch.is_tensor(v) else v for k, v in target.items()}
        m = match_predictions_to_gt(pred_cpu, tgt_cpu, iou_threshold=float(config["matching"].get("iou_threshold", 0.5)),
                                     score_threshold=float(config["matching"].get("score_threshold", 0.05)))
        scores = pred_cpu.get("scores", torch.empty((0,)))
        for match in m["matches"]:
            matched_ious.append(match["iou"])
            matched_scores.append(float(scores[match["pred_index"]].item()))

    metrics.update(summarize_iou_diagnostics(matched_ious, matched_scores))
    metrics["patch_mode"] = args.patch_mode
    metrics["patch_type"] = patch_type if args.patch_mode != "none" else "none"
    save_json(metrics, run_dir / "eval_metrics.json")
    print(metrics)


if __name__ == "__main__":
    main()
