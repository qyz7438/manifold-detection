# Compatibility shim; new code should use canonical trainers paths.
from legacy.trainers.detection.train_manifold_posttrain import *  # noqa: F401,F403


@torch.no_grad()
def eval_metrics(model, val_loader, device, config: dict, num_classes: int) -> dict:
    """Evaluate through canonical globals so instrumentation and monkeypatching work."""
    model.eval()
    predictions = []
    targets = []
    for images, batch_targets in val_loader:
        outputs = model([image.to(device) for image in images])
        predictions.extend(
            [{key: value.detach().cpu() for key, value in output.items()} for output in outputs]
        )
        targets.extend(
            [
                {
                    key: value.detach().cpu() if torch.is_tensor(value) else value
                    for key, value in target.items()
                }
                for target in batch_targets
            ]
        )
    return evaluate_detection_predictions(
        predictions,
        targets,
        iou_threshold=float(config["matching"].get("iou_threshold", 0.5)),
        score_threshold=float(config["matching"].get("score_threshold", 0.05)),
        high_conf_threshold=float(config["eval"].get("high_conf_threshold", 0.7)),
        per_class=True,
        num_classes=num_classes,
    )
