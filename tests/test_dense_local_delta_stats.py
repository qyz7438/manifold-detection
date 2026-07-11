from __future__ import annotations

import torch

from scripts.analyze_dense_local_delta_stats import apply_set_perturbation, load_config


def _prediction():
    return {
        "boxes": torch.tensor([[1.0, 1.0, 5.0, 5.0], [6.0, 6.0, 9.0, 9.0]]),
        "scores": torch.tensor([0.8, 0.6]),
        "labels": torch.tensor([1, 2]),
    }


def test_local_delta_config_forbids_detector_validation():
    config = load_config()

    assert config["dataset"]["detector_validation_forbidden"] is True
    assert config["perturbations"]["top_detections_per_image"] == 3


def test_identity_permutation_preserves_set_and_drop_removes_one():
    prediction = _prediction()
    identity = apply_set_perturbation(
        prediction, "identity_permutation", 0, (10, 10), score_step=0.02, box_step=0.02
    )
    dropped = apply_set_perturbation(
        prediction, "drop", 0, (10, 10), score_step=0.02, box_step=0.02
    )

    assert torch.equal(identity["scores"], prediction["scores"].flip(0))
    assert torch.equal(identity["labels"], prediction["labels"].flip(0))
    assert dropped["boxes"].shape == (1, 4)
    assert prediction["boxes"].shape == (2, 4)


def test_score_and_box_perturbations_are_bounded():
    prediction = _prediction()
    scored = apply_set_perturbation(
        prediction, "score_up", 0, (10, 10), score_step=0.02, box_step=0.02
    )
    moved = apply_set_perturbation(
        prediction, "translate_right", 0, (10, 10), score_step=0.02, box_step=0.02
    )

    assert torch.isclose(scored["scores"][0], torch.tensor(0.82))
    assert torch.allclose(moved["boxes"][0], torch.tensor([1.08, 1.0, 5.08, 5.0]))
    assert torch.equal(moved["scores"], prediction["scores"])
