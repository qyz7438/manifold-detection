from __future__ import annotations

import torch

from scripts.train_dense_local_delta_learner import (
    imagewise_pairwise_accuracy,
    load_config,
    split_image_ids,
)


def test_local_learner_config_locks_train_only_boundary():
    config = load_config()

    assert config["dataset"]["all_other_splits_forbidden"] is True
    assert config["model"]["prediction"] == "Q_perturbed_minus_Q_base"


def test_local_learner_split_is_disjoint_and_complete():
    config = load_config()
    fit, tune = split_image_ids(list(range(64)), config)

    assert len(fit) == 48 and len(tune) == 16
    assert set(fit).isdisjoint(tune)
    assert sorted([*fit, *tune]) == list(range(64))


def test_imagewise_pairwise_accuracy_uses_only_within_image_pairs():
    prediction = torch.tensor([0.0, 1.0, 3.0, 2.0])
    target = torch.tensor([0.0, 1.0, 2.0, 3.0])
    image_ids = torch.tensor([1, 1, 2, 2])

    assert imagewise_pairwise_accuracy(prediction, target, image_ids) == 0.5
