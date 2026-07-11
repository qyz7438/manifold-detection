from __future__ import annotations

import torch

from scripts.train_dense_endpoint_geometry_control import (
    apply_geometry_control,
    load_config,
    resplit_image_ids,
)


def test_geometry_control_config_locks_previous_split_boundary():
    config = load_config()

    assert config["sources"]["old_inner_tune_deserialized_but_discarded_before_analysis"] is True
    assert config["sources"]["old_outer_forbidden"] is True
    assert config["sources"]["new_data_forbidden"] is True


def test_resplit_is_deterministic_disjoint_and_complete():
    config = load_config()
    image_ids = list(range(318))
    fit, holdout = resplit_image_ids(image_ids, config)

    assert len(fit) == 254 and len(holdout) == 64
    assert set(fit).isdisjoint(holdout)
    assert sorted([*fit, *holdout]) == image_ids


def test_strong_geometry_control_shuffles_all_geometry_columns():
    config = load_config()
    features = torch.arange(4 * 22, dtype=torch.float32).reshape(4, 22)
    controlled = apply_geometry_control(features, seed=7, config=config)
    columns = config["controls"]["strong_geometry_columns"]
    keep = [index for index in range(22) if index not in columns]

    assert torch.equal(controlled[:, keep], features[:, keep])
    for column in columns:
        assert torch.equal(controlled[:, column].sort().values, features[:, column].sort().values)
    assert not torch.equal(controlled[:, columns], features[:, columns])
