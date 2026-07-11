from __future__ import annotations

import pytest

from scripts.validate_dense_endpoint_cleanval import combine_train_records, load_config


def test_cleanval_config_locks_single_read_and_frozen_hyperparameters():
    config = load_config()

    assert config["dataset"]["validation_read_once_after_all_models_frozen"] is True
    assert config["training"]["all_hyperparameters_frozen"] is True
    assert config["arms"][-1] == "teacher_shuffle"


def test_combine_train_records_requires_disjoint_unique_image_ids():
    first = [{"image_id": 1}, {"image_id": 2}]
    second = [{"image_id": 3}]

    combined = combine_train_records(first, second)

    assert [row["image_id"] for row in combined] == [1, 2, 3]
    with pytest.raises(ValueError, match="duplicate"):
        combine_train_records(first, [{"image_id": 2}])
