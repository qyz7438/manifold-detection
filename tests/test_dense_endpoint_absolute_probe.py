from __future__ import annotations

import torch

from scripts.train_nwpu_dense_endpoint_absolute import (
    feature_alignment_shuffle,
    load_config,
    regression_metrics,
)


def test_absolute_endpoint_config_locks_nested_outer_boundary():
    config = load_config()

    assert config["dataset"]["detector_validation_forbidden"] is True
    assert config["dataset"]["outer_read_once_after_selection"] is True
    assert config["teacher"]["calibration_error_role"] == "diagnostic_only_not_in_quality"


def test_feature_alignment_shuffle_preserves_column_marginals_but_breaks_rows():
    features = torch.arange(4 * 22, dtype=torch.float32).reshape(4, 22)
    shuffled = feature_alignment_shuffle(features, seed=13)

    assert torch.equal(shuffled[:, 0], features[:, 0])
    assert torch.equal(shuffled[:, 7:], features[:, 7:])
    for column in range(1, 7):
        assert torch.equal(shuffled[:, column].sort().values, features[:, column].sort().values)
    assert not torch.equal(shuffled[:, 1:7], features[:, 1:7])


def test_regression_metrics_report_ranking_and_constant_baselines():
    target = torch.tensor([-2.0, -1.0, 1.0, 2.0])
    perfect = regression_metrics(target, target, fit_mean=0.0)
    reversed_metrics = regression_metrics(-target, target, fit_mean=0.0)

    assert perfect["mae"] == 0.0
    assert perfect["pairwise_accuracy"] == 1.0
    assert perfect["auroc_positive"] == 1.0
    assert reversed_metrics["pairwise_accuracy"] == 0.0
    assert reversed_metrics["auroc_positive"] == 0.0
    assert perfect["zero_baseline_mae"] == perfect["fit_mean_baseline_mae"] == 1.5
