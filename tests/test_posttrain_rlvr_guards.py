import json

import pytest

from spectral_detection_posttrain.train.posttrain_rlvr import (
    load_r_amp_stats_for_signal,
    validate_policy_objective_args,
)


def test_amp_signal_requires_stats_path():
    with pytest.raises(ValueError, match="r-amp-stats"):
        load_r_amp_stats_for_signal("ramp", None)


def test_amp_stats_must_have_samples_and_percentiles(tmp_path):
    stats_path = tmp_path / "r_amp_stats.json"
    stats_path.write_text(json.dumps({"p05": 0.0, "p95": 1.0, "count": 0}), encoding="utf-8")

    with pytest.raises(ValueError, match="no samples"):
        load_r_amp_stats_for_signal("amp_structure", str(stats_path))


def test_valid_amp_stats_are_loaded(tmp_path):
    stats_path = tmp_path / "r_amp_stats.json"
    stats_path.write_text(json.dumps({"p05": 0.1, "p95": 0.9, "count": 12}), encoding="utf-8")

    stats = load_r_amp_stats_for_signal("shuffled_amp", str(stats_path))

    assert stats == {"p05": 0.1, "p95": 0.9, "count": 12}


def test_non_amp_signal_does_not_require_stats():
    assert load_r_amp_stats_for_signal("structure", None) is None


def test_weighted_ce_rejects_unimplemented_bbox_box_loss():
    with pytest.raises(ValueError, match="bbox targets"):
        validate_policy_objective_args("weighted_ce", 1.0)


def test_signed_objective_allows_box_loss_argument():
    validate_policy_objective_args("signed", 1.0)
