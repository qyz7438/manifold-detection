from __future__ import annotations

import torch

from scripts.audit_dense_endpoint_shift import empirical_ks, load_config


def test_shift_audit_forbids_new_training_and_inference():
    config = load_config()

    assert config["sources"]["new_inference_forbidden"] is True
    assert config["sources"]["training_forbidden"] is True


def test_empirical_ks_is_zero_for_same_values_and_one_for_disjoint_ranges():
    left = torch.tensor([0.0, 1.0, 2.0])

    assert empirical_ks(left, left) == 0.0
    assert empirical_ks(left, torch.tensor([3.0, 4.0, 5.0])) == 1.0
