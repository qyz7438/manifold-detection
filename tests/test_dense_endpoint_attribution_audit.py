from __future__ import annotations

import torch

from scripts.audit_dense_endpoint_attribution import apply_node_ablation, load_config


def test_attribution_config_forbids_outer_and_new_data():
    config = load_config()

    assert config["source"]["outer_cache_forbidden"] is True
    assert config["source"]["new_data_forbidden"] is True
    assert config["ablations"][-1] == "pair_branch_disabled"


def test_node_ablation_zeros_only_locked_feature_group():
    config = load_config()
    features = torch.arange(44, dtype=torch.float32).reshape(2, 22)

    ablated = apply_node_ablation(features, "all_geometry_zero", config)

    columns = config["feature_groups"]["all_geometry"]
    assert torch.count_nonzero(ablated[:, columns]) == 0
    keep = [index for index in range(22) if index not in columns]
    assert torch.equal(ablated[:, keep], features[:, keep])
