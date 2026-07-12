from __future__ import annotations

import torch

from scripts.analyze_dense_local_delta_families import (
    family_pair_breakdown,
    margin_breakdown,
)


def test_family_pair_breakdown_is_image_equal_and_keeps_same_family_pairs():
    prediction = torch.tensor([0.0, 1.0, 2.0, 1.0, 0.0])
    target = torch.tensor([0.0, 1.0, 2.0, 0.0, 1.0])
    image_ids = torch.tensor([1, 1, 1, 2, 2])
    families = ["a", "a", "b", "a", "b"]

    result = family_pair_breakdown(prediction, target, image_ids, families)

    assert result["a|a"]["images"] == 1
    assert result["a|a"]["candidate_pairs"] == 1
    assert result["a|a"]["eligible_pairs"] == 1
    assert result["a|a"]["zero_margin_pairs"] == 0
    assert result["a|a"]["image_equal_accuracy"] == 1.0
    assert result["a|b"]["images"] == 2
    assert result["a|b"]["candidate_pairs"] == 3
    assert result["a|b"]["eligible_pairs"] == 3
    assert result["a|b"]["image_equal_accuracy"] == 0.5
    assert result["a|b"]["candidate_weighted_accuracy"] == 2.0 / 3.0


def test_margin_breakdown_uses_fixed_protocol_bins_and_counts_ties():
    prediction = torch.tensor([0.0, 0.5, 2.0, 4.0])
    target = torch.tensor([0.0, 0.0, 0.02, 0.20])
    image_ids = torch.tensor([7, 7, 7, 7])

    result = margin_breakdown(
        prediction,
        target,
        image_ids,
        boundaries=[0.000001, 0.01, 0.05],
    )

    assert result["[0,1e-06]"]["candidate_pairs"] == 1
    assert result["[0,1e-06]"]["eligible_pairs"] == 0
    assert result["(0.01,0.05]"]["eligible_pairs"] == 2
    assert result["(0.01,0.05]"]["image_equal_accuracy"] == 1.0
    assert result["(0.05,inf)"]["eligible_pairs"] == 3
    assert result["(0.05,inf)"]["image_equal_accuracy"] == 1.0
    assert sum(item["candidate_pairs"] for item in result.values()) == 6
