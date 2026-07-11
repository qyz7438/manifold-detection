from __future__ import annotations

import math

import torch

from scripts.analyze_action_cache_lattice import summarize_records


def test_summarize_records_reports_sparse_discrete_utility() -> None:
    records = [
        {
            "observable_mask": torch.tensor([True, True]),
            "delta_u": torch.tensor([[0.0, -0.1, 1.2], [0.0, float("-inf"), 0.0]]),
        },
        {
            "observable_mask": torch.tensor([True]),
            "delta_u": torch.tensor([[0.0, -0.2, -0.3]]),
        },
    ]
    summary = summarize_records(records)
    assert summary["images"] == 2
    assert summary["candidate_count"] == 5
    assert summary["positive_candidate_count"] == 1
    assert summary["positive_image_count"] == 1
    assert summary["positive_image_rate"] == 0.5
    assert math.isclose(summary["positive_best_delta_u_mean"], 1.2, rel_tol=1e-6)
    assert summary["unique_rounded_delta_u"] == 5
