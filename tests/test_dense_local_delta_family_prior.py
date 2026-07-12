from __future__ import annotations

import torch

from scripts.analyze_dense_local_delta_family_prior import (
    fit_family_means,
    predict_family_means,
)


def test_family_means_use_fit_targets_only_and_fall_back_to_global_mean():
    families = ["a", "a", "b"]
    targets = torch.tensor([1.0, 3.0, -2.0])

    fitted = fit_family_means(families, targets)
    prediction = predict_family_means(["b", "a", "missing"], fitted)

    assert fitted["family_means"] == {"a": 2.0, "b": -2.0}
    assert fitted["global_mean"] == 2.0 / 3.0
    assert torch.allclose(prediction, torch.tensor([-2.0, 2.0, 2.0 / 3.0]))


def test_family_mean_fit_rejects_mismatched_rows():
    try:
        fit_family_means(["a"], torch.tensor([1.0, 2.0]))
    except ValueError as error:
        assert "match" in str(error)
    else:
        raise AssertionError("expected mismatched rows to fail")
