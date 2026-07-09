from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "train_action_benefit_energy.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("train_action_benefit_energy", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_threshold_parser_includes_primary_and_has_stable_names() -> None:
    module = _load_module()

    assert module.parse_thresholds(".01,-.01", 0.0) == [-0.01, 0.0, 0.01]
    assert module._threshold_name(-0.005) == "gate_tm0p005"


def test_binary_metrics_reward_perfect_ranking() -> None:
    module = _load_module()
    labels = torch.tensor([True, False, True, False])
    perfect = torch.tensor([0.9, 0.2, 0.8, 0.1])
    reversed_scores = -perfect

    assert module._binary_average_precision(labels, perfect) == pytest.approx(1.0)
    assert module._binary_roc_auc(labels, perfect) == pytest.approx(1.0)
    assert module._binary_roc_auc(labels, reversed_scores) == pytest.approx(0.0)
    assert module._binary_roc_auc(labels, torch.zeros_like(perfect)) == pytest.approx(0.5)


def test_value_permutation_stays_within_images() -> None:
    module = _load_module()
    values = torch.tensor([1.0, 2.0, 10.0, 20.0])
    image_indices = torch.tensor([0, 0, 1, 1])

    permuted = module._permute_values_within_images(
        values,
        image_indices,
        generator=torch.Generator().manual_seed(3),
    )

    assert sorted(permuted[:2].tolist()) == [1.0, 2.0]
    assert sorted(permuted[2:].tolist()) == [10.0, 20.0]
