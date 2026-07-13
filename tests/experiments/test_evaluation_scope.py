"""Contracts v2 evaluation-scope provenance tests (plan Task 5).

Proves that ``limit_train``, ``limit_val``, image counts, and
``evaluation_scope.kind`` are recorded in the resolved config, the run
metadata, and the final metrics — and that contracts v2 rejects ambiguous
scope declarations (``full_val`` with limits, ``smoke``/``limited`` without a
positive explicit limit, unknown kinds/keys). Missing scopes normalize to
``limited_unknown`` with a non-formal marker and an explicit warning.
"""

from __future__ import annotations

import copy
import json

import pytest

from scripts import round28_train_eval
from spectral_detection_posttrain.experiments.contracts import (
    CONTRACTS_VERSION,
    EvaluationScope,
    normalize_evaluation_scope,
    parse_evaluation_scope,
)
from spectral_detection_posttrain.experiments.metadata import (
    collect_experiment_metadata,
    evaluation_scope_provenance,
)
from spectral_detection_posttrain.experiments.schema import validate_experiment_config


def _base_config() -> dict:
    return {
        "seed": 42,
        "device": "cpu",
        "model": {
            "name": "fasterrcnn_mobilenet_v3_large_320_fpn",
            "pretrained": False,
            "allow_random_init_fallback": False,
            "num_classes": 2,
            "min_size": 320,
            "max_size": 320,
        },
        "train": {"batch_size": 1, "epochs": 1},
        "posttrain": {"batch_size": 1, "epochs": 1},
        "matching": {"iou_threshold": 0.5, "score_threshold": 0.05},
        "eval": {"batch_size": 1},
    }


# ---------------------------------------------------------------------------
# contracts v2 cross-field validation
# ---------------------------------------------------------------------------


def test_contracts_version_is_v2() -> None:
    assert CONTRACTS_VERSION == "v2"


def test_full_val_rejects_limit_val():
    with pytest.raises(ValueError, match="full_val"):
        EvaluationScope(kind="full_val", image_count=196, limit_val=32)


def test_full_val_rejects_limit_train() -> None:
    with pytest.raises(ValueError, match="full_val"):
        EvaluationScope(kind="full_val", image_count=4952, limit_train=100)


def test_full_val_without_limits_is_accepted() -> None:
    scope = EvaluationScope(kind="full_val", image_count=196)
    assert scope.limit_train is None
    assert scope.limit_val is None


@pytest.mark.parametrize("kind", ["smoke", "limited"])
def test_smoke_and_limited_require_positive_limit(kind: str) -> None:
    with pytest.raises(ValueError, match="limit"):
        EvaluationScope(kind=kind, image_count=None, limit_train=None, limit_val=None)
    with pytest.raises(ValueError, match="limit"):
        EvaluationScope(kind=kind, image_count=None, limit_train=0, limit_val=0)
    # One explicit positive limit is sufficient.
    assert EvaluationScope(kind=kind, limit_train=None, limit_val=4).limit_val == 4
    assert EvaluationScope(kind=kind, limit_train=4, limit_val=None).limit_train == 4


def test_bool_is_not_an_int_for_counts() -> None:
    with pytest.raises(ValueError):
        EvaluationScope(kind="smoke", image_count=None, limit_train=True, limit_val=None)
    with pytest.raises(ValueError):
        EvaluationScope(kind="full_val", image_count=False)


def test_negative_counts_rejected() -> None:
    with pytest.raises(ValueError):
        EvaluationScope(kind="full_val", image_count=-1)
    with pytest.raises(ValueError):
        EvaluationScope(kind="smoke", image_count=None, limit_train=-1, limit_val=None)


def test_unknown_scope_kind_rejected() -> None:
    with pytest.raises(ValueError, match="kind"):
        EvaluationScope(kind="bogus", image_count=None)
    with pytest.raises(ValueError, match="kind"):
        parse_evaluation_scope(
            {"kind": "bogus", "image_count": None, "limit_train": None, "limit_val": None}
        )
    with pytest.raises(ValueError, match="kind"):
        normalize_evaluation_scope(
            {"kind": "bogus"}, limit_train=None, limit_val=None, image_count=None
        )


def test_explicit_scope_dict_parses_strictly() -> None:
    with pytest.raises(ValueError, match="unknown"):
        normalize_evaluation_scope(
            {"kind": "smoke", "limit_val": 8, "bogus": 1},
            limit_train=None,
            limit_val=None,
            image_count=None,
        )
    scope, warnings = normalize_evaluation_scope(
        {"kind": "smoke", "limit_val": 8},
        limit_train=None,
        limit_val=None,
        image_count=8,
    )
    assert scope.kind == "smoke"
    assert scope.limit_val == 8
    assert scope.image_count == 8
    assert warnings == []


# ---------------------------------------------------------------------------
# Missing-scope normalization (backward compatibility)
# ---------------------------------------------------------------------------


def test_missing_scope_normalizes_to_limited_unknown_with_warning() -> None:
    scope, warnings = normalize_evaluation_scope(
        None, limit_train=None, limit_val=32, image_count=None
    )
    assert scope.kind == "limited_unknown"
    assert scope.limit_val == 32
    assert warnings, "missing scope must produce an explicit warning"
    assert any("limited_unknown" in warning for warning in warnings)
    assert any("non-formal" in warning for warning in warnings)


def test_config_without_scope_runs_non_formal() -> None:
    config = _base_config()
    normalized = validate_experiment_config(config, formal=True)
    assert normalized["evaluation_scope"]["kind"] == "limited_unknown"
    assert normalized["evaluation_scope_formal"] is False
    assert any("limited_unknown" in w for w in normalized["normalization_warnings"])


# ---------------------------------------------------------------------------
# Resolved config records scope provenance
# ---------------------------------------------------------------------------


def test_resolved_config_records_limits_image_count_and_kind() -> None:
    config = _base_config()
    config["evaluation_scope"] = {
        "kind": "limited",
        "image_count": 64,
        "limit_train": 48,
        "limit_val": 16,
    }
    normalized = validate_experiment_config(config)
    assert normalized["limit_train"] == 48
    assert normalized["limit_val"] == 16
    assert normalized["image_count"] == 64
    assert normalized["evaluation_scope"]["kind"] == "limited"
    assert normalized["evaluation_scope"]["image_count"] == 64
    assert normalized["evaluation_scope_formal"] is True
    assert normalized["normalization_warnings"] == []


def test_formal_full_val_rejects_limits_at_config_validation() -> None:
    config = _base_config()
    config["evaluation_scope"] = {
        "kind": "full_val",
        "image_count": 196,
        "limit_train": None,
        "limit_val": 32,
    }
    with pytest.raises(ValueError, match="full_val"):
        validate_experiment_config(config, formal=True)


def test_validation_is_idempotent_for_normalized_configs() -> None:
    config = _base_config()
    once = validate_experiment_config(config, formal=True)
    twice = validate_experiment_config(once, formal=True)
    assert twice["evaluation_scope"] == once["evaluation_scope"]
    assert twice["evaluation_scope_formal"] is False
    assert twice["normalization_warnings"] == once["normalization_warnings"]


# ---------------------------------------------------------------------------
# Metadata records scope provenance and warnings
# ---------------------------------------------------------------------------


def test_metadata_includes_scope_provenance_and_warnings() -> None:
    config = _base_config()
    normalized = validate_experiment_config(config, formal=True)
    metadata = collect_experiment_metadata(normalized)
    provenance = metadata["evaluation_scope"]
    assert provenance["kind"] == "limited_unknown"
    assert "limit_train" in provenance
    assert "limit_val" in provenance
    assert "image_count" in provenance
    assert provenance["formal"] is False
    assert any("limited_unknown" in w for w in metadata["normalization_warnings"])


def test_metadata_for_explicit_scope_marks_formal() -> None:
    config = _base_config()
    config["evaluation_scope"] = {
        "kind": "full_val",
        "image_count": 196,
        "limit_train": None,
        "limit_val": None,
    }
    normalized = validate_experiment_config(config, formal=True)
    metadata = collect_experiment_metadata(normalized)
    assert metadata["evaluation_scope"]["kind"] == "full_val"
    assert metadata["evaluation_scope"]["image_count"] == 196
    assert metadata["evaluation_scope"]["formal"] is True
    assert metadata["normalization_warnings"] == []


# ---------------------------------------------------------------------------
# Final metrics: additive provenance, existing keys byte-compatible
# ---------------------------------------------------------------------------


def test_metrics_gain_scope_provenance_without_changing_existing_keys() -> None:
    metrics = {
        "ap50": 0.61,
        "ap75": 0.43,
        "precision": 0.7,
        "recall": 0.5,
        "num_predictions": 12,
        "completed": True,
        "history": [{"epoch": 1, "val_ap50": 0.61}],
    }
    before = copy.deepcopy(metrics)
    before_json = json.dumps(metrics, indent=2, ensure_ascii=False)
    config = {
        "evaluation_scope": {
            "kind": "limited_unknown",
            "image_count": 3,
            "limit_train": None,
            "limit_val": 3,
        },
        "evaluation_scope_formal": False,
    }

    round28_train_eval._attach_evaluation_scope(metrics, config)

    added = set(metrics) - set(before)
    assert added == {"evaluation_scope"}
    for key, value in before.items():
        assert metrics[key] == value
    assert json.dumps(
        {key: metrics[key] for key in before}, indent=2, ensure_ascii=False
    ) == before_json
    provenance = metrics["evaluation_scope"]
    assert provenance["kind"] == "limited_unknown"
    assert provenance["limit_val"] == 3
    assert provenance["image_count"] == 3
    assert provenance["formal"] is False


def test_evaluation_scope_provenance_helper_reads_normalized_config() -> None:
    config = _base_config()
    config["evaluation_scope"] = {
        "kind": "smoke",
        "image_count": 32,
        "limit_train": 16,
        "limit_val": 16,
    }
    normalized = validate_experiment_config(config, formal=True)
    provenance = evaluation_scope_provenance(normalized)
    assert provenance == {
        "kind": "smoke",
        "image_count": 32,
        "limit_train": 16,
        "limit_val": 16,
        "formal": True,
    }
