from __future__ import annotations

import pytest

from spectral_detection_posttrain.experiments.schema import validate_experiment_config
from spectral_detection_posttrain.models import build_detector


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


def test_unknown_model_name_is_rejected_by_schema() -> None:
    config = _base_config()
    config["model"]["name"] = "not_a_detector"

    with pytest.raises(ValueError, match="Unknown model"):
        validate_experiment_config(config)


def test_unknown_model_name_is_rejected_by_builder() -> None:
    config = _base_config()
    del config["model"]["name"]
    config["model"]["model_name"] = "not_a_detector"

    with pytest.raises(ValueError, match="Unknown model"):
        build_detector(config)


def test_conflicting_model_name_aliases_are_rejected() -> None:
    config = _base_config()
    config["model"]["model_name"] = "fasterrcnn_resnet50_fpn"

    with pytest.raises(ValueError, match="conflict"):
        validate_experiment_config(config)


def test_formal_config_forbids_random_init_fallback() -> None:
    config = _base_config()
    config["model"]["allow_random_init_fallback"] = True

    with pytest.raises(ValueError, match="random-init fallback"):
        validate_experiment_config(config, formal=True)


def test_non_formal_config_can_allow_random_init_fallback() -> None:
    config = _base_config()
    config["model"]["allow_random_init_fallback"] = True

    normalized = validate_experiment_config(config, formal=False)

    assert normalized["model"]["allow_random_init_fallback"] is True


def test_afm_channels_are_inferred_for_supported_roi_afm() -> None:
    config = _base_config()
    config["model"]["afm_type"] = "mplseg_mid"

    normalized = validate_experiment_config(config)

    assert normalized["model"]["afm_channels"] == 256


def test_wrong_afm_channels_are_rejected_for_supported_detector() -> None:
    config = _base_config()
    config["model"].update({"afm_type": "mplseg_mid", "afm_channels": 128})

    with pytest.raises(ValueError, match="afm_channels"):
        validate_experiment_config(config)
