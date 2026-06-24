from __future__ import annotations

from copy import deepcopy
from typing import Any


SUPPORTED_MODEL_NAMES = {
    "fasterrcnn_mobilenet_v3_large_320_fpn",
    "fasterrcnn_resnet50_fpn",
}

AFM_TYPES_REQUIRING_ROI_CHANNELS = {
    "old",
    "identity",
    "mplseg",
    "mplseg_weak",
    "mplseg_mid",
    "mplseg_frozen",
    "mplseg_notune",
    "mplseg_mag_only",
    "mplseg_phase_only",
}


def resolve_model_name(model_cfg: dict[str, Any]) -> str:
    name = model_cfg.get("name")
    model_name = model_cfg.get("model_name")
    if name is not None and model_name is not None and str(name) != str(model_name):
        raise ValueError(f"model.name and model.model_name conflict: {name!r} != {model_name!r}")
    resolved = str(model_name or name or "fasterrcnn_mobilenet_v3_large_320_fpn")
    if resolved not in SUPPORTED_MODEL_NAMES:
        raise ValueError(f"Unknown model name: {resolved}. Supported: {sorted(SUPPORTED_MODEL_NAMES)}")
    return resolved


def infer_afm_channels(model_name: str) -> int:
    if model_name in SUPPORTED_MODEL_NAMES:
        return 256
    raise ValueError(f"Cannot infer AFM channels for unknown model: {model_name}")


def validate_experiment_config(config: dict[str, Any], formal: bool = True) -> dict[str, Any]:
    if not isinstance(config, dict):
        raise ValueError("Config must be a mapping")
    normalized = deepcopy(config)
    model_cfg = normalized.setdefault("model", {})
    if not isinstance(model_cfg, dict):
        raise ValueError("Config field 'model' must be a mapping")

    model_name = resolve_model_name(model_cfg)
    model_cfg["name"] = model_name
    model_cfg["model_name"] = model_name

    if formal and bool(model_cfg.get("allow_random_init_fallback", False)):
        raise ValueError("Formal experiments must not allow random-init fallback")

    afm_fpn = bool(model_cfg.get("afm_fpn", False))
    afm_type = str(model_cfg.get("afm_type", "none"))
    has_roi_afm = afm_type != "none" and not afm_fpn
    if has_roi_afm:
        expected_channels = infer_afm_channels(model_name)
        if "afm_channels" not in model_cfg or int(model_cfg.get("afm_channels", 0)) <= 0:
            model_cfg["afm_channels"] = expected_channels
        elif int(model_cfg["afm_channels"]) != expected_channels:
            raise ValueError(
                f"afm_channels={model_cfg['afm_channels']} does not match expected "
                f"ROI channels {expected_channels} for {model_name}"
            )
        if afm_type not in AFM_TYPES_REQUIRING_ROI_CHANNELS:
            raise ValueError(f"Unknown afm_type: {afm_type}")
    elif "afm_channels" in model_cfg:
        model_cfg["afm_channels"] = int(model_cfg.get("afm_channels", 0))

    return normalized
