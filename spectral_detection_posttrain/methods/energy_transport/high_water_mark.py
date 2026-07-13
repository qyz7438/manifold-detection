"""Compatibility shim: transport diagnostics moved to ``energy_transport.diagnostics``.

The implementations now live in
``spectral_detection_posttrain.methods.energy_transport.diagnostics.high_water_mark``.
This module only re-exports the public names so existing flat imports keep
working during the package-boundary refactor (plan Tasks 13/14).
"""

from spectral_detection_posttrain.methods.energy_transport.diagnostics.high_water_mark import (
    HighWaterMarkLossConfig,
    HighWaterMarkModuleSnapshot,
    should_update_high_water_mark,
    capture_high_water_mark_module,
    load_high_water_mark_module,
    ap75_boundary_weights,
    high_water_mark_action_loss,
    stop_high_water_mark_loss,
)

__all__ = [
    "HighWaterMarkLossConfig",
    "HighWaterMarkModuleSnapshot",
    "should_update_high_water_mark",
    "capture_high_water_mark_module",
    "load_high_water_mark_module",
    "ap75_boundary_weights",
    "high_water_mark_action_loss",
    "stop_high_water_mark_loss",
]
