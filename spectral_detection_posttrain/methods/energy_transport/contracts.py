"""Compatibility shim: typed contracts moved to ``energy_transport.action``.

The implementations now live in
``spectral_detection_posttrain.methods.energy_transport.action.contracts``.
This module only re-exports the public names so existing flat imports keep
working during the package-boundary refactor (plan Tasks 13/14).
"""

from spectral_detection_posttrain.methods.energy_transport.action.contracts import (
    ActionOutcome,
    ConstraintConfig,
    PreferenceBatch,
    ROIActionState,
)

__all__ = [
    "ActionOutcome",
    "ConstraintConfig",
    "PreferenceBatch",
    "ROIActionState",
]
