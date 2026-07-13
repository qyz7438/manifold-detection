"""Compatibility shim: preference helpers moved to ``energy_transport.action``.

The implementation now lives in
``spectral_detection_posttrain.methods.energy_transport.action.preferences``.
This module only re-exports the public names so existing flat imports keep
working during the package-boundary refactor (plan Tasks 13/14).
"""

from spectral_detection_posttrain.methods.energy_transport.action.preferences import (
    build_top_bottom_preferences,
)

__all__ = [
    "build_top_bottom_preferences",
]
