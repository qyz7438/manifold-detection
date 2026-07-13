"""Compatibility shim: box operators moved to ``energy_transport.action``.

The implementations now live in
``spectral_detection_posttrain.methods.energy_transport.action.operators``.
This module only re-exports the public names so existing flat imports keep
working during the package-boundary refactor (plan Tasks 13/14).
"""

from spectral_detection_posttrain.methods.energy_transport.action.operators import (
    apply_box_delta,
    clip_boxes_to_image,
)

__all__ = [
    "apply_box_delta",
    "clip_boxes_to_image",
]
