"""Compatibility shim: selection policies moved to ``energy_transport.policy``.

The implementations now live in
``spectral_detection_posttrain.methods.energy_transport.policy.post_nms_suppress``.
This module only re-exports the public names so existing flat imports keep
working during the package-boundary refactor (plan Tasks 13/14).
"""

from spectral_detection_posttrain.methods.energy_transport.policy.post_nms_suppress import (
    PostNMSSuppression,
    PostNMSSuppressPolicyHead,
    build_post_nms_detection_features,
    select_post_nms_suppression,
    suppress_detection,
)

__all__ = [
    "PostNMSSuppression",
    "PostNMSSuppressPolicyHead",
    "build_post_nms_detection_features",
    "select_post_nms_suppression",
    "suppress_detection",
]
