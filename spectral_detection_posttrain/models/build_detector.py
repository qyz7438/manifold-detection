"""Compatibility shim for migrated detector construction helpers."""

from types import ModuleType
import sys

from spectral_detection_posttrain.core.models.build_detector import *  # noqa: F401,F403
from spectral_detection_posttrain.core.models.build_detector import build_detector as _build_detector


class _CallableBuildDetectorModule(ModuleType):
    def __call__(self, *args, **kwargs):
        return _build_detector(*args, **kwargs)


sys.modules[__name__].__class__ = _CallableBuildDetectorModule
