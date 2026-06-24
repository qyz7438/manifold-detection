"""Compatibility entry point for the migrated quality-head trainer."""

from importlib import import_module
import sys


_module = import_module("spectral_detection_posttrain.trainers.detection.train_quality_head")

if __name__ == "__main__":
    _module.main()
else:
    sys.modules[__name__] = _module
