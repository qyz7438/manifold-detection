"""Compatibility entry point for the migrated action verifier trainer."""

from importlib import import_module
import sys


_module = import_module("spectral_detection_posttrain.trainers.detection.action_verifier_posttrain")

if __name__ == "__main__":
    _module.main()
else:
    sys.modules[__name__] = _module
