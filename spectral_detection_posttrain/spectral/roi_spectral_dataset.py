"""Compatibility entry point for migrated ROI spectral candidate caching."""

from importlib import import_module
import sys


_module = import_module("spectral_detection_posttrain.signals.fft.roi_spectral_dataset")

if __name__ == "__main__":
    _module.main()
else:
    sys.modules[__name__] = _module
