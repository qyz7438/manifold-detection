"""Compatibility entry point for the migrated reward-weighted trainer."""

from importlib import import_module
import sys


_module = import_module("spectral_detection_posttrain.trainers.detection.posttrain_reward_weighted")

if __name__ == "__main__":
    _module.main()
else:
    sys.modules[__name__] = _module
