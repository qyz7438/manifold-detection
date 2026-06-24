"""Compatibility shim for migrated rollout helpers."""

from importlib import import_module
import sys


sys.modules[__name__] = import_module("spectral_detection_posttrain.trainers.detection.rollout")
