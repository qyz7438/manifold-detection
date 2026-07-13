"""Compatibility shim import coverage (refactor Task 17).

Every tracked compatibility namespace listed in AGENTS.md —
``spectral_detection_posttrain/spectral/``, ``/models/``, ``/rlvr/``, ``/train/``,
and ``/matching/`` — must remain importable.  Sibling shims must not be
affected when one shim fails because an optional legacy dependency or its
canonical target is missing in this environment.
"""

from __future__ import annotations

import importlib
import re
from types import ModuleType

import pytest

REPO_PKG = "spectral_detection_posttrain"

# Compatibility shim packages and, where meaningful, the submodules under them.
# ``__init__.py`` re-export shims are covered by importing the package and
# checking ``__all__``; per-file module-swap shims are covered by the submodule
# list.
SHIM_PACKAGES: dict[str, list[str] | None] = {
    "spectral": [
        "fft_features",
        "radial_profile",
        "rlvr_reward",
        "roi_crop",
        "roi_spectral_dataset",
        "round211_spectral_gate",
        "spectral_reward",
    ],
    "models": [
        "bbox_adapter",
        "build_detector",
        "micro_afm",
        "spectral_quality_head",
    ],
    "rlvr": [
        "action_verifier",
        "confidence_rescue",
        "detection_verifier",
        "roi_policy_loss",
        "round211_spatial_verifier",
    ],
    "train": [
        "action_verifier_posttrain",
        "posttrain_reward_weighted",
        "posttrain_rlvr",
        "rollout",
        "train_baseline",
        "train_quality_head",
    ],
    "matching": [
        "box_iou",
        "pred_gt_matcher",
    ],
}

# Module-swap shims that are expected to fail in this environment because the
# canonical target they delegate to has been removed.  The failure must be a
# targeted ``ModuleNotFoundError`` naming the missing target, not an import
# cascade that breaks the parent package or sibling shims.
EXPECTED_MISSING_TARGETS = {
    f"{REPO_PKG}.train.posttrain_reward_weighted",
    f"{REPO_PKG}.train.train_quality_head",
}


class ShimImportError(Exception):
    """A shim failed to import with an unexpected error."""


def _import_one(dotted: str) -> ModuleType:
    return importlib.import_module(dotted)


def _canon_source_for(dotted: str) -> str | None:
    """Return the canonical module a shim star-imports from, if detectable."""
    module = _import_one(dotted)
    source = getattr(module, "__doc__", "") or ""
    m = re.search(
        r"(?:new code should use|migrated to|Compatibility shim for)(?: the)?\s*`?([\w.]+)",
        source,
    )
    if m:
        return m.group(1)
    return None


@pytest.mark.parametrize(
    "package_name",
    sorted(SHIM_PACKAGES),
    ids=sorted(SHIM_PACKAGES),
)
def test_shim_package_imports_and_reexports(package_name: str):
    """Top-level ``spectral_detection_posttrain.<shim>`` resolves and its
    ``__all__`` symbols (when declared) are present."""
    dotted = f"{REPO_PKG}.{package_name}"
    module = _import_one(dotted)
    names = getattr(module, "__all__", None)
    if names is None:
        pytest.skip(f"{dotted} does not declare __all__")
    for name in names:
        assert hasattr(module, name), f"{dotted}.__all__ declares {name} but it is missing"


@pytest.mark.parametrize(
    "dotted",
    [
        f"{REPO_PKG}.{package}.{submodule}"
        for package, submodules in SHIM_PACKAGES.items()
        if submodules
        for submodule in submodules
    ],
)
def test_shim_submodule_imports(dotted: str):
    """Each submodule under a shim namespace imports without unexpected errors.

    Missing canonical targets produce a targeted ``ModuleNotFoundError`` and do
    not break the parent package.
    """
    try:
        module = _import_one(dotted)
    except ModuleNotFoundError as exc:
        # Missing canonical target is an expected, bounded failure.
        if dotted in EXPECTED_MISSING_TARGETS:
            pytest.xfail(f"{dotted} target is missing in this environment: {exc}")
        # Missing optional legacy dependency (e.g. sklearn) is allowed as long
        # as the message names the dependency.
        msg = str(exc)
        if "sklearn" in msg or "scikit" in msg:
            pytest.xfail(f"{dotted} requires optional legacy dependency: {exc}")
        raise ShimImportError(f"{dotted}: unexpected ModuleNotFoundError: {exc}") from exc

    # For module-swap shims, assert the swapped-in module is the canonical one.
    if "train." in dotted and dotted not in EXPECTED_MISSING_TARGETS:
        canonical_target = dotted.replace(".train.", ".trainers.detection.")
        assert module is importlib.import_module(canonical_target), (
            f"{dotted} should be a module-swap shim for {canonical_target}"
        )


def test_expected_missing_shims_fail_cleanly():
    """The two known missing trainer targets raise before any side effects."""
    for dotted in EXPECTED_MISSING_TARGETS:
        with pytest.raises(ModuleNotFoundError):
            _import_one(dotted)


def test_sibling_shims_survive_one_missing_target():
    """A missing train shim must not prevent the package and its siblings from
    resolving."""
    parent = _import_one(f"{REPO_PKG}.train")
    # Posttrain_rlvr and rollout still import through the train namespace.
    assert hasattr(parent, "posttrain_rlvr")
    assert hasattr(parent, "rollout")
    # The missing entries are absent rather than half-initialized.
    assert "posttrain_reward_weighted" not in parent.__dict__
    assert "train_quality_head" not in parent.__dict__
