"""Old/new import-path parity for the energy_transport action core (plan T13).

Refactor plan Task 13 moves the five action-core modules
(``actions``, ``contracts``, ``operators``, ``preferences``,
``geometric_constraints``) from the flat
``spectral_detection_posttrain.methods.energy_transport`` package into the new
``energy_transport.action`` subpackage, leaving the flat modules as pure
forwarding shims.

This test pins the migration contract:

1. Every public symbol resolves to the *same object* through the old flat
   path and the new ``energy_transport.action.*`` path (identity parity).
2. Callables keep identical signatures, and seeded CPU inputs produce
   bit-identical outputs through both paths.
3. The flat shims contain imports and ``__all__`` only — no function bodies,
   no classes, no logic.
4. The hash-locked synthetic checkpoint fixture
   (``tests/fixtures/checkpoints/energy_transport_checkpoint_manifest.json``)
   strict-loads through both import paths with identical state-dict keys,
   tensors, and deterministic outputs. The fixture files are SYNTHETIC parity
   fixtures created in T13, not historical experiment weights.
"""

from __future__ import annotations

import ast
import hashlib
import importlib
import inspect
import json
from pathlib import Path

import pytest
import torch

PACKAGE_ROOT = "spectral_detection_posttrain.methods.energy_transport"
REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = REPO_ROOT / "spectral_detection_posttrain" / "methods" / "energy_transport"
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "checkpoints"
MANIFEST_PATH = FIXTURE_DIR / "energy_transport_checkpoint_manifest.json"

# Public surface of each action-core module. The flat module and the new
# action submodule must expose exactly these names, pointing to the same
# objects. Ordered to match the package facade imports.
ACTION_PUBLIC_SYMBOLS: dict[str, tuple[str, ...]] = {
    "actions": (
        "ActionLocalTransportHead",
        "ROITransportActions",
        "apply_bounded_score_delta",
        "rescue_budget_loss",
        "summarize_score_actions",
        "threshold_preservation_loss",
        "transport_action_energy",
    ),
    "contracts": (
        "ActionOutcome",
        "ConstraintConfig",
        "PreferenceBatch",
        "ROIActionState",
    ),
    "operators": (
        "apply_box_delta",
        "clip_boxes_to_image",
    ),
    "preferences": (
        "build_top_bottom_preferences",
    ),
    "geometric_constraints": (
        "GeometricConstraintConfig",
        "SAMPLE_TP",
        "SAMPLE_CLS_ERR",
        "SAMPLE_LOC_ERR",
        "SAMPLE_PURE_BG",
        "SAMPLE_AMBIGUOUS_BG",
        "bbox_aware_action_loss",
        "bbox_aware_direct_loss",
        "bbox_aware_linearization_loss",
        "bg_proto_action_loss",
        "classify_error_modes",
        "cls_err_action_loss",
        "compute_action_local_prototypes",
        "feat_preserve_action_loss",
        "fg_bg_sep_action_loss",
        "geometric_transport_loss",
        "intra_tp_action_loss",
        "loc_err_action_loss",
    ),
}

_ALL_SYMBOLS = [
    (module, symbol)
    for module, symbols in ACTION_PUBLIC_SYMBOLS.items()
    for symbol in symbols
]


def _flat_module(stem: str):
    return importlib.import_module(f"{PACKAGE_ROOT}.{stem}")


def _action_module(stem: str):
    return importlib.import_module(f"{PACKAGE_ROOT}.action.{stem}")


# ---------------------------------------------------------------------------
# 1. Identity and signature parity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "stem,symbol",
    _ALL_SYMBOLS,
    ids=[f"{stem}.{symbol}" for stem, symbol in _ALL_SYMBOLS],
)
def test_public_symbol_identity_between_old_and_new_paths(stem: str, symbol: str):
    flat_obj = getattr(_flat_module(stem), symbol)
    new_obj = getattr(_action_module(stem), symbol)
    assert flat_obj is new_obj, (
        f"{stem}.{symbol}: flat import and action.* import resolve to "
        "different objects; the flat module must be a pure forwarding shim"
    )


@pytest.mark.parametrize(
    "stem,symbol",
    [(m, s) for m, s in _ALL_SYMBOLS if s.islower()],
    ids=[f"{stem}.{symbol}" for stem, symbol in _ALL_SYMBOLS if symbol.islower()],
)
def test_callable_signature_parity(stem: str, symbol: str):
    flat_obj = getattr(_flat_module(stem), symbol)
    new_obj = getattr(_action_module(stem), symbol)
    assert str(inspect.signature(flat_obj)) == str(inspect.signature(new_obj))


def test_facade_reexports_match_action_subpackage():
    """Symbols re-exported by the package facade are the action.* objects."""
    facade = importlib.import_module(PACKAGE_ROOT)
    for stem, symbols in ACTION_PUBLIC_SYMBOLS.items():
        new_module = _action_module(stem)
        for symbol in symbols:
            if symbol in facade.__all__:
                assert getattr(facade, symbol) is getattr(new_module, symbol), (
                    f"facade re-export {symbol} does not match action.{stem}"
                )


# ---------------------------------------------------------------------------
# 2. Flat shims are pure forwarding modules (imports + __all__ only)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("stem", sorted(ACTION_PUBLIC_SYMBOLS))
def test_flat_shim_contains_no_function_bodies(stem: str):
    path = PACKAGE_DIR / f"{stem}.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for statement in tree.body:
        if isinstance(statement, (ast.Import, ast.ImportFrom)):
            continue
        if isinstance(statement, ast.Expr) and isinstance(
            statement.value, ast.Constant
        ):
            continue  # module docstring
        if isinstance(statement, ast.Assign):
            targets = {
                target.id for target in statement.targets if isinstance(target, ast.Name)
            }
            if targets == {"__all__"}:
                continue
        raise AssertionError(
            f"{stem}.py is not a pure shim: unexpected top-level statement "
            f"{type(statement).__name__} at line {statement.lineno}"
        )


@pytest.mark.parametrize("stem", sorted(ACTION_PUBLIC_SYMBOLS))
def test_flat_shim_all_matches_public_surface(stem: str):
    flat = _flat_module(stem)
    assert set(flat.__all__) == set(ACTION_PUBLIC_SYMBOLS[stem]), (
        f"{stem}.py __all__ drifted from the documented public surface: "
        f"missing {sorted(set(ACTION_PUBLIC_SYMBOLS[stem]) - set(flat.__all__))}, "
        f"extra {sorted(set(flat.__all__) - set(ACTION_PUBLIC_SYMBOLS[stem]))}"
    )


# ---------------------------------------------------------------------------
# 3. Deterministic CPU output parity through both import paths
# ---------------------------------------------------------------------------


def _tensor(seed: int, *shape: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=generator)


def test_action_head_forward_parity():
    head_cls_flat = getattr(_flat_module("actions"), "ActionLocalTransportHead")
    head_cls_new = getattr(_action_module("actions"), "ActionLocalTransportHead")
    features = _tensor(7, 6, 8)

    torch.manual_seed(42)
    head_flat = head_cls_flat(feature_dim=8, hidden_dim=8, residual_scale=0.05).eval()
    torch.manual_seed(42)
    head_new = head_cls_new(feature_dim=8, hidden_dim=8, residual_scale=0.05).eval()

    with torch.no_grad():
        out_flat = head_flat(features)
        out_new = head_new(features)
    for field in ("feature_delta", "score_delta", "box_delta", "keep_logit"):
        assert torch.equal(getattr(out_flat, field), getattr(out_new, field)), field


def test_score_and_energy_function_parity():
    actions_flat = _flat_module("actions")
    actions_new = _action_module("actions")

    scores = torch.sigmoid(_tensor(1, 6))
    raw_delta = _tensor(2, 6)
    assert torch.equal(
        actions_flat.apply_bounded_score_delta(scores, raw_delta, 0.2),
        actions_new.apply_bounded_score_delta(scores, raw_delta, 0.2),
    )

    feature_delta = _tensor(3, 6, 8)
    box_delta = _tensor(4, 6, 4)
    assert torch.equal(
        actions_flat.transport_action_energy(
            feature_delta, score_delta=raw_delta, box_delta=box_delta
        ),
        actions_new.transport_action_energy(
            feature_delta, score_delta=raw_delta, box_delta=box_delta
        ),
    )

    low_quality_mask = torch.tensor([True, False, True, False, True, False])
    assert torch.equal(
        actions_flat.threshold_preservation_loss(
            scores, raw_delta, low_quality_mask=low_quality_mask, threshold=0.5
        ),
        actions_new.threshold_preservation_loss(
            scores, raw_delta, low_quality_mask=low_quality_mask, threshold=0.5
        ),
    )

    candidate_mask = torch.tensor([True, True, True, False, True, False])
    assert torch.equal(
        actions_flat.rescue_budget_loss(
            scores,
            raw_delta,
            candidate_mask=candidate_mask,
            threshold=0.5,
            max_rescues=1.0,
        ),
        actions_new.rescue_budget_loss(
            scores,
            raw_delta,
            candidate_mask=candidate_mask,
            threshold=0.5,
            max_rescues=1.0,
        ),
    )

    assert (
        actions_flat.summarize_score_actions(scores, raw_delta, threshold=0.5)
        == actions_new.summarize_score_actions(scores, raw_delta, threshold=0.5)
    )


def test_box_operator_parity():
    operators_flat = _flat_module("operators")
    operators_new = _action_module("operators")

    boxes = torch.tensor(
        [[10.0, 10.0, 30.0, 40.0], [0.0, 0.0, 50.0, 60.0], [5.0, 5.0, 8.0, 9.0]]
    )
    deltas = _tensor(5, 3, 4) * 0.1
    assert torch.equal(
        operators_flat.apply_box_delta(boxes, deltas, image_size=(64, 64)),
        operators_new.apply_box_delta(boxes, deltas, image_size=(64, 64)),
    )
    assert torch.equal(
        operators_flat.clip_boxes_to_image(boxes, (64, 64)),
        operators_new.clip_boxes_to_image(boxes, (64, 64)),
    )


def test_preference_builder_parity():
    preferences_flat = _flat_module("preferences")
    preferences_new = _action_module("preferences")

    quality = _tensor(6, 7)
    ious = torch.rand(7, generator=torch.Generator().manual_seed(8))
    group_ids = torch.tensor([0, 0, 1, 1, 1, 2, 2])

    batch_flat = preferences_flat.build_top_bottom_preferences(
        quality, ious, group_ids, min_quality_margin=0.0
    )
    batch_new = preferences_new.build_top_bottom_preferences(
        quality, ious, group_ids, min_quality_margin=0.0
    )
    for field in (
        "chosen_indices",
        "rejected_indices",
        "quality_gap",
        "iou_gap",
        "valid_mask",
    ):
        assert torch.equal(getattr(batch_flat, field), getattr(batch_new, field)), field


def _make_roi_state(contracts_module):
    batch, dim = 6, 8
    features = _tensor(10, batch, dim)
    boxes = torch.tensor(
        [
            [10.0, 10.0, 30.0, 40.0],
            [0.0, 0.0, 50.0, 60.0],
            [5.0, 5.0, 8.0, 9.0],
            [12.0, 14.0, 20.0, 22.0],
            [1.0, 2.0, 9.0, 12.0],
            [30.0, 30.0, 45.0, 48.0],
        ]
    )
    return contracts_module.ROIActionState(
        features=features,
        boxes=boxes,
        scores=torch.sigmoid(_tensor(11, batch)),
        labels=torch.tensor([1, 2, 1, 0, 2, 1]),
        image_indices=torch.zeros(batch, dtype=torch.long),
        proposal_indices=torch.arange(batch),
        ious=torch.tensor([0.9, 0.75, 0.4, 0.2, 0.1, 0.55]),
    )


def _make_actions(actions_module, batch: int = 6, dim: int = 8):
    return actions_module.ROITransportActions(
        feature_delta=_tensor(12, batch, dim) * 0.01,
        score_delta=_tensor(13, batch) * 0.01,
        box_delta=_tensor(14, batch, 4) * 0.01,
        keep_logit=_tensor(15, batch),
    )


def test_geometric_transport_loss_parity():
    geo_flat = _flat_module("geometric_constraints")
    geo_new = _action_module("geometric_constraints")
    contracts_flat = _flat_module("contracts")
    contracts_new = _action_module("contracts")
    actions_flat = _flat_module("actions")
    actions_new = _action_module("actions")

    state_flat = _make_roi_state(contracts_flat)
    state_new = _make_roi_state(contracts_new)
    actions_flat_obj = _make_actions(actions_flat)
    actions_new_obj = _make_actions(actions_new)

    config_flat = geo_flat.GeometricConstraintConfig(lambda_intra_tp=0.5)
    config_new = geo_new.GeometricConstraintConfig(lambda_intra_tp=0.5)

    losses_flat = geo_flat.geometric_transport_loss(
        state_flat, actions_flat_obj, 3, config_flat
    )
    losses_new = geo_new.geometric_transport_loss(
        state_new, actions_new_obj, 3, config_new
    )
    assert losses_flat.keys() == losses_new.keys()
    for key in losses_flat:
        assert torch.equal(losses_flat[key], losses_new[key]), key


# ---------------------------------------------------------------------------
# 4. Hash-locked synthetic checkpoint fixture: strict-load through both paths
# ---------------------------------------------------------------------------


def _load_manifest() -> dict:
    assert MANIFEST_PATH.exists(), (
        f"checkpoint manifest missing: {MANIFEST_PATH} "
        "(synthetic parity fixtures are created in plan Task 13 step 2)"
    )
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def test_manifest_declares_synthetic_provenance():
    manifest = _load_manifest()
    provenance = manifest.get("provenance", "")
    assert "SYNTHETIC" in provenance
    assert "not historical" in provenance.lower()


def test_fixture_files_match_manifest_sha256():
    manifest = _load_manifest()
    for entry in manifest["fixtures"]:
        path = FIXTURE_DIR / entry["file"]
        assert path.exists(), f"fixture file missing: {path}"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == entry["sha256"], (
            f"fixture {entry['file']} sha256 mismatch: manifest "
            f"{entry['sha256']} vs actual {digest} (hash-locked fixture)"
        )


def test_fixture_state_dict_strict_loads_through_both_paths():
    manifest = _load_manifest()
    for entry in manifest["fixtures"]:
        payload = torch.load(
            FIXTURE_DIR / entry["file"], map_location="cpu", weights_only=True
        )
        module_stem = entry["module"].rsplit(".", 1)[-1]
        init_kwargs = entry["init_kwargs"]

        heads = []
        for loader in (_flat_module, _action_module):
            cls = getattr(loader(module_stem), entry["class"])
            module = cls(**init_kwargs).eval()
            module.load_state_dict(payload, strict=True)
            heads.append(module)

        flat_head, new_head = heads
        assert list(flat_head.state_dict()) == list(payload), (
            f"state-dict key drift for {entry['class']} (flat path)"
        )
        assert list(new_head.state_dict()) == list(payload), (
            f"state-dict key drift for {entry['class']} (action.* path)"
        )
        for key, tensor in payload.items():
            assert torch.equal(flat_head.state_dict()[key], tensor), key
            assert torch.equal(new_head.state_dict()[key], tensor), key

        features = _tensor(99, 4, init_kwargs["feature_dim"])
        with torch.no_grad():
            out_flat = flat_head(features)
            out_new = new_head(features)
        for field in ("feature_delta", "score_delta", "box_delta", "keep_logit"):
            assert torch.equal(getattr(out_flat, field), getattr(out_new, field)), (
                f"deterministic output drift on field {field} for {entry['class']}"
            )


def test_manifest_covers_every_action_module():
    """Each action module is either fixture-backed or explicitly skipped."""
    manifest = _load_manifest()
    covered = {entry["module"].rsplit(".", 1)[-1] for entry in manifest["fixtures"]}
    skipped = {entry["module"].rsplit(".", 1)[-1] for entry in manifest["skipped"]}
    assert covered | skipped == set(ACTION_PUBLIC_SYMBOLS), (
        f"manifest coverage drift: uncovered "
        f"{sorted(set(ACTION_PUBLIC_SYMBOLS) - covered - skipped)}"
    )
    assert not (covered & skipped), "module listed as both fixture and skipped"
    for entry in manifest["skipped"]:
        assert entry.get("reason"), (
            f"skipped module {entry['module']} must document why no fixture exists"
        )
