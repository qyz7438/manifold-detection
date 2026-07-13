"""Old/new import-path parity for energy_transport shims (plan T13/T14).

Refactor plan Task 13 moves the action-core modules (``actions``,
``contracts``, ``operators``, ``preferences``, ``geometric_constraints``) into
``energy_transport.action`` and the native detector-coupled modules
(``native_contract``, ``native_topology``, ``candidate_energy``,
``benefit_energy``) into ``energy_transport.native``; Task 14 phase 1 moves
the policy selection modules (``search``, ``set_search``, ``set_policy``,
``global_top1``, ``listwise_noop``, ``joint_delta_u``, ``adaptive_consensus``,
``post_nms_suppress``) into ``energy_transport.policy``. The flat modules
remain as pure forwarding shims.

This test pins the migration contract for all three subpackages:

1. Every public symbol resolves to the *same object* through the old flat
   path and the new ``energy_transport.action.*`` / ``energy_transport.native.*``
   / ``energy_transport.policy.*`` path (identity parity).
2. Callables keep identical signatures, and seeded CPU inputs produce
   bit-identical outputs through both paths — including strict zero-action
   native parity helpers, class expansion, BoxCoder decode, small-box
   filtering, class-wise NMS, ordering, top-K behavior, policy search
   tie-breaking and budgets, seeded policy statistics, policy head forwards,
   and listwise no-op calibration.
3. The flat shims contain imports and ``__all__`` only — no function bodies,
   no classes, no logic.
4. The hash-locked synthetic checkpoint fixture
   (``tests/fixtures/checkpoints/energy_transport_checkpoint_manifest.json``)
   strict-loads through both import paths with identical state-dict keys,
   tensors, and deterministic outputs. The fixture files are SYNTHETIC parity
   fixtures created in T13/T14, not historical experiment weights.
"""

from __future__ import annotations

import ast
import dataclasses
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

# Public surface of each native detector-coupled module. Same contract as
# ACTION_PUBLIC_SYMBOLS. ``native_contract`` is not re-exported by the package
# facade; consumers import the flat shim directly.
NATIVE_PUBLIC_SYMBOLS: dict[str, tuple[str, ...]] = {
    "native_contract": (
        "DetectorNativeCandidates",
        "build_native_c1_deltas",
        "build_detector_native_candidates",
        "validate_strict_parity_artifact",
        "evaluate_native_contract_gates",
    ),
    "native_topology": (
        "NATIVE_TOPOLOGY_FEATURE_NAMES",
        "native_action_nms_topology",
    ),
    "candidate_energy": (
        "CandidateEnergyLossConfig",
        "CandidateGainLossConfig",
        "CandidateQualityTargets",
        "ContextOnlyCandidateEnergyHead",
        "SpatialCandidateEnergyHead",
        "build_candidate_quality_targets",
        "build_symmetric_box_candidates",
        "candidate_action_energy_loss",
        "candidate_action_gain_loss",
        "select_min_energy_box_actions",
    ),
    "benefit_energy": (
        "ActionBenefitEnergyHead",
        "ActionBenefitTargets",
        "BenefitEnergyLossConfig",
        "action_benefit_energy_loss",
        "apply_action_benefit_gate",
        "build_action_benefit_targets",
    ),
}

# Public surface of each policy selection module (Task 14 phase 1).
# ``SetEvaluator`` is a public type alias used in set_search signatures.
# ``listwise_noop`` is not re-exported by the package facade; consumers import
# the flat shim directly.
POLICY_PUBLIC_SYMBOLS: dict[str, tuple[str, ...]] = {
    "search": (
        "ActionSearchConfig",
        "ScoreActionSearchResult",
        "apply_score_action_to_prediction",
        "select_min_energy_score_actions",
    ),
    "set_search": (
        "ActionCandidate",
        "PairedBootstrapSummary",
        "SetEvaluator",
        "SetOutcome",
        "SetSearchResult",
        "beam_search",
        "deterministic_delta_permutation",
        "greedy_positive_marginal_selection",
        "local_top_b",
        "paired_bootstrap_summary",
        "set_outcome_from_prediction",
    ),
    "set_policy": (
        "NMSAwareSetPolicyHead",
        "SetPolicyLossConfig",
        "SetPolicyOutput",
        "SetPolicySelection",
        "class_aware_conflict_statistics",
        "select_set_policy_actions",
        "set_policy_loss",
    ),
    "global_top1": (
        "FlattenedGlobalLogits",
        "GlobalTop1Output",
        "GlobalTop1PolicyHead",
        "GlobalTop1Selection",
        "GlobalTop1Target",
        "SetContextGlobalTop1PolicyHead",
        "ActionTopologyGlobalTop1PolicyHead",
        "AdaptiveConsensusGlobalTop1PolicyHead",
        "NativeActionTopologyGlobalTop1PolicyHead",
        "action_conditioned_nms_topology",
        "build_global_top1_target",
        "flatten_observable_action_logits",
        "global_top1_balanced_margin_loss",
        "global_top1_loss",
        "select_global_top1_action",
        "select_adaptive_consensus_action",
    ),
    "listwise_noop": (
        "ListwiseBatch",
        "LinearListwiseModel",
        "build_listwise_batch",
        "calibrate_noop_margin",
        "evaluate_listwise_gates",
        "fit_listwise_model",
        "listwise_decision_metrics",
        "paired_gain_stats",
        "predict_listwise_scores",
    ),
    "joint_delta_u": (
        "GroupHeldoutSplit",
        "JointDeltaULossConfig",
        "JointDeltaUProbe",
        "JointDeltaUSelection",
        "ProposalSetEdges",
        "build_proposal_set_edges",
        "group_heldout_split",
        "joint_delta_u_loss",
        "joint_delta_u_metrics",
        "select_joint_delta_u_actions",
    ),
    "adaptive_consensus": (
        "proposal_graph_consensus_deltas",
    ),
    "post_nms_suppress": (
        "PostNMSSuppression",
        "PostNMSSuppressPolicyHead",
        "build_post_nms_detection_features",
        "select_post_nms_suppression",
        "suppress_detection",
    ),
}

SHIM_PUBLIC_SYMBOLS: dict[str, tuple[str, ...]] = {
    **ACTION_PUBLIC_SYMBOLS,
    **NATIVE_PUBLIC_SYMBOLS,
    **POLICY_PUBLIC_SYMBOLS,
}
SUBPACKAGE_BY_STEM = {
    **{stem: "action" for stem in ACTION_PUBLIC_SYMBOLS},
    **{stem: "native" for stem in NATIVE_PUBLIC_SYMBOLS},
    **{stem: "policy" for stem in POLICY_PUBLIC_SYMBOLS},
}

_ALL_SYMBOLS = [
    (module, symbol)
    for module, symbols in SHIM_PUBLIC_SYMBOLS.items()
    for symbol in symbols
]


def _flat_module(stem: str):
    return importlib.import_module(f"{PACKAGE_ROOT}.{stem}")


def _new_module(stem: str):
    subpackage = SUBPACKAGE_BY_STEM[stem]
    return importlib.import_module(f"{PACKAGE_ROOT}.{subpackage}.{stem}")


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
    new_obj = getattr(_new_module(stem), symbol)
    assert flat_obj is new_obj, (
        f"{stem}.{symbol}: flat import and {SUBPACKAGE_BY_STEM[stem]}.* import "
        "resolve to different objects; the flat module must be a pure "
        "forwarding shim"
    )


@pytest.mark.parametrize(
    "stem,symbol",
    [(m, s) for m, s in _ALL_SYMBOLS if s.islower()],
    ids=[f"{stem}.{symbol}" for stem, symbol in _ALL_SYMBOLS if symbol.islower()],
)
def test_callable_signature_parity(stem: str, symbol: str):
    flat_obj = getattr(_flat_module(stem), symbol)
    new_obj = getattr(_new_module(stem), symbol)
    assert str(inspect.signature(flat_obj)) == str(inspect.signature(new_obj))


def test_facade_reexports_match_new_subpackage():
    """Symbols re-exported by the package facade are the subpackage objects."""
    facade = importlib.import_module(PACKAGE_ROOT)
    for stem, symbols in SHIM_PUBLIC_SYMBOLS.items():
        new_module = _new_module(stem)
        for symbol in symbols:
            if symbol in facade.__all__:
                assert getattr(facade, symbol) is getattr(new_module, symbol), (
                    f"facade re-export {symbol} does not match "
                    f"{SUBPACKAGE_BY_STEM[stem]}.{stem}"
                )


# ---------------------------------------------------------------------------
# 2. Flat shims are pure forwarding modules (imports + __all__ only)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("stem", sorted(SHIM_PUBLIC_SYMBOLS))
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


@pytest.mark.parametrize("stem", sorted(SHIM_PUBLIC_SYMBOLS))
def test_flat_shim_all_matches_public_surface(stem: str):
    flat = _flat_module(stem)
    assert set(flat.__all__) == set(SHIM_PUBLIC_SYMBOLS[stem]), (
        f"{stem}.py __all__ drifted from the documented public surface: "
        f"missing {sorted(set(SHIM_PUBLIC_SYMBOLS[stem]) - set(flat.__all__))}, "
        f"extra {sorted(set(flat.__all__) - set(SHIM_PUBLIC_SYMBOLS[stem]))}"
    )


# ---------------------------------------------------------------------------
# 3. Deterministic CPU output parity through both import paths
# ---------------------------------------------------------------------------


def _tensor(seed: int, *shape: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=generator)


def _assert_value_equal(left, right, context: str) -> None:
    """Exact equality for tensors, dataclasses, mappings, sequences, scalars."""
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        assert isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor), (
            context
        )
        assert torch.equal(left, right), context
        return
    if dataclasses.is_dataclass(left) and not isinstance(left, type):
        assert dataclasses.is_dataclass(right) and not isinstance(right, type), context
        for field in dataclasses.fields(left):
            _assert_value_equal(
                getattr(left, field.name),
                getattr(right, field.name),
                f"{context}.{field.name}",
            )
        return
    if isinstance(left, dict):
        assert isinstance(right, dict) and left.keys() == right.keys(), context
        for key in left:
            _assert_value_equal(left[key], right[key], f"{context}[{key!r}]")
        return
    if isinstance(left, (tuple, list)) and not isinstance(left, str):
        assert isinstance(right, type(left)) and len(left) == len(right), context
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            _assert_value_equal(left_item, right_item, f"{context}[{index}]")
        return
    assert left == right, context


def test_action_head_forward_parity():
    head_cls_flat = getattr(_flat_module("actions"), "ActionLocalTransportHead")
    head_cls_new = getattr(_new_module("actions"), "ActionLocalTransportHead")
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
    actions_new = _new_module("actions")

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
    operators_new = _new_module("operators")

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
    preferences_new = _new_module("preferences")

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
    geo_new = _new_module("geometric_constraints")
    contracts_flat = _flat_module("contracts")
    contracts_new = _new_module("contracts")
    actions_flat = _flat_module("actions")
    actions_new = _new_module("actions")

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
# 3b. Native detector-coupled modules: behavioral parity
# ---------------------------------------------------------------------------


def test_native_contract_parity():
    """Zero-action parity helpers and candidate contract, both import paths."""
    flat = _flat_module("native_contract")
    new = _new_module("native_contract")

    assert torch.equal(flat.build_native_c1_deltas(0.05), new.build_native_c1_deltas(0.05))

    candidates_flat = flat.build_detector_native_candidates(
        spatial_features=_tensor(21, 3, 2, 7, 7),
        class_logits=torch.tensor([[0.0, 2.0], [2.0, 0.0], [0.0, 2.0]]),
        scores=torch.tensor([0.8, 0.9, 0.01]),
        boxes=torch.tensor([[0.0, 0.0, 2.0, 2.0]] * 3),
        image_indices=torch.tensor([0, 0, 1]),
        score_threshold=0.05,
    )
    candidates_new = new.build_detector_native_candidates(
        spatial_features=_tensor(21, 3, 2, 7, 7),
        class_logits=torch.tensor([[0.0, 2.0], [2.0, 0.0], [0.0, 2.0]]),
        scores=torch.tensor([0.8, 0.9, 0.01]),
        boxes=torch.tensor([[0.0, 0.0, 2.0, 2.0]] * 3),
        image_indices=torch.tensor([0, 0, 1]),
        score_threshold=0.05,
    )
    for field in (
        "spatial_features",
        "class_logits",
        "labels",
        "scores",
        "boxes",
        "image_indices",
        "proposal_indices",
    ):
        assert torch.equal(getattr(candidates_flat, field), getattr(candidates_new, field)), field

    artifact = {
        "completed": True,
        "aggregate_zero_action_parity": {"passed": True},
        "strict_zero_action_parity": {
            "passed": True,
            "images": 196,
            "mismatched_images": 0,
            "actions_are_exact_zero": True,
            "max_box_abs_error": 0.0,
            "max_score_abs_error": 0.0,
        },
    }
    assert flat.validate_strict_parity_artifact(artifact) == new.validate_strict_parity_artifact(artifact)

    payload = {
        "parity": {"baseline": {"passed": True}, "fullft": {"passed": True}},
        "candidate_contract": {
            "candidate_count": 9,
            "noop_is_first": True,
            "max_abs_delta": 0.05,
            "max_actions_per_image": 1,
            "detector_only": True,
            "budget_enforced": True,
            "score_threshold": 0.05,
            "candidate_source": "detector_visible_proposals_only",
            "training_gt_scope": "utility_labels_only_never_candidate_filter",
        },
    }
    config = {
        "gates": {
            "required_candidate_count": 9,
            "required_max_abs_delta": 0.05,
            "required_max_actions_per_image": 1,
            "required_score_threshold": 0.05,
            "required_candidate_source": "detector_visible_proposals_only",
            "required_training_gt_scope": "utility_labels_only_never_candidate_filter",
        }
    }
    assert flat.evaluate_native_contract_gates(payload, config) == new.evaluate_native_contract_gates(payload, config)


def test_native_topology_parity():
    """Class expansion, small-box filtering, class-wise NMS, top-K, ordering."""
    flat = _flat_module("native_topology")
    new = _new_module("native_topology")

    # Scenario 1 (from tests/test_energy_transport_native_topology.py):
    # competing classes, class-expanded NMS, normalized-rank ordering.
    boxes = torch.tensor(
        [
            [[0.0, 0.0, 4.0, 4.0], [0.0, 0.0, 4.0, 4.0], [7.0, 7.0, 9.0, 9.0]],
            [[0.0, 0.0, 4.0, 4.0], [0.5, 0.5, 4.5, 4.5], [6.0, 6.0, 9.0, 9.0]],
            [[5.0, 0.0, 8.0, 3.0], [5.0, 0.0, 8.0, 3.0], [5.0, 0.0, 8.0, 3.0]],
        ]
    )
    probabilities = torch.tensor(
        [[0.01, 0.90, 0.12], [0.01, 0.80, 0.95], [0.01, 0.70, 0.20]]
    )
    labels = torch.tensor([1, 1, 1])
    deltas = torch.tensor([[0.0, 0.0, 0.0, 0.0], [0.10, 0.0, 0.0, 0.0]])
    observable = torch.ones(3, dtype=torch.bool)
    kwargs = dict(
        image_size=(10, 10),
        candidate_deltas=deltas,
        observable_mask=observable,
        score_threshold=0.05,
        nms_threshold=0.5,
        detections_per_img=1,  # exercises the per-image top-K cap
    )
    assert torch.equal(
        flat.native_action_nms_topology(boxes, probabilities, labels, **kwargs),
        new.native_action_nms_topology(boxes, probabilities, labels, **kwargs),
    )

    # Scenario 2: a destructive delta clips the box and triggers small-box
    # removal plus class expansion across two foreground classes.
    boxes2 = torch.tensor([[[0.0, 0.0, 2.0, 2.0]] * 3])
    probabilities2 = torch.tensor([[0.01, 0.90, 0.80]])
    kwargs2 = dict(
        image_size=(4, 4),
        candidate_deltas=torch.tensor([[0.0, 0.0, 0.0, 0.0], [-10.0, 0.0, 0.0, 0.0]]),
        observable_mask=torch.tensor([True]),
        score_threshold=0.05,
        nms_threshold=0.5,
        detections_per_img=10,
    )
    assert torch.equal(
        flat.native_action_nms_topology(
            boxes2, probabilities2, torch.tensor([1]), **kwargs2
        ),
        new.native_action_nms_topology(
            boxes2, probabilities2, torch.tensor([1]), **kwargs2
        ),
    )


def _make_native_roi_state(contracts_module):
    """ROI state with logits and matched GT indices for native energy tests."""
    batch, dim, num_classes = 5, 8, 3
    return contracts_module.ROIActionState(
        features=_tensor(30, batch, dim),
        boxes=torch.tensor(
            [
                [10.0, 10.0, 30.0, 40.0],
                [2.0, 2.0, 20.0, 22.0],
                [5.0, 5.0, 9.0, 10.0],
                [12.0, 14.0, 24.0, 30.0],
                [30.0, 30.0, 48.0, 50.0],
            ]
        ),
        scores=torch.sigmoid(_tensor(31, batch)),
        labels=torch.tensor([1, 2, 1, 2, 1]),
        image_indices=torch.tensor([0, 0, 1, 1, 1]),
        proposal_indices=torch.arange(batch),
        logits=_tensor(32, batch, num_classes),
        matched_gt_indices=torch.tensor([0, 1, -1, 2, 0]),
        ious=torch.tensor([0.8, 0.6, 0.1, 0.78, 0.4]),
    )


_MATCHED_GT_BOXES = torch.tensor(
    [
        [11.0, 11.0, 29.0, 39.0],
        [3.0, 3.0, 19.0, 21.0],
        [0.0, 0.0, 1.0, 1.0],
        [13.0, 15.0, 23.0, 29.0],
        [31.0, 31.0, 47.0, 49.0],
    ]
)
_MATCHED_GT_LABELS = torch.tensor([1, 2, 0, 2, 1])
_IMAGE_SIZES = [(64, 64), (64, 64)]


def test_candidate_energy_parity():
    """Candidate ordering, BoxCoder decode, energy losses, top-K selection."""
    flat = _flat_module("candidate_energy")
    new = _new_module("candidate_energy")
    contracts_flat = _flat_module("contracts")
    contracts_new = _new_module("contracts")

    candidates_flat = flat.build_symmetric_box_candidates((0.05, 0.10))
    candidates_new = new.build_symmetric_box_candidates((0.05, 0.10))
    assert torch.equal(candidates_flat, candidates_new)

    state_flat = _make_native_roi_state(contracts_flat)
    state_new = _make_native_roi_state(contracts_new)

    targets_flat = flat.build_candidate_quality_targets(
        state_flat,
        candidates_flat,
        matched_gt_boxes=_MATCHED_GT_BOXES,
        matched_gt_labels=_MATCHED_GT_LABELS,
        image_sizes=_IMAGE_SIZES,
    )
    targets_new = new.build_candidate_quality_targets(
        state_new,
        candidates_new,
        matched_gt_boxes=_MATCHED_GT_BOXES,
        matched_gt_labels=_MATCHED_GT_LABELS,
        image_sizes=_IMAGE_SIZES,
    )
    for field in (
        "candidate_quality",
        "target_indices",
        "base_iou",
        "oracle_gain",
        "matched",
        "class_correct",
    ):
        assert torch.equal(getattr(targets_flat, field), getattr(targets_new, field)), field

    energies = _tensor(33, state_flat.batch_size, candidates_flat.shape[0])
    scores = state_flat.scores
    loss_flat = flat.candidate_action_energy_loss(
        energies,
        candidate_quality=targets_flat.candidate_quality,
        target_indices=targets_flat.target_indices,
        scores=scores,
    )
    loss_new = new.candidate_action_energy_loss(
        energies,
        candidate_quality=targets_new.candidate_quality,
        target_indices=targets_new.target_indices,
        scores=scores,
    )
    assert loss_flat.keys() == loss_new.keys()
    for key in loss_flat:
        assert torch.equal(loss_flat[key], loss_new[key]), key

    gain_flat = flat.candidate_action_gain_loss(
        energies,
        candidate_quality=targets_flat.candidate_quality,
        scores=scores,
    )
    gain_new = new.candidate_action_gain_loss(
        energies,
        candidate_quality=targets_new.candidate_quality,
        scores=scores,
    )
    assert gain_flat.keys() == gain_new.keys()
    for key in gain_flat:
        assert torch.equal(gain_flat[key], gain_new[key]), key

    # Ordering + per-image top-K: only the strongest energy drop per image acts.
    actions_flat, selected_flat, move_flat = flat.select_min_energy_box_actions(
        state_flat,
        candidates_flat,
        energies,
        max_actions_per_image=1,
    )
    actions_new, selected_new, move_new = new.select_min_energy_box_actions(
        state_new,
        candidates_new,
        energies,
        max_actions_per_image=1,
    )
    assert torch.equal(selected_flat, selected_new)
    assert torch.equal(move_flat, move_new)
    assert torch.equal(actions_flat.box_delta, actions_new.box_delta)

    # Seeded head forward parity (both head variants).
    for class_name, init, features in (
        (
            "SpatialCandidateEnergyHead",
            dict(in_channels=4, num_classes=3, hidden_dim=16, spatial_size=3),
            _tensor(34, 5, 4, 3, 3),
        ),
        (
            "ContextOnlyCandidateEnergyHead",
            dict(num_classes=3, hidden_dim=16),
            torch.empty(5, 0),
        ),
    ):
        torch.manual_seed(7)
        head_flat = getattr(flat, class_name)(**init).eval()
        torch.manual_seed(7)
        head_new = getattr(new, class_name)(**init).eval()
        class_logits = _tensor(35, 5, 3)
        labels = torch.tensor([1, 2, 1, 2, 1])
        head_scores = torch.rand(5, generator=torch.Generator().manual_seed(36))
        box_delta = _tensor(37, 5, 4) * 0.1
        with torch.no_grad():
            out_flat = head_flat(features, class_logits, labels, head_scores, box_delta)
            out_new = head_new(features, class_logits, labels, head_scores, box_delta)
        assert torch.equal(out_flat, out_new), class_name


def test_benefit_energy_parity():
    """Benefit targets, pairwise loss, gating order, energy-gap parity."""
    flat = _flat_module("benefit_energy")
    new = _new_module("benefit_energy")
    contracts_flat = _flat_module("contracts")
    contracts_new = _new_module("contracts")
    actions_flat = _flat_module("actions")
    actions_new = _new_module("actions")

    state_flat = _make_native_roi_state(contracts_flat)
    state_new = _make_native_roi_state(contracts_new)
    batch, dim = 5, 8
    action_flat = _make_actions(actions_flat, batch=batch, dim=dim)
    action_new = _make_actions(actions_new, batch=batch, dim=dim)

    targets_flat = flat.build_action_benefit_targets(
        state_flat,
        action_flat,
        matched_gt_boxes=_MATCHED_GT_BOXES,
        matched_gt_labels=_MATCHED_GT_LABELS,
        image_sizes=_IMAGE_SIZES,
    )
    targets_new = new.build_action_benefit_targets(
        state_new,
        action_new,
        matched_gt_boxes=_MATCHED_GT_BOXES,
        matched_gt_labels=_MATCHED_GT_LABELS,
        image_sizes=_IMAGE_SIZES,
    )
    for field in (
        "true_gain",
        "base_iou",
        "post_iou",
        "matched",
        "class_correct",
        "foreground_dominant",
    ):
        assert torch.equal(getattr(targets_flat, field), getattr(targets_new, field)), field

    predicted_gain = _tensor(38, batch)
    loss_flat = flat.action_benefit_energy_loss(
        predicted_gain,
        true_gain=targets_flat.true_gain,
        base_iou=targets_flat.base_iou,
        class_correct=targets_flat.class_correct,
        foreground_dominant=targets_flat.foreground_dominant,
        scores=state_flat.scores,
    )
    loss_new = new.action_benefit_energy_loss(
        predicted_gain,
        true_gain=targets_new.true_gain,
        base_iou=targets_new.base_iou,
        class_correct=targets_new.class_correct,
        foreground_dominant=targets_new.foreground_dominant,
        scores=state_new.scores,
    )
    assert loss_flat.keys() == loss_new.keys()
    for key in loss_flat:
        assert torch.equal(loss_flat[key], loss_new[key]), key

    gated_flat, mask_flat = flat.apply_action_benefit_gate(
        state_flat, action_flat, predicted_gain, max_actions_per_image=1
    )
    gated_new, mask_new = new.apply_action_benefit_gate(
        state_new, action_new, predicted_gain, max_actions_per_image=1
    )
    assert torch.equal(mask_flat, mask_new)
    assert torch.equal(gated_flat.box_delta, gated_new.box_delta)

    torch.manual_seed(9)
    head_flat = flat.ActionBenefitEnergyHead(feature_dim=dim, num_classes=3, hidden_dim=16).eval()
    torch.manual_seed(9)
    head_new = new.ActionBenefitEnergyHead(feature_dim=dim, num_classes=3, hidden_dim=16).eval()
    class_logits = _tensor(39, batch, 3)
    labels = torch.tensor([1, 2, 1, 2, 1])
    box_delta = _tensor(40, batch, 4) * 0.1
    with torch.no_grad():
        gap_flat = head_flat.energy_gap(
            state_flat.features, class_logits, labels, state_flat.scores, box_delta
        )
        gap_new = head_new.energy_gap(
            state_new.features, class_logits, labels, state_new.scores, box_delta
        )
    assert torch.equal(gap_flat, gap_new)


# ---------------------------------------------------------------------------
# 3c. Policy selection modules: behavioral parity (Task 14 phase 1)
# ---------------------------------------------------------------------------


_CANDIDATE_DELTAS = torch.tensor(
    [[0.0, 0.0, 0.0, 0.0], [0.05, 0.0, 0.0, 0.0], [-0.05, 0.0, 0.0, 0.0]]
)


def _detector_input_tensors(
    seed: int, batch: int, in_channels: int, num_classes: int, spatial: int = 3
):
    """Seeded detector-visible proposal inputs for policy head parity."""
    generator = torch.Generator().manual_seed(seed)
    spatial_features = torch.randn(batch, in_channels, spatial, spatial, generator=generator)
    class_logits = torch.randn(batch, num_classes, generator=generator)
    labels = torch.randint(0, num_classes, (batch,), generator=generator)
    scores = torch.rand(batch, generator=generator)
    corners = torch.rand(batch, 4, generator=generator) * 48.0
    boxes = torch.stack(
        (
            torch.minimum(corners[:, 0], corners[:, 2]),
            torch.minimum(corners[:, 1], corners[:, 3]),
            torch.maximum(corners[:, 0], corners[:, 2]) + 4.0,
            torch.maximum(corners[:, 1], corners[:, 3]) + 4.0,
        ),
        dim=1,
    )
    return spatial_features, class_logits, labels, scores, boxes


def test_search_parity():
    """Per-GT min-energy ties, ordering, per-image budget, and demote path."""
    flat = _flat_module("search")
    new = _new_module("search")

    scores = torch.tensor([0.04, 0.04, 0.03, 0.10, 0.12])
    ious = torch.tensor([0.80, 0.78, 0.85, 0.20, 0.10])
    image_indices = torch.tensor([0, 0, 0, 1, 1])
    gt_indices = torch.tensor([0, 0, 1, -1, -1])
    config_flat = flat.ActionSearchConfig(max_rescues_per_image=1, demote_low_quality=True)
    config_new = new.ActionSearchConfig(max_rescues_per_image=1, demote_low_quality=True)

    result_flat = flat.select_min_energy_score_actions(
        scores, ious, image_indices, gt_indices=gt_indices, config=config_flat
    )
    result_new = new.select_min_energy_score_actions(
        scores, ious, image_indices, gt_indices=gt_indices, config=config_new
    )
    _assert_value_equal(result_flat, result_new, "search result")

    prediction = {
        "boxes": torch.tensor([[0.0, 0.0, 4.0, 4.0]] * 5),
        "scores": scores.clone(),
        "labels": torch.tensor([1, 1, 1, 2, 2]),
    }
    applied_flat = flat.apply_score_action_to_prediction(prediction, result_flat.score_delta)
    applied_new = new.apply_score_action_to_prediction(prediction, result_new.score_delta)
    assert applied_flat.keys() == applied_new.keys()
    for key in applied_flat:
        assert torch.equal(applied_flat[key], applied_new[key]), key


def test_set_search_parity():
    """Canonical ordering, seeded statistics, and native-routed matcher."""
    flat = _flat_module("set_search")
    new = _new_module("set_search")

    def make_candidates(module):
        return (
            module.ActionCandidate("rescue-2", action_energy=2.0),
            module.ActionCandidate("rescue-1", action_energy=1.0),
            module.ActionCandidate("suppress-1", action_energy=0.5),
        )

    def make_evaluator(module):
        def evaluate(actions):
            ids = {action.action_id for action in actions}
            tp75 = 1 + (2 if "rescue-1" in ids else 0) + (1 if "rescue-2" in ids else 0)
            fp = 4 - (2 if "suppress-1" in ids else 0)
            energy = sum(float(action.action_energy) for action in actions)
            return module.SetOutcome(
                tp50=tp75 + 1,
                tp75=tp75,
                fp50=fp,
                fp75=fp,
                prediction_count=6,
                action_energy=energy,
                action_count=len(actions),
            )

        return evaluate

    candidates_flat = make_candidates(flat)
    candidates_new = make_candidates(new)
    evaluate_flat = make_evaluator(flat)
    evaluate_new = make_evaluator(new)

    local_flat = flat.local_top_b(candidates_flat, evaluate_flat, top_b=2)
    local_new = new.local_top_b(candidates_new, evaluate_new, top_b=2)
    _assert_value_equal(local_flat, local_new, "local_top_b")

    greedy_flat = flat.greedy_positive_marginal_selection(
        candidates_flat, evaluate_flat, max_actions=2
    )
    greedy_new = new.greedy_positive_marginal_selection(
        candidates_new, evaluate_new, max_actions=2
    )
    _assert_value_equal(greedy_flat, greedy_new, "greedy selection")

    beam_flat = flat.beam_search(candidates_flat, evaluate_flat, beam_width=2, max_depth=2)
    beam_new = new.beam_search(candidates_new, evaluate_new, beam_width=2, max_depth=2)
    _assert_value_equal(beam_flat, beam_new, "beam search")

    assert flat.deterministic_delta_permutation(
        (0.1, -0.1, 0.2, -0.2), seed=7
    ) == new.deterministic_delta_permutation((0.1, -0.1, 0.2, -0.2), seed=7)

    summary_flat = flat.paired_bootstrap_summary(
        (0.3, -0.1, 0.2, 0.05), (0.1, 0.0, 0.1, -0.05), num_resamples=100, seed=11
    )
    summary_new = new.paired_bootstrap_summary(
        (0.3, -0.1, 0.2, 0.05), (0.1, 0.0, 0.1, -0.05), num_resamples=100, seed=11
    )
    _assert_value_equal(summary_flat, summary_new, "paired bootstrap")

    prediction = {
        "boxes": torch.tensor(
            [[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0], [40.0, 40.0, 45.0, 45.0]]
        ),
        "labels": torch.tensor([1, 2, 1]),
        "scores": torch.tensor([0.9, 0.8, 0.1]),
    }
    target = {
        "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 31.0, 31.0]]),
        "labels": torch.tensor([1, 2]),
    }
    outcome_flat = flat.set_outcome_from_prediction(prediction, target)
    outcome_new = new.set_outcome_from_prediction(prediction, target)
    _assert_value_equal(outcome_flat, outcome_new, "set outcome")
    assert outcome_flat.tp50 == 2 and outcome_flat.fp50 == 1


def test_set_policy_parity():
    """Head forward, conflict statistics, budgeted selection, and loss."""
    flat = _flat_module("set_policy")
    new = _new_module("set_policy")

    inputs = _detector_input_tensors(41, 4, 2, 3)
    torch.manual_seed(5)
    head_flat = flat.NMSAwareSetPolicyHead(
        2, 3, _CANDIDATE_DELTAS, hidden_dim=8, spatial_size=2
    ).eval()
    torch.manual_seed(5)
    head_new = new.NMSAwareSetPolicyHead(
        2, 3, _CANDIDATE_DELTAS, hidden_dim=8, spatial_size=2
    ).eval()
    with torch.no_grad():
        out_flat = head_flat(*inputs, (128, 128))
        out_new = head_new(*inputs, (128, 128))
    _assert_value_equal(out_flat, out_new, "set policy head output")

    _, _, labels, scores, boxes = inputs
    assert torch.equal(
        flat.class_aware_conflict_statistics(boxes, labels, scores, (128, 128)),
        new.class_aware_conflict_statistics(boxes, labels, scores, (128, 128)),
    )

    action_logits = torch.tensor(
        [[0.1, 0.5, -0.2], [0.3, 0.1, 0.4], [0.0, 0.9, 0.2], [0.2, -0.1, 0.1]]
    )
    move_logits = torch.tensor([0.5, -1.0, 0.3, 2.0])
    conflict = torch.zeros(4, 4)
    output_flat = flat.SetPolicyOutput(action_logits, move_logits, conflict)
    output_new = new.SetPolicyOutput(action_logits, move_logits, conflict)
    image_indices = torch.tensor([0, 0, 1, 1])
    observable = torch.ones(4, dtype=torch.bool)
    selection_flat = flat.select_set_policy_actions(
        output_flat, _CANDIDATE_DELTAS, image_indices, observable, max_actions_per_image=1
    )
    selection_new = new.select_set_policy_actions(
        output_new, _CANDIDATE_DELTAS, image_indices, observable, max_actions_per_image=1
    )
    _assert_value_equal(selection_flat, selection_new, "set policy selection")
    assert selection_flat.selected_mask.tolist() == [True, False, True, False]

    action_targets = torch.tensor([1, 2, 1, 1])
    move_targets = torch.tensor([1.0, 0.0, 1.0, 0.0])
    supervision_mask = torch.tensor([True, True, True, False])
    loss_flat = flat.set_policy_loss(
        output_flat,
        action_targets,
        move_targets,
        supervision_mask,
        flat.SetPolicyLossConfig(),
    )
    loss_new = new.set_policy_loss(
        output_new,
        action_targets,
        move_targets,
        supervision_mask,
        new.SetPolicyLossConfig(),
    )
    _assert_value_equal(loss_flat, loss_new, "set policy loss")


def test_global_top1_parity():
    """Seeded head forwards, flatten/select ordering, topology, and losses."""
    flat = _flat_module("global_top1")
    new = _new_module("global_top1")

    inputs = _detector_input_tensors(51, 4, 2, 3)
    observable = torch.tensor([True, True, False, True])

    for class_name, extra in (
        ("GlobalTop1PolicyHead", {}),
        ("SetContextGlobalTop1PolicyHead", {}),
        ("ActionTopologyGlobalTop1PolicyHead", {}),
        ("NativeActionTopologyGlobalTop1PolicyHead", {"topology_dim": 8}),
    ):
        init = dict(in_channels=2, num_classes=3, candidate_deltas=_CANDIDATE_DELTAS,
                    hidden_dim=8, spatial_size=2, **extra)
        torch.manual_seed(13)
        head_flat = getattr(flat, class_name)(**init).eval()
        torch.manual_seed(13)
        head_new = getattr(new, class_name)(**init).eval()
        kwargs = {}
        if class_name == "ActionTopologyGlobalTop1PolicyHead":
            kwargs["topology_shuffle_seed"] = 7
        if class_name == "NativeActionTopologyGlobalTop1PolicyHead":
            kwargs["native_topology"] = _tensor(52, 4, 3, 8)
            kwargs["topology_shuffle_seed"] = 7
        with torch.no_grad():
            out_flat = head_flat(*inputs, (128, 128), observable, **kwargs)
            out_new = head_new(*inputs, (128, 128), observable, **kwargs)
        _assert_value_equal(out_flat, out_new, f"{class_name} output")

    _, _, labels, scores, boxes = inputs
    assert torch.equal(
        flat.action_conditioned_nms_topology(
            boxes, labels, scores, (128, 128), _CANDIDATE_DELTAS, 0.5
        ),
        new.action_conditioned_nms_topology(
            boxes, labels, scores, (128, 128), _CANDIDATE_DELTAS, 0.5
        ),
    )

    action_logits = torch.tensor(
        [[0.1, 0.9, -0.3], [0.2, -0.1, 0.5], [0.0, 0.4, 0.6], [0.3, 0.2, 0.1]]
    )
    output_flat = flat.GlobalTop1Output(action_logits, torch.tensor(0.05), torch.zeros(4, 4))
    output_new = new.GlobalTop1Output(action_logits, torch.tensor(0.05), torch.zeros(4, 4))
    full_mask = torch.ones(4, dtype=torch.bool)

    flattened_flat = flat.flatten_observable_action_logits(output_flat, full_mask)
    flattened_new = new.flatten_observable_action_logits(output_new, full_mask)
    _assert_value_equal(flattened_flat, flattened_new, "flattened logits")

    selection_flat = flat.select_global_top1_action(output_flat, _CANDIDATE_DELTAS, full_mask)
    selection_new = new.select_global_top1_action(output_new, _CANDIDATE_DELTAS, full_mask)
    _assert_value_equal(selection_flat, selection_new, "global top1 selection")

    delta_u = torch.tensor(
        [[0.0, 0.5, -0.1], [0.0, -0.2, 0.3], [0.0, 0.1, 0.2], [0.0, 0.0, 0.0]]
    )
    target_flat = flat.build_global_top1_target(delta_u, full_mask)
    target_new = new.build_global_top1_target(delta_u, full_mask)
    _assert_value_equal(target_flat, target_new, "global top1 target")
    assert not target_flat.is_noop and target_flat.proposal_index == 0

    loss_flat = flat.global_top1_loss(output_flat, full_mask, target_flat)
    loss_new = new.global_top1_loss(output_new, full_mask, target_new)
    _assert_value_equal(loss_flat, loss_new, "global top1 loss")

    margin_flat = flat.global_top1_balanced_margin_loss(output_flat, full_mask, target_flat)
    margin_new = new.global_top1_balanced_margin_loss(output_new, full_mask, target_new)
    _assert_value_equal(margin_flat, margin_new, "global top1 margin loss")


def test_listwise_noop_parity():
    """Batching, float64 LBFGS fit, calibration, gates, and paired stats."""
    flat = _flat_module("listwise_noop")
    new = _new_module("listwise_noop")

    features = torch.tensor([[2.0, 0.5], [1.0, -0.5], [-1.0, 0.5], [-2.0, -0.5]])
    target = torch.tensor([0.3, -0.1, 0.2, -0.2])
    image_ids = torch.tensor([0, 0, 1, 1])

    batch_flat = flat.build_listwise_batch(features, target, image_ids, target_epsilon=0.001)
    batch_new = new.build_listwise_batch(features, target, image_ids, target_epsilon=0.001)
    _assert_value_equal(batch_flat, batch_new, "listwise batch")

    model_flat = flat.fit_listwise_model(
        features, target, image_ids, target_epsilon=0.001, l2=0.01, max_iter=5
    )
    model_new = new.fit_listwise_model(
        features, target, image_ids, target_epsilon=0.001, l2=0.01, max_iter=5
    )
    _assert_value_equal(model_flat, model_new, "listwise model")

    scores_flat = flat.predict_listwise_scores(model_flat, features)
    scores_new = new.predict_listwise_scores(model_new, features)
    assert torch.equal(scores_flat, scores_new)

    metrics_flat = flat.listwise_decision_metrics(
        scores_flat,
        noop_logit=model_flat.noop_logit,
        noop_margin=0.0,
        target=target,
        image_ids=image_ids,
        target_epsilon=0.001,
        lcb_z=1.96,
    )
    metrics_new = new.listwise_decision_metrics(
        scores_new,
        noop_logit=model_new.noop_logit,
        noop_margin=0.0,
        target=target,
        image_ids=image_ids,
        target_epsilon=0.001,
        lcb_z=1.96,
    )
    _assert_value_equal(metrics_flat, metrics_new, "listwise decision metrics")

    calibration_flat = flat.calibrate_noop_margin(
        scores_flat,
        noop_logit=model_flat.noop_logit,
        target=target,
        image_ids=image_ids,
        target_epsilon=0.001,
        min_selected_actions=1,
        max_action_image_rate=1.0,
        min_positive_precision_lift=-1.0,
        min_mean_delta_u_lcb=-1.0,
        lcb_z=1.96,
    )
    calibration_new = new.calibrate_noop_margin(
        scores_new,
        noop_logit=model_new.noop_logit,
        target=target,
        image_ids=image_ids,
        target_epsilon=0.001,
        min_selected_actions=1,
        max_action_image_rate=1.0,
        min_positive_precision_lift=-1.0,
        min_mean_delta_u_lcb=-1.0,
        lcb_z=1.96,
    )
    _assert_value_equal(calibration_flat, calibration_new, "noop calibration")

    assert flat.paired_gain_stats([0.3, 0.1], [0.1, 0.0], lcb_z=1.96) == new.paired_gain_stats(
        [0.3, 0.1], [0.1, 0.0], lcb_z=1.96
    )

    metrics = {
        "selected_count": 3,
        "action_image_rate": 0.5,
        "positive_precision_lift": 0.2,
        "mean_delta_u_lcb": 0.1,
        "mean_delta_u": 0.15,
    }
    payload = {
        "support": {"candidate_count": 10, "image_count": 5},
        "arms": {"local_full": metrics},
        "selection": {
            "local_full": {
                "used_identity_fallback": False,
                "fit_metrics": {"mean_delta_u": 0.2},
                "tune_metrics": {"mean_delta_u": 0.18},
            }
        },
        "paired_gains": {
            "vs_rate_matched_random": {"lcb": 0.05},
            "vs_feature_shuffle": {"lcb": 0.04},
            "vs_label_shuffle": {"lcb": 0.03},
        },
    }
    config = {
        "gates": {
            "min_candidates": 4,
            "min_images": 2,
            "min_selected_actions": 1,
            "max_action_image_rate": 1.0,
            "min_positive_precision_lift": 0.0,
            "min_mean_delta_u_lcb": 0.0,
            "min_paired_gain_lcb": 0.01,
            "max_fit_to_outer_mean_gap": 0.1,
            "max_tune_to_outer_mean_gap": 0.1,
        }
    }
    gates_flat = flat.evaluate_listwise_gates(payload, config)
    gates_new = new.evaluate_listwise_gates(payload, config)
    _assert_value_equal(gates_flat, gates_new, "listwise gates")
    assert gates_flat["all_passed"] is True


def test_joint_delta_u_parity():
    """Set edges, heldout split, probe forward, selection, loss, metrics."""
    flat = _flat_module("joint_delta_u")
    new = _new_module("joint_delta_u")

    inputs = _detector_input_tensors(61, 4, 2, 3)
    _, class_logits, labels, scores, boxes = inputs
    image_indices = torch.tensor([0, 0, 1, 1])
    image_sizes = torch.tensor([[64.0, 64.0], [64.0, 64.0]])

    edges_flat = flat.build_proposal_set_edges(
        boxes, labels, scores, image_indices, image_sizes, max_neighbors=2
    )
    edges_new = new.build_proposal_set_edges(
        boxes, labels, scores, image_indices, image_sizes, max_neighbors=2
    )
    _assert_value_equal(edges_flat, edges_new, "proposal set edges")

    group_ids = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    split_flat = flat.group_heldout_split(group_ids, 0.25, seed=7)
    split_new = new.group_heldout_split(group_ids, 0.25, seed=7)
    _assert_value_equal(split_flat, split_new, "group heldout split")

    torch.manual_seed(17)
    probe_flat = flat.JointDeltaUProbe(2, 3, _CANDIDATE_DELTAS, hidden_dim=8).eval()
    torch.manual_seed(17)
    probe_new = new.JointDeltaUProbe(2, 3, _CANDIDATE_DELTAS, hidden_dim=8).eval()
    with torch.no_grad():
        delta_u_flat = probe_flat(*inputs, image_indices, image_sizes, edges_flat)
        delta_u_new = probe_new(*inputs, image_indices, image_sizes, edges_new)
    assert torch.equal(delta_u_flat, delta_u_new)

    selection_flat = flat.select_joint_delta_u_actions(
        _CANDIDATE_DELTAS, delta_u_flat, image_indices, max_actions_per_image=1
    )
    selection_new = new.select_joint_delta_u_actions(
        _CANDIDATE_DELTAS, delta_u_new, image_indices, max_actions_per_image=1
    )
    _assert_value_equal(selection_flat, selection_new, "joint delta-u selection")

    target = _tensor(62, 4, 3)
    mask = torch.tensor(
        [[True, True, True], [True, True, False], [True, True, True], [False, True, True]]
    )
    loss_flat = flat.joint_delta_u_loss(delta_u_flat, target, mask, image_indices)
    loss_new = new.joint_delta_u_loss(delta_u_new, target, mask, image_indices)
    _assert_value_equal(loss_flat, loss_new, "joint delta-u loss")

    metrics_flat = flat.joint_delta_u_metrics(
        delta_u_flat,
        target,
        mask,
        selected_mask=selection_flat.selected_mask,
        candidate_indices=selection_flat.candidate_indices,
    )
    metrics_new = new.joint_delta_u_metrics(
        delta_u_new,
        target,
        mask,
        selected_mask=selection_new.selected_mask,
        candidate_indices=selection_new.candidate_indices,
    )
    _assert_value_equal(metrics_flat, metrics_new, "joint delta-u metrics")


def test_adaptive_consensus_parity():
    """Consensus deltas are identical through both import paths."""
    flat = _flat_module("adaptive_consensus")
    new = _new_module("adaptive_consensus")

    boxes = torch.tensor(
        [[0.0, 0.0, 4.0, 4.0], [1.0, 0.0, 5.0, 4.0], [20.0, 20.0, 24.0, 24.0]]
    )
    labels = torch.tensor([1, 1, 2])
    scores = torch.tensor([0.5, 0.9, 0.7])
    observable = torch.tensor([True, True, False])
    deltas_flat = flat.proposal_graph_consensus_deltas(
        boxes,
        labels=labels,
        scores=scores,
        observable_mask=observable,
        min_peer_iou=0.3,
        max_abs_delta=0.05,
    )
    deltas_new = new.proposal_graph_consensus_deltas(
        boxes,
        labels=labels,
        scores=scores,
        observable_mask=observable,
        min_peer_iou=0.3,
        max_abs_delta=0.05,
    )
    assert torch.equal(deltas_flat, deltas_new)


def test_post_nms_suppress_parity():
    """Feature builder, head forward, suppression selection, and removal."""
    flat = _flat_module("post_nms_suppress")
    new = _new_module("post_nms_suppress")

    boxes = torch.tensor(
        [[0.0, 0.0, 2.0, 2.0], [1.0, 1.0, 3.0, 3.0], [3.0, 0.0, 4.0, 1.0]]
    )
    scores = torch.tensor([0.9, 0.7, 0.5])
    labels = torch.tensor([1, 1, 2])
    features_flat = flat.build_post_nms_detection_features(
        boxes, scores, labels, (4, 4), num_classes=3
    )
    features_new = new.build_post_nms_detection_features(
        boxes, scores, labels, (4, 4), num_classes=3
    )
    assert torch.equal(features_flat, features_new)
    assert features_flat.shape == (3, 14)

    observable = torch.tensor([True, True, False])
    torch.manual_seed(23)
    head_flat = flat.PostNMSSuppressPolicyHead(14, hidden_dim=8).eval()
    torch.manual_seed(23)
    head_new = new.PostNMSSuppressPolicyHead(14, hidden_dim=8).eval()
    with torch.no_grad():
        out_flat = head_flat(features_flat, observable)
        out_new = head_new(features_new, observable)
    _assert_value_equal(out_flat, out_new, "post-nms head output")

    selection_flat = flat.select_post_nms_suppression(out_flat, observable)
    selection_new = new.select_post_nms_suppression(out_new, observable)
    _assert_value_equal(selection_flat, selection_new, "post-nms selection")

    prediction = {"boxes": boxes, "scores": scores, "labels": labels}
    kept_flat = flat.suppress_detection(prediction, 1)
    kept_new = new.suppress_detection(prediction, 1)
    assert kept_flat.keys() == kept_new.keys()
    for key in kept_flat:
        assert torch.equal(kept_flat[key], kept_new[key]), key


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


def _fixture_forward(entry: dict, head: torch.nn.Module):
    """Build the deterministic CPU input declared by the fixture manifest."""
    spec = entry["input_spec"]
    init_kwargs = entry["init_kwargs"]
    tensor_kwargs = entry.get("tensor_kwargs", {})
    batch = int(spec["batch"])
    generator = torch.Generator().manual_seed(int(spec["seed"]))
    kind = spec["kind"]
    if kind == "feature_only":
        features = torch.randn(batch, init_kwargs["feature_dim"], generator=generator)
        return head(features)
    if kind == "roi_energy":
        num_classes = int(init_kwargs["num_classes"])
        if spec.get("spatial"):
            features = torch.randn(
                batch,
                int(init_kwargs["in_channels"]),
                int(init_kwargs["spatial_size"]),
                int(init_kwargs["spatial_size"]),
                generator=generator,
            )
        else:
            features = torch.randn(
                batch, int(init_kwargs.get("feature_dim", 0)), generator=generator
            )
        class_logits = torch.randn(batch, num_classes, generator=generator)
        labels = torch.randint(0, num_classes, (batch,), generator=generator)
        scores = torch.rand(batch, generator=generator)
        box_delta = torch.randn(batch, 4, generator=generator) * 0.1
        return head(features, class_logits, labels, scores, box_delta)
    if kind in {
        "set_policy_output",
        "global_top1_output",
        "adaptive_consensus_output",
        "native_topology_output",
    }:
        inputs = _detector_input_tensors(
            int(spec["seed"]),
            batch,
            int(init_kwargs["in_channels"]),
            int(init_kwargs["num_classes"]),
        )
        image_size = tuple(spec.get("image_size", [128, 128]))
        if kind == "set_policy_output":
            return head(*inputs, image_size)
        observable_mask = torch.tensor(
            spec.get("observable", [True] * batch), dtype=torch.bool
        )
        if kind == "global_top1_output":
            return head(*inputs, image_size, observable_mask)
        if kind == "adaptive_consensus_output":
            adaptive_deltas = torch.randn(batch, 4, generator=generator) * 0.05
            return head(
                *inputs, image_size, observable_mask, adaptive_deltas=adaptive_deltas
            )
        topology_dim = int(init_kwargs["topology_dim"])
        candidate_count = len(tensor_kwargs["candidate_deltas"])
        native_topology = torch.randn(
            batch, candidate_count, topology_dim, generator=generator
        )
        return head(
            *inputs, image_size, observable_mask, native_topology=native_topology
        )
    if kind == "joint_delta_u_probe":
        inputs = _detector_input_tensors(
            int(spec["seed"]),
            batch,
            int(init_kwargs["in_channels"]),
            int(init_kwargs["num_classes"]),
        )
        image_indices = torch.zeros(batch, dtype=torch.long)
        image_sizes = torch.tensor(spec["image_sizes"], dtype=torch.float32)
        return head(*inputs, image_indices, image_sizes)
    if kind == "post_nms_output":
        features = torch.randn(batch, int(init_kwargs["input_dim"]), generator=generator)
        observable_mask = torch.tensor(
            spec.get("observable", [True] * batch), dtype=torch.bool
        )
        return head(features, observable_mask)
    raise AssertionError(f"unknown fixture input_spec kind: {kind}")


def test_fixture_state_dict_strict_loads_through_both_paths():
    manifest = _load_manifest()
    for entry in manifest["fixtures"]:
        payload = torch.load(
            FIXTURE_DIR / entry["file"], map_location="cpu", weights_only=True
        )
        module_stem = entry["module"].rsplit(".", 1)[-1]
        init_kwargs = dict(entry["init_kwargs"])
        for name, value in entry.get("tensor_kwargs", {}).items():
            init_kwargs[name] = torch.tensor(value)

        heads = []
        for loader in (_flat_module, _new_module):
            cls = getattr(loader(module_stem), entry["class"])
            module = cls(**init_kwargs).eval()
            module.load_state_dict(payload, strict=True)
            heads.append(module)

        flat_head, new_head = heads
        assert list(flat_head.state_dict()) == list(payload), (
            f"state-dict key drift for {entry['class']} (flat path)"
        )
        assert list(new_head.state_dict()) == list(payload), (
            f"state-dict key drift for {entry['class']} "
            f"({SUBPACKAGE_BY_STEM[module_stem]}.* path)"
        )
        for key, tensor in payload.items():
            assert torch.equal(flat_head.state_dict()[key], tensor), key
            assert torch.equal(new_head.state_dict()[key], tensor), key

        with torch.no_grad():
            out_flat = _fixture_forward(entry, flat_head)
            out_new = _fixture_forward(entry, new_head)
        _assert_value_equal(
            out_flat,
            out_new,
            f"deterministic output drift for {entry['class']}",
        )


def test_manifest_covers_every_shimmed_module():
    """Each action/native/policy module is fixture-backed or explicitly skipped."""
    manifest = _load_manifest()
    covered = {entry["module"].rsplit(".", 1)[-1] for entry in manifest["fixtures"]}
    skipped = {entry["module"].rsplit(".", 1)[-1] for entry in manifest["skipped"]}
    expected = set(SHIM_PUBLIC_SYMBOLS)
    assert covered | skipped == expected, (
        f"manifest coverage drift: uncovered {sorted(expected - covered - skipped)}, "
        f"unexpected {sorted((covered | skipped) - expected)}"
    )
    assert not (covered & skipped), "module listed as both fixture and skipped"
    for entry in manifest["skipped"]:
        assert entry.get("reason"), (
            f"skipped module {entry['module']} must document why no fixture exists"
        )
    for entry in manifest["fixtures"]:
        assert entry.get("input_spec"), (
            f"fixture {entry['name']} must declare a deterministic input_spec"
        )
