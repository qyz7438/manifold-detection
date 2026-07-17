"""Documentation contracts for oracle-utility-weighted box-head training.

Executable behavior is covered by the AWR weighting, dataset, bootstrap, and
runner tests. These checks only keep the preregistration text synchronized.
"""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "docs" / "awr_weighted_boxhead_protocol.md"


def _normalize(text: str) -> str:
    return " ".join(text.split())


def _section(text: str, heading: str) -> str:
    marker = f"## {heading}"
    assert marker in text
    remainder = text.split(marker, 1)[1]
    return _normalize(remainder.split("\n## ", 1)[0])


def test_protocol_states_weighted_erm_claim_boundary() -> None:
    text = PROTOCOL_PATH.read_text(encoding="utf-8")
    status = _section(text, "Status")
    boundary = _section(text, "Claim Boundary")
    assert "weighted supervised empirical-risk minimization" in status
    assert "not an implementation of AWR, AWAC, CRR, or RL" in status
    assert "does not establish proposal-level weighting" in boundary
    assert "manifold correction" in boundary


def test_protocol_locks_train_only_split_roles() -> None:
    section = _section(PROTOCOL_PATH.read_text(encoding="utf-8"), "Data Boundary")
    assert "fit 250, tune 68, calibration 68, outer_heldout 68" in section
    assert "Training reads fit only" in section
    assert "Exploratory gates read tune only" in section
    assert "Calibration and outer_heldout remain unread" in section
    assert "held out from the weighted continuation" in section


def test_protocol_uses_only_noop_relative_re_roi_utility() -> None:
    section = _section(PROTOCOL_PATH.read_text(encoding="utf-8"), "Oracle Utility")
    assert "only permitted source" in section
    assert "There is no fallback utility source" in section
    assert (
        "candidate_utility(S, i) = max_{a != identity} Q_teacher_raw(S, i, a)"
        in section
    )
    assert "raw_locked_scalarization_v1" in section
    assert "image_utility(S) = max(0, max_i candidate_utility(S, i))" in section
    assert "no valid non-identity action receives `image_utility = 0`" in section
    assert "IoU-score rescue" in section


def test_protocol_locks_global_fit_mean_weight_normalization() -> None:
    section = _section(PROTOCOL_PATH.read_text(encoding="utf-8"), "Locked Weights")
    assert "lambda = 1.0" in section
    assert "w_max = 20.0" in section
    assert "w(S) = raw_w(S) / mean_fit(raw_w)" in section
    assert "Global fit-mean normalization is mandatory" in section
    assert "Batch-local normalization is forbidden" in section
    assert "effective sample size" in section


def test_protocol_locks_five_arms_and_primary_contrasts() -> None:
    section = _section(PROTOCOL_PATH.read_text(encoding="utf-8"), "Arms")
    for arm in ("`Z` `zero_train`", "`U` `uniform`", "`W` `oracle_weighted`", "`F` `positive_filter`", "`S` `utility_shuffle`"):
        assert arm in section
    assert "primary transfer test is `W versus U`" in section
    assert "causal assignment test is `W versus S`" in section
    assert "exact weight multiset is preserved" in section


def test_protocol_locks_training_seeds_and_final_epoch() -> None:
    section = _section(PROTOCOL_PATH.read_text(encoding="utf-8"), "Locked Training Configuration")
    assert "Training seeds: `42`, `2024`, `999`" in section
    assert "Epochs: 10 fixed epochs" in section
    assert "Logical batch size: 8 images" in section
    assert "Checkpoint selection: final epoch only" in section
    assert "Tune may not select an epoch" in section
    assert "SGD, learning rate `0.001`, momentum `0.9`, weight decay `0.0005`" in section


def test_protocol_forbids_weighting_batch_aggregated_loss() -> None:
    section = _section(PROTOCOL_PATH.read_text(encoding="utf-8"), "Locked Training Configuration")
    assert "L_B = sum_{S in B} w(S) * L_box(S) / |B|" in section
    assert "not the batch weight sum" in section
    assert "already batch-aggregated TorchVision loss is forbidden" in section
    assert "All arms use the same per-image-loss path" in section


def test_protocol_recomputes_global_ap_for_paired_bootstrap() -> None:
    section = _section(PROTOCOL_PATH.read_text(encoding="utf-8"), "Evaluation And Bootstrap")
    assert "recomputes global AP75 after every resample" in section
    assert "Averaging per-image AP is forbidden" in section
    assert "10,000 image resamples with seed 42" in section
    assert "hierarchical seed-and-image paired interval" in section


def test_protocol_separates_contract_and_scientific_failures() -> None:
    section = _section(PROTOCOL_PATH.read_text(encoding="utf-8"), "Failure And Branch Rules")
    assert "contract failure" in section
    assert "invalidates the run" in section
    assert "does not count as scientific evidence" in section
    assert "any scientific gate fails" in section
    assert "freeze this image-level oracle-utility weighting direction" in section


def test_protocol_locks_support_weight_budget_and_detector_gates() -> None:
    section = _section(PROTOCOL_PATH.read_text(encoding="utf-8"), "Locked Gates")
    for gate in (
        "`provenance`",
        "`support`",
        "`weight_health`",
        "`budget_match`",
        "`advantage_causality`",
        "`uniform_gain`",
        "`safety`",
        "`calibration`",
        "`generalization`",
    ):
        assert gate in section
    assert "ESS is at least 50%" in section
    assert "at least `+0.002`" in section
    assert "no per-seed point delta is below `-0.002`" in section


def test_protocol_rejects_incompatible_transform_cache() -> None:
    section = _section(PROTOCOL_PATH.read_text(encoding="utf-8"), "Locked Training Configuration")
    assert "`min_size=max_size=480`" in section
    assert "a cache generated at 320 is incompatible" in section


def test_protocol_requires_new_confirmation_before_heldout_read() -> None:
    section = _section(PROTOCOL_PATH.read_text(encoding="utf-8"), "Failure And Branch Rules")
    assert "write and commit a separate confirmation document" in section
    assert "before reading calibration or outer_heldout" in section
    assert "not a detector-validation or AP improvement claim" in section
