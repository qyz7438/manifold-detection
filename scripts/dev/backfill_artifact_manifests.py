"""Deterministic generator for the backfilled reviewed artifact manifests (refactor Task 10).

Emits the 17 required reviewed manifests under
``spectral_detection_posttrain/configs/registry/artifacts/`` — one per
critical experiment artifact, grouped in six families:

* nwpu-mobile:       nwpu_mob_strong_cosine_s42_bs8_36ep
* zero-parity:       native_zero_parity_baseline, native_zero_parity_fullft
* set-policy:        det.energy.set_policy.m1.001
* native-actions:    det.energy.global_delta_u.c3b.balanced.001,
                     det.energy.native_topology.d2b.001,
                     det.energy.post_nms_suppress.e1.001
* dense-endpoint:    det.energy.dense_endpoint.absolute.001,
                     det.energy.dense_endpoint.geometry_control.001,
                     det.energy.dense_endpoint.cleanval.001,
                     det.energy.dense_endpoint.shift_audit.001
* dense-local-delta: det.energy.dense_local_delta_stats.002,
                     det.energy.dense_local_delta_learner.001,
                     det.energy.dense_local_delta_family_audit.001,
                     det.energy.dense_local_delta_family_prior.001
* 2026-07-21 closure: det.energy.re_roi_counterfactual_evidence.001,
                      det.energy.oracle_utility_boxhead.001

Evidence rules (plan Task 10 Steps 3-4):

* every value embedded in this file is COPIED from existing evidence:
  run result JSONs (``.agent_reports/c1_native/`` copies and their remote
  originals under ``runs/``), committed review reports under ``docs/reports/``,
  ``docs/autonomous_exploration_ledger.md``, committed version configs under
  ``spectral_detection_posttrain/configs/versions/``, and the committed split
  manifest under ``spectral_detection_posttrain/configs/splits/``;
* nothing is computed: no metric is re-derived, no status is re-judged, no
  hash is shortened; unknown values stay out of the manifest and are named in
  ``missing_evidence``;
* historical runs predate runtime manifests, so every manifest records the
  substitution explicitly: ``runtime_manifest_sha256`` binds the SHA-256 of
  the run's primary result JSON instead of a (nonexistent)
  ``runs/<run_id>/manifest.json``;
* repo-relative refs (version configs, split manifest, metric protocol
  module, environment snapshot) are re-hashed at generation time and the
  generator FAILS on drift rather than emitting a stale hash;
* remote/derived refs carry the SHA-256 (and size when observed) copied from
  the remote cross-check; absolute user paths never appear — remote
  artifacts use the ``remote:manifold`` alias (the host alias registered in
  ``configs/registry/environments/remote_gpu2_py310_cu121.json``), derived
  digests use the ``derived:`` alias, and the one checkpoint that lives in a
  workspace outside the permitted read scope uses ``remote:rlimage``.

Remote cross-check record (read-only SSH, 2026-07-13, plan Step 4):

* remote workspace revision ``7e1f2404699df997883678274491e88d3ec253c0``,
  ``git status --short`` clean; all 12 evidence git commits and both
  bundle-named commits are ancestors of that revision;
* SHA-256 and size of all 15 primary result JSONs, the strong run's
  ``config.json``, the NWPU annotation JSON, and both verifiable checkpoints
  match the local evidence copies exactly;
* no artifact was absent and none differed, so no manifest is marked
  ``invalid`` or ``unavailable``.

Determinism rules (enforced by ``tests/experiments/test_backfilled_artifacts.py``):

* no timestamps: the observation time is the fixed backfill constant below;
* UTF-8, LF line endings, exactly one trailing newline (via
  ``serialize_artifact_manifest``);
* two ``--write`` runs are byte-identical.

CLI:

* ``python scripts/dev/backfill_artifact_manifests.py --write``  emit the 17 manifests;
* ``python scripts/dev/backfill_artifact_manifests.py --check``  exit 1 on drift;
* ``--out-dir <dir>`` writes/compares against another directory (used by tests).

No experiment is authorized by this generator; manifests are provenance
records, not run requests.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
REGISTRY_DIR_REPO = "spectral_detection_posttrain/configs/registry"
ARTIFACTS_DIR_REPO = f"{REGISTRY_DIR_REPO}/artifacts"
DEFAULT_ARTIFACTS_DIR = ROOT / "spectral_detection_posttrain" / "configs" / "registry" / "artifacts"
GENERATOR_REPO_PATH = "scripts/dev/backfill_artifact_manifests.py"

sys.path.insert(0, str(ROOT))

from spectral_detection_posttrain.experiments.artifacts import (  # noqa: E402
    SCHEMA_VERSION,
    ArtifactManifest,
    ArtifactRef,
    EvaluationScope,
    serialize_artifact_manifest,
)

# ---------------------------------------------------------------------------
# Observation constants (fixed for determinism; see module docstring)
# ---------------------------------------------------------------------------

OBSERVED_AT_UTC = "2026-07-13T06:21:43Z"
OBSERVED_HOST_ALIAS = "remote:manifold"
OBSERVED_WORKSPACE_REVISION = "7e1f2404699df997883678274491e88d3ec253c0"

# The two terminal 2026-07-21 records were reviewed in a later bounded
# read-only remote observation. Per-spec overrides keep the original 15
# backfilled manifests byte-identical.
CLOSURE_OBSERVED_AT_UTC = "2026-07-20T16:36:35Z"
CLOSURE_OBSERVED_WORKSPACE_REVISION = "ce49b5747800e0a7b76e1f5be8e2b46f6500dbea"

REMOTE_RUNS = "remote:manifold/runs"
STRONG_RUN = f"{REMOTE_RUNS}/nwpu_mob_strong_cosine_s42_bs8_36ep"

# ---------------------------------------------------------------------------
# Shared missing-evidence wording
# ---------------------------------------------------------------------------

ME_RUNTIME_SUBSTITUTION = (
    "no runtime manifest.json exists for this historical run; "
    "runtime_manifest_sha256 binds the primary result JSON instead"
)
ME_INVOCATION = (
    "original CLI invocation not recorded in the evidence; entrypoints, where recorded, "
    "are documented in spectral_detection_posttrain/configs/registry/experiments.json"
)
ME_ENV_VERSIONS = (
    "run-time python/torch versions not recorded in the evidence; "
    "see spectral_detection_posttrain/configs/registry/environments/ snapshots"
)
ME_DERIVED_SPLIT = (
    "split manifest is a derived image-id digest recorded in the run evidence, "
    "not a standalone file"
)

# ---------------------------------------------------------------------------
# Shared refs
# ---------------------------------------------------------------------------

#: NWPU VHR-10 annotation JSON — sha256/size verified remotely 2026-07-13.
ANNOTATION_REF = (
    "dataset_annotation",
    "remote:manifold/data/NWPU_VHR10_coco.json",
    "dde27d9362d9c6aa358a1cab160c9e075e57405d3fe876de049973c6e6150f0e",
    1265416,
)

#: Seed-42 0.7 validation image-id digest (196 images) — a derived digest, not a file.
SPLIT_VAL_REF = (
    "split_manifest",
    "derived:nwpu_vhr10/seed42/validation_image_ids",
    "49f05cc9fa82ccaf924ff3be58a8f6376387c225e020ea6138182a4637219684",
    None,
)

#: Strong detector checkpoint (m1/cleanval initial weights; strong run output).
#: sha256/size verified remotely 2026-07-13.
STRONG_CHECKPOINT_REF = (
    "initial_checkpoint",
    f"{STRONG_RUN}/checkpoint_best.pth",
    "de126708d063a82dd5c721799cd124fc4e52d28866139e103b1cd65427742027",
    76205690,
)

VERSIONS_DIR_REPO = "spectral_detection_posttrain/configs/versions"


def _repo_ref(kind: str, repo_path: str, expected_sha256: str) -> ArtifactRef:
    """Repo-relative ref, re-hashed at generation time; fails loudly on drift."""
    path = ROOT / repo_path
    if not path.is_file():
        raise ValueError(f"repo-relative ref missing: {repo_path}")
    data = path.read_bytes()
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected_sha256:
        raise ValueError(
            f"sha256 drift for {repo_path}: expected {expected_sha256}, actual {actual}; "
            "update the evidence constant in this generator deliberately"
        )
    return ArtifactRef(
        semantic_kind=kind,
        logical_path=repo_path,
        sha256=expected_sha256,
        size_bytes=len(data),
    )


def _const_ref(spec: tuple[str, str, str, int | None]) -> ArtifactRef:
    kind, logical_path, sha256, size_bytes = spec
    return ArtifactRef(
        semantic_kind=kind,
        logical_path=logical_path,
        sha256=sha256,
        size_bytes=size_bytes,
    )


def _version_config_ref(version_id: str, sha256: str, kind: str = "resolved_config") -> ArtifactRef:
    return _repo_ref(kind, f"{VERSIONS_DIR_REPO}/{version_id}.json", sha256)


#: Detection metric protocol module — byte-identical between every evidence
#: commit that used it and the observed workspace revision (verified with
#: ``git diff <commit>..HEAD`` for dfaf494, da54b19, bf85749, 835503a, b0cc542).
#: Repo-relative: re-hashed when this module loads; drift fails loudly.
METRIC_PROTOCOL_REF = _repo_ref(
    "metric_protocol",
    "spectral_detection_posttrain/eval/detection_metrics.py",
    "f5b547d99420c8e0a0d59320671b0ec5eb05ecf279ec4d833f8aeca672a30848",
)

#: The strong run started from torchvision pretrained weights (run config
#: records pretrained=true); the weight file hash was never recorded, so the
#: initial_checkpoint ref identifies the weight source through the pinned
#: torchvision version in the remote environment snapshot. Repo-relative:
#: re-hashed when this module loads; drift fails loudly.
PRETRAINED_WEIGHT_SOURCE_REF = _repo_ref(
    "initial_checkpoint",
    "spectral_detection_posttrain/configs/registry/environments/remote_gpu2_py310_cu121.json",
    "d1256cf14f31ac27f31812197705b910985ead42262cb1b0d769d75d69d8570f",
)


# ---------------------------------------------------------------------------
# Manifest specifications (all values copied from evidence; see docstring)
# ---------------------------------------------------------------------------


def _eval_ref(run_dir: str, filename: str, sha256: str, size: int, kind: str = "eval_metrics") -> tuple:
    return (kind, f"{REMOTE_RUNS}/{run_dir}/{filename}", sha256, size)


def _specs() -> dict[str, dict[str, Any]]:
    specs: dict[str, dict[str, Any]] = {}

    # -- family 1: nwpu-mobile ---------------------------------------------
    strong_eval = _eval_ref(
        "nwpu_mob_strong_cosine_s42_bs8_36ep",
        "eval_metrics.json",
        "2de795a78dc4f29375f61e67a06ccdb0a0fa1e549ef20f398adb7132cbd3ebd6",
        15712,
    )
    strong_config = (
        "resolved_config",
        f"{STRONG_RUN}/config.json",
        "74f20c711826a32598b422d520735600275e4556a701320068ed85c5f2282341",
        2843,
    )
    specs["nwpu_mob_strong_cosine_s42_bs8_36ep"] = {
        "run_id": "nwpu_mob_strong_cosine_s42_bs8_36ep",
        "completion": "completed",
        "scope": ("full_val", 196, None, None),
        "git_commit": "dfaf494b8d8cafeace6d6661912c23304a9b8fdb",
        "git_dirty": False,
        "environment": {"host_alias": "remote:manifold"},
        "resolved_config": strong_config,
        "inputs": [
            ANNOTATION_REF,
            SPLIT_VAL_REF,
            PRETRAINED_WEIGHT_SOURCE_REF,
            METRIC_PROTOCOL_REF,
            ("postprocess_config", strong_config[1], strong_config[2], strong_config[3]),
        ],
        "outputs": [
            strong_eval,
            (
                "checkpoint",
                f"{STRONG_RUN}/checkpoint_best.pth",
                "de126708d063a82dd5c721799cd124fc4e52d28866139e103b1cd65427742027",
                76205690,
            ),
        ],
        "runtime_manifest_sha256": strong_eval[2],
        "metrics_summary": {
            "ap50": 0.6608888572398024,
            "ap75": 0.308384053391374,
            "precision": 0.5658796648895659,
            "recall": 0.7137367915465899,
            "false_positive_rate": 0.43412033511043413,
            "ece": 0.11224613767899444,
            "num_predictions": 1313,
            "num_gt": 1041,
            "best_ap50": 0.6721873811449312,
            "best_ap75": 0.31772923834372174,
            "best_epoch": 33,
            "selection_metric": "ap75",
        },
        "gates": {},
        "source_evidence_pointer": "docs/reports/nwpu_strong_native_candidate_energy_review_2026-07-10.md",
        "missing_evidence": [
            ME_RUNTIME_SUBSTITUTION,
            ME_INVOCATION,
            ME_ENV_VERSIONS,
            ME_DERIVED_SPLIT,
            (
                "torchvision pretrained weight file sha256 not recorded; the initial_checkpoint ref "
                "identifies the weight source by the pinned torchvision version in the remote environment "
                "snapshot (run config records pretrained=true)"
            ),
            (
                "run git_commit copied from the remote run config.json; the cited review report lists "
                "implementation commits but not the run commit"
            ),
            "no preregistered gates recorded in the evidence",
        ],
    }

    # -- family 2: zero-parity ----------------------------------------------
    parity_common_inputs_tail = [METRIC_PROTOCOL_REF]
    parity_cases = {
        "native_zero_parity_baseline": {
            "run_id": "det_action_zero_parity_nativefix_baseline_s42",
            "eval_sha": "d4db4d9f5a8e152fbac3ec5fb0946693fbee62d150723fb49d2e1a82532fff00",
            "eval_size": 5835,
            "initial_checkpoint": (
                "initial_checkpoint",
                "remote:rlimage/runs/nwpu_mob_baseline_s42_12ep/checkpoint_best.pth",
                "19845c4463d91b178c1289fc772a4bc0d34f8e9e0535951d7dd2b760fbb5f18a",
                None,
            ),
            "metrics": {
                "ap50": 0.5176863360079142,
                "ap75": 0.19165639069786988,
                "precision": 0.3305827746260959,
                "recall": 0.6157540826128722,
                "false_positive_rate": 0.669417225373904,
                "ece": 0.03679973743707088,
                "num_predictions": 1939,
                "num_gt": 1041,
            },
            "extra_missing": [
                (
                    "initial checkpoint bytes not re-verified: the RLimage workspace is outside the "
                    "permitted read scope; sha256 copied from the parity evidence JSON and "
                    "docs/reports/nwpu_c0_native_parity_review_2026-07-10.md"
                ),
            ],
        },
        "native_zero_parity_fullft": {
            "run_id": "det_action_zero_parity_nativefix_fullft18best_s42",
            "eval_sha": "5324392a0629f7b450ab3332bf7e22fd9631698fb1f1e71352fb8832e135c42e",
            "eval_size": 5884,
            "initial_checkpoint": (
                "initial_checkpoint",
                f"{REMOTE_RUNS}/ctrl_full_ft18_from12best_fullnw0_nwpu_s42_bs8_ep18/checkpoint_best.pth",
                "950bc89a8083ece77e39fd22c757462a66f63d3d028a94e61f42d6cd09eb9668",
                76205690,
            ),
            "metrics": {
                "ap50": 0.6022590778178841,
                "ap75": 0.28152208933069595,
                "precision": 0.42708968883465526,
                "recall": 0.6724303554274735,
                "false_positive_rate": 0.5729103111653447,
                "ece": 0.09222601517015794,
                "num_predictions": 1639,
                "num_gt": 1041,
            },
            "extra_missing": [],
        },
    }
    for experiment_id, case in parity_cases.items():
        run_id = case["run_id"]
        eval_ref = _eval_ref(run_id, "eval_metrics.json", case["eval_sha"], case["eval_size"])
        specs[experiment_id] = {
            "run_id": run_id,
            "completion": "completed",
            "scope": ("full_val", 196, None, None),
            "git_commit": "da54b19df7576d2dd2864a17bb4daaee650c1c48",
            "git_dirty": False,
            "environment": {"host_alias": "remote:manifold", "cuda_visible_devices": "2"},
            "resolved_config": ("resolved_config", eval_ref[1], eval_ref[2], eval_ref[3]),
            "inputs": [
                ANNOTATION_REF,
                SPLIT_VAL_REF,
                case["initial_checkpoint"],
                *parity_common_inputs_tail,
                ("postprocess_config", eval_ref[1], eval_ref[2], eval_ref[3]),
            ],
            "outputs": [eval_ref],
            "runtime_manifest_sha256": case["eval_sha"],
            "metrics_summary": {
                **case["metrics"],
                "strict_parity_images": 196,
                "strict_parity_mismatched_images": 0,
                "strict_parity_max_box_abs_error": 0.0,
                "strict_parity_max_score_abs_error": 0.0,
            },
            "gates": {
                "aggregate_zero_action_parity": True,
                "strict_zero_action_parity": True,
            },
            "source_evidence_pointer": "docs/reports/nwpu_c0_native_parity_review_2026-07-10.md",
            "missing_evidence": [
                ME_RUNTIME_SUBSTITUTION,
                ME_INVOCATION,
                ME_ENV_VERSIONS,
                (
                    "split digest copied from the strong-baseline run config (same deterministic "
                    "seed-42 0.7 split); the parity evidence JSON records no split digest"
                ),
                (
                    "the cited c0 review doc predates these runs (queued state); parity results are "
                    "reported in docs/reports/nwpu_strong_native_candidate_energy_review_2026-07-10.md "
                    "and the metrics bind the hashed result JSONs"
                ),
                "no standalone resolved config file; the resolved config is embedded in the result JSON",
                *case["extra_missing"],
            ],
        }

    # -- family 3: set-policy -------------------------------------------------
    m1_eval = _eval_ref(
        "nwpu_m1_set_policy_s42_fulltrain_fullval",
        "eval_metrics.json",
        "9b089b6625d0dfc9d9fe6a1553be6d5646b78a8179c7de15e506b2e0215a88e5",
        24080,
    )
    specs["det.energy.set_policy.m1.001"] = {
        "run_id": "nwpu_m1_set_policy_s42_fulltrain_fullval",
        "completion": "completed",
        "scope": ("full_val", 196, None, None),
        "git_commit": "bf8574970412ed111341ebe8d7532d75d7a740df",
        "git_dirty": False,
        "environment": {"host_alias": "remote:manifold"},
        "resolved_config": _version_config_ref(
            "det.energy.set_policy.m1.001",
            "37238720b1eaf6a0a56ace8644a4f3248be566b301c3e6ad7393764b596daf32",
        ),
        "inputs": [
            ANNOTATION_REF,
            SPLIT_VAL_REF,
            STRONG_CHECKPOINT_REF,
            METRIC_PROTOCOL_REF,
            _version_config_ref(
                "det.energy.set_policy.m1.001",
                "37238720b1eaf6a0a56ace8644a4f3248be566b301c3e6ad7393764b596daf32",
                kind="postprocess_config",
            ),
        ],
        "outputs": [m1_eval],
        "runtime_manifest_sha256": m1_eval[2],
        "metrics_summary": {
            "learned_set_ap50": 0.6628623111199616,
            "learned_set_ap75": 0.3134798624094865,
            "identity_ap75": 0.3132802297748511,
            "energy_zero_ap75": 0.3134764279263538,
            "shuffled_spatial_ap75": 0.30534972699870516,
            "learned_set_num_predictions": 1274,
            "num_gt": 1041,
            "scientific_status": "structural_gates_failed",
        },
        "gates": {
            "G0_parity": True,
            "G1_gt_free_eval": True,
            "G2_learned_vs_identity": False,
            "G3_learned_vs_shuffled": True,
            "G4_energy": True,
            "all_passed": False,
            "all_with_energy_passed": False,
        },
        "source_evidence_pointer": "docs/reports/nwpu_set_action_m0_m1_report_2026-07-10.md",
        "missing_evidence": [
            ME_RUNTIME_SUBSTITUTION,
            ME_INVOCATION,
            ME_ENV_VERSIONS,
            ME_DERIVED_SPLIT,
        ],
    }

    # -- family 4: native-actions ---------------------------------------------
    native_cases = {
        "det.energy.global_delta_u.c3b.balanced.001": {
            "run_id": "nwpu_c3b_balanced_smoke_s42",
            "config_sha": "c859e31f42faccbb0c87d0807c03dce0b79477389ff85c913363c27f5e7195d0",
            "eval_sha": "f404ea84770ac19c1054d78761feda0f29ad79e78a8ec34c7e4cf6a43e1e9e6a",
            "eval_size": 13303,
            "git_commit": "835503af4a9af2f37e098f1d1e1f95a9afa8bafd",
            "metrics_summary": {
                "identity_ap50": 0.7399366265892431,
                "identity_ap75": 0.4636476212171552,
                "learned_ap50": 0.7494837783323818,
                "learned_ap75": 0.45495986086517165,
                "ap75_delta_vs_identity": -0.008687760351983542,
                "ap50_delta_vs_identity": 0.009547151743138693,
                "support_images": 32,
                "scientific_status": "current_route_frozen",
            },
            "gates": {
                "native_parity": True,
                "non_degenerate": True,
                "detector_delta": False,
                "control_delta": False,
                "all_passed": False,
            },
        },
        "det.energy.native_topology.d2b.001": {
            "run_id": "nwpu_d2b_native_topology_smoke_s42",
            "config_sha": "1b07a3f3862cc49945b0a972a8ed96298c1924ced23638f0d09b7debb7836f92",
            "eval_sha": "c35e7e20492b3a39a985f2e8b09a77b0eed853bbacd60c3f9f929d2984989bf0",
            "eval_size": 13910,
            "git_commit": "6a3634d2ebce010ed3dad9b7d13f66e5087586cc",
            "metrics_summary": {
                "identity_ap50": 0.7399366265892431,
                "identity_ap75": 0.4636476212171552,
                "learned_ap50": 0.7397870997296508,
                "learned_ap75": 0.4477563296520475,
                "ap75_delta_vs_identity": -0.015891291565107712,
                "ap50_delta_vs_identity": -0.00014952685959224166,
                "support_images": 32,
                "scientific_status": "d2b_native_topology_frozen_move_d3",
            },
            "gates": {
                "cache_alignment": True,
                "native_parity": True,
                "non_degenerate": True,
                "detector_delta": False,
                "control_delta": False,
                "all_passed": False,
            },
        },
        "det.energy.post_nms_suppress.e1.001": {
            "run_id": "nwpu_e1_post_nms_suppress_smoke_s42",
            "config_sha": "72405a2fdaf19b1b5d9b3ae28b57e21c5c33343e177f8e181273ba4d53ef25bb",
            "eval_sha": "f2e1a65d2fe54f8e85db8db9fc7ab629654514b211f856b04480e3dabf2e7e46",
            "eval_size": 12451,
            "git_commit": "35ece286a1d356f9624f0dc702ec9cde5c4ae408",
            "metrics_summary": {
                "identity_ap50": 0.7399366265892431,
                "identity_ap75": 0.4636476212171552,
                "learned_ap50": 0.6201309300139223,
                "learned_ap75": 0.362689001207016,
                "learned_num_predictions": 162,
                "support_images": 32,
                "scientific_status": "post_nms_suppress_frozen",
            },
            "gates": {
                "candidate_support": True,
                "label_support": True,
                "native_parity": True,
                "non_degenerate": False,
                "detector_delta": False,
                "safety": False,
                "control_delta": True,
                "all_passed": False,
            },
        },
    }
    for experiment_id, case in native_cases.items():
        run_id = case["run_id"]
        eval_ref = _eval_ref(run_id, "eval_metrics.json", case["eval_sha"], case["eval_size"])
        specs[experiment_id] = {
            "run_id": run_id,
            "completion": "completed",
            "scope": ("smoke", 64, 32, 32),
            "git_commit": case["git_commit"],
            "git_dirty": False,
            "environment": {"host_alias": "remote:manifold"},
            "resolved_config": _version_config_ref(experiment_id, case["config_sha"]),
            "inputs": [METRIC_PROTOCOL_REF],
            "outputs": [eval_ref],
            "runtime_manifest_sha256": case["eval_sha"],
            "metrics_summary": case["metrics_summary"],
            "gates": case["gates"],
            "source_evidence_pointer": "docs/autonomous_exploration_ledger.md",
            "missing_evidence": [
                ME_RUNTIME_SUBSTITUTION,
                ME_INVOCATION,
                ME_ENV_VERSIONS,
            ],
        }

    # -- family 5: dense-endpoint ---------------------------------------------
    specs["det.energy.dense_endpoint.absolute.001"] = {
        "run_id": "nwpu_dense_endpoint_absolute_s42_nested",
        "completion": "completed",
        "scope": ("train_only", 454, None, None),
        "git_commit": "52c826859e5edaecbf7f6121fff7bc2feb6816ee",
        "git_dirty": False,
        "environment": {"host_alias": "remote:manifold", "cuda_visible_devices": "2"},
        "resolved_config": _version_config_ref(
            "det.energy.dense_endpoint.absolute.001",
            "e69f42129ec6aa409299c64ecbb4b8eecdb82365004f333249fa4db92e8798fa",
        ),
        "inputs": [
            ANNOTATION_REF,
            _repo_ref(
                "split_manifest",
                "spectral_detection_posttrain/configs/splits/nwpu_dense_endpoint_s42_nested.json",
                "ce19316aeaef1cbdf85f2c9668c5ae2c8443da8e848de2d22ed0f0f3a687e080",
            ),
            STRONG_CHECKPOINT_REF,
        ],
        "outputs": [
            _eval_ref(
                "nwpu_dense_endpoint_absolute_s42_nested",
                "eval_metrics.json",
                "d0e4afb68bd9c6f7636672e501c3638400efe82c91ec24ef5c68740b30043c2f",
                13959,
            )
        ],
        "runtime_manifest_sha256": "d0e4afb68bd9c6f7636672e501c3638400efe82c91ec24ef5c68740b30043c2f",
        "metrics_summary": {
            "tune_count": 68,
            "tune_mae": 0.30195393714615526,
            "tune_rmse": 0.45398632106863723,
            "tune_pearson": 0.9793149711176107,
            "tune_pairwise_accuracy": 0.885864794254303,
            "tune_auroc_positive": 0.9721011519432068,
            "scientific_status": "reduced_absolute_endpoint_frozen",
        },
        "gates": {
            "outer_support": True,
            "outer_pairwise": True,
            "outer_auroc": True,
            "pairwise_controls": False,
            "mae_controls": False,
            "constant_baseline": True,
            "fit_outer_gap": True,
            "all_passed": False,
        },
        "source_evidence_pointer": "docs/autonomous_exploration_ledger.md",
        "missing_evidence": [
            ME_RUNTIME_SUBSTITUTION,
            ME_INVOCATION,
            ME_ENV_VERSIONS,
        ],
    }

    specs["det.energy.dense_endpoint.geometry_control.001"] = {
        "run_id": "nwpu_dense_endpoint_geometry_control_s42_resplit",
        "completion": "completed",
        "scope": ("train_only", 318, None, None),
        "git_commit": "c29264184acafb17d436272cd143e079080ac2a0",
        "git_dirty": False,
        "environment": {"host_alias": "remote:manifold"},
        "resolved_config": _version_config_ref(
            "det.energy.dense_endpoint.geometry_control.001",
            "612dfd2be6cc57767a4234267a5ed3f078d84048129905bc2f86256b3e0b04b4",
        ),
        "inputs": [],
        "outputs": [
            _eval_ref(
                "nwpu_dense_endpoint_geometry_control_s42_resplit",
                "eval_metrics.json",
                "ca5fbbef4ee19faa35a925eec6245bac456bc90525c76426e250a24f69b1171d",
                7247,
            )
        ],
        "runtime_manifest_sha256": "ca5fbbef4ee19faa35a925eec6245bac456bc90525c76426e250a24f69b1171d",
        "metrics_summary": {
            "fit_mae": 0.2716106102603862,
            "fit_pearson": 0.9808445975701344,
            "fit_pairwise_accuracy": 0.9279201030731201,
            "fit_auroc_positive": 0.9944541454315186,
            "holdout_mae": 0.42828662757528946,
            "holdout_pearson": 0.9723083847372462,
            "holdout_pairwise_accuracy": 0.8784722089767456,
            "holdout_auroc_positive": 0.984375,
            "scientific_status": "geometry_endpoint_signal_supported_train_only_resplit",
        },
        "gates": {
            "holdout_support": True,
            "full_pairwise": True,
            "full_auroc": True,
            "constant_baseline": True,
            "pairwise_controls": True,
            "mae_controls": True,
            "fit_holdout_gap": True,
            "all_passed": True,
        },
        "source_evidence_pointer": "docs/autonomous_exploration_ledger.md",
        "missing_evidence": [
            ME_RUNTIME_SUBSTITUTION,
            ME_INVOCATION,
            ME_ENV_VERSIONS,
        ],
    }

    specs["det.energy.dense_endpoint.cleanval.001"] = {
        "run_id": "nwpu_dense_endpoint_cleanval_s42",
        "completion": "completed",
        "scope": ("detector_unseen", 196, None, None),
        "git_commit": "b0cc54282b50005f81e1e1a11028d5ef7bbec6a7",
        "git_dirty": False,
        "environment": {"host_alias": "remote:manifold", "cuda_visible_devices": "2"},
        "resolved_config": _version_config_ref(
            "det.energy.dense_endpoint.cleanval.001",
            "82f56e7772042904c6da3dbb8559e630a351a35ccba5582421aeb1d0d8f8d05e",
        ),
        "inputs": [
            ANNOTATION_REF,
            SPLIT_VAL_REF,
            STRONG_CHECKPOINT_REF,
            _repo_ref(
                "metric_protocol",
                "scripts/validate_dense_endpoint_cleanval.py",
                "066a5b11313c97325c3ddf57d73619cb28e0c9f556f980229c5a3c0c101a0c57",
            ),
            _version_config_ref(
                "det.energy.dense_endpoint.cleanval.001",
                "82f56e7772042904c6da3dbb8559e630a351a35ccba5582421aeb1d0d8f8d05e",
                kind="postprocess_config",
            ),
        ],
        "outputs": [
            _eval_ref(
                "nwpu_dense_endpoint_cleanval_s42",
                "eval_metrics.json",
                "024f9ed8cd89a52ac743d35b1bb6a54717772dfd51b2027e47ad30de51f80f8f",
                8660,
            )
        ],
        "runtime_manifest_sha256": "024f9ed8cd89a52ac743d35b1bb6a54717772dfd51b2027e47ad30de51f80f8f",
        "metrics_summary": {
            "train_mae": 0.25106445890297446,
            "train_pearson": 0.9874941779103765,
            "train_pairwise_accuracy": 0.9261701107025146,
            "train_auroc_positive": 0.9843994379043579,
            "validation_count": 196,
            "validation_mae": 1.7698389031563182,
            "validation_pearson": 0.8846051876117843,
            "validation_pairwise_accuracy": 0.8092621564865112,
            "validation_auroc_positive": 0.9357954263687134,
            "scientific_status": "detector_unseen_absolute_endpoint_frozen",
        },
        "gates": {
            "validation_support": True,
            "full_pairwise": True,
            "full_auroc": True,
            "constant_baseline": True,
            "pairwise_controls": True,
            "mae_controls": True,
            "train_validation_gap": False,
            "all_passed": False,
        },
        "source_evidence_pointer": "docs/autonomous_exploration_ledger.md",
        "missing_evidence": [
            ME_RUNTIME_SUBSTITUTION,
            ME_INVOCATION,
            ME_ENV_VERSIONS,
            ME_DERIVED_SPLIT,
        ],
    }

    specs["det.energy.dense_endpoint.shift_audit.001"] = {
        "run_id": "nwpu_dense_endpoint_shift_audit_s42",
        "completion": "completed",
        "scope": ("cache_only", None, None, None),
        "git_commit": "b831dcba5bcc812f3d92dedd040c3d5703a78995",
        "git_dirty": False,
        "environment": {"host_alias": "remote:manifold"},
        "resolved_config": _version_config_ref(
            "det.energy.dense_endpoint.shift_audit.001",
            "daeb0ee3229e5554ce034d7293ed3bb4fa5ee4cbfb830e0dcbe2c1a88e9b2d69",
        ),
        "inputs": [],
        "outputs": [
            _eval_ref(
                "nwpu_dense_endpoint_shift_audit_s42",
                "eval_metrics.json",
                "62fc164491e325926873771a1941da0fd8f2a94cdb231b48fa12033d66096ce1",
                10044,
            )
        ],
        "runtime_manifest_sha256": "62fc164491e325926873771a1941da0fd8f2a94cdb231b48fa12033d66096ce1",
        "metrics_summary": {
            "train_mae": 0.2510644601255918,
            "validation_mae": 1.7698389530295924,
            "validation_pearson": 0.8846051851990766,
            "validation_pairwise_accuracy": 0.8092621564865112,
            "validation_residual_mean": 1.707392930984497,
            "oracle_centered_mae_relative_gain": 0.3984907222011253,
            "scientific_status": "calibration_shift_without_feature_ood",
        },
        "gates": {},
        "source_evidence_pointer": "docs/autonomous_exploration_ledger.md",
        "missing_evidence": [
            ME_RUNTIME_SUBSTITUTION,
            ME_INVOCATION,
            ME_ENV_VERSIONS,
            (
                "evaluation_scope.image_count is null by construction for this cache-only audit; "
                "the evidence JSON records explicit counts (454 train / 196 validation) but the "
                "audit consumes cached predictions rather than running an evaluation pass"
            ),
            "evidence records no gates: cache-only post-hoc shift diagnosis",
        ],
    }

    # -- family 6: dense-local-delta --------------------------------------------
    specs["det.energy.dense_local_delta_stats.002"] = {
        "run_id": "nwpu_dense_local_delta_stats_v2_s42_train64",
        "completion": "completed",
        "scope": ("train_only", 64, None, None),
        "git_commit": "84bd1f7ebecf105bf684866ed11ac0e7ccebe42a",
        "git_dirty": False,
        "environment": {"host_alias": "remote:manifold"},
        "resolved_config": _version_config_ref(
            "det.energy.dense_local_delta_stats.002",
            "86b2d70495c8174b22fa325ae76ac0410414ce47da643b81bc8376c4cd26ffc6",
        ),
        "inputs": [],
        "outputs": [
            _eval_ref(
                "nwpu_dense_local_delta_stats_v2_s42_train64",
                "eval_metrics.json",
                "10804a48fae1b3d81d7ebac1b51660524f2008f57a333a4da15ed185fe4d3010",
                218338,
            )
        ],
        "runtime_manifest_sha256": "10804a48fae1b3d81d7ebac1b51660524f2008f57a333a4da15ed185fe4d3010",
        "metrics_summary": {
            "nonidentity_count": 1557,
            "nonidentity_mean": -0.08224122039495214,
            "nonidentity_std": 0.28378792244414314,
            "nonidentity_positive_fraction": 0.27167630057803466,
            "identity_max_abs_delta_q": 0.0,
            "scientific_status": "shift_invariant_local_delta_learner_warranted",
        },
        "gates": {
            "image_support": True,
            "identity": True,
            "nonzero": True,
            "positive": True,
            "negative": True,
            "unique": True,
            "magnitude": True,
            "all_passed": True,
        },
        "source_evidence_pointer": "docs/autonomous_exploration_ledger.md",
        "missing_evidence": [
            ME_RUNTIME_SUBSTITUTION,
            ME_INVOCATION,
            ME_ENV_VERSIONS,
        ],
    }

    specs["det.energy.dense_local_delta_learner.001"] = {
        "run_id": "nwpu_dense_local_delta_learner_s42_48_16",
        "completion": "completed",
        "scope": ("limited", 64, 48, 16),
        "git_commit": "58908126bcd79738ebd6a226c419b4e33ed65989",
        "git_dirty": False,
        "environment": {"host_alias": "remote:manifold"},
        "resolved_config": _version_config_ref(
            "det.energy.dense_local_delta_learner.001",
            "6bfee0169539dc32f0e90736f0cb1de7a59db4928e9af22cd79f16461882185b",
        ),
        "inputs": [],
        "outputs": [
            _eval_ref(
                "nwpu_dense_local_delta_learner_s42_48_16",
                "eval_metrics.json",
                "a7570b9e89e733174272cd06956e6c2646b3fc73274c46e8b04ed71d5f969f30",
                6478,
            )
        ],
        "runtime_manifest_sha256": "a7570b9e89e733174272cd06956e6c2646b3fc73274c46e8b04ed71d5f969f30",
        "metrics_summary": {
            "fit_mae": 0.06011749088131901,
            "tune_mae": 0.08584910252801894,
            "tune_zero_baseline_mae": 0.10997341155268,
            "tune_imagewise_pairwise_accuracy": 0.5746907591819763,
            "tune_sign_auroc": 0.6901063323020935,
            "scientific_status": "train_only_local_delta_learner_frozen",
        },
        "gates": {
            "tune_support": True,
            "identity": True,
            "pairwise": False,
            "sign": True,
            "zero_baseline": True,
            "pairwise_controls": True,
            "mae_controls": True,
            "fit_tune_gap": True,
            "all_passed": False,
        },
        "source_evidence_pointer": "docs/autonomous_exploration_ledger.md",
        "missing_evidence": [
            ME_RUNTIME_SUBSTITUTION,
            ME_INVOCATION,
            ME_ENV_VERSIONS,
        ],
    }

    family_cases = {
        "det.energy.dense_local_delta_family_audit.001": {
            "config_sha": "cb412b9c6f6ffffae2a2b681f5472defac93f0c588fff73656688333cc3cf490",
            "result_file": "family_audit_metrics.json",
            "result_sha": "4c0a150e97135d44c3eca5b6caeaf74c28a298b6ed0ccd1314f0f5747a399b53",
            "result_size": 167160,
            "git_commit": "923db6a7d361a85369e032aed81da0a69ce8acba",
            "bundle_name": "family_audit_923db6a.bundle",
            "scope": ("cache_only", 16, None, None),
            "metrics_summary": {
                "candidate_pairs": 5301,
                "eligible_pairs": 5237,
                "tune_images": 16,
                "image_equal_accuracy": 0.5714652188617558,
                "candidate_weighted_accuracy": 0.5739927439373688,
                "scientific_status": "posthoc_family_diagnostic_complete",
            },
        },
        "det.energy.dense_local_delta_family_prior.001": {
            "config_sha": "d2723fe0fba3526035c35c32aa3206c27863f2a5edda0f2aa72c1362ecd489c5",
            "result_file": "family_prior_metrics.json",
            "result_sha": "2d578fd55f64f28f5fb2d363dc4d8df7be8997aa9487ae12ca9caa8e016481c6",
            "result_size": 3711,
            "git_commit": "d81b27a662b673cd8720d5cea4e95f2acf45590f",
            "bundle_name": "family_prior_d81b27a.bundle",
            "scope": ("cache_only", 64, 48, 16),
            "metrics_summary": {
                "local_full_mae": 0.0858491068471961,
                "family_fit_mean_mae": 0.07722179204898162,
                "global_fit_mean_mae": 0.12307347376619417,
                "zero_baseline_mae": 0.10997341155268,
                "family_fit_mean_pairwise_accuracy": 0.6724072098731995,
                "scientific_status": "posthoc_family_prior_diagnostic_complete",
            },
        },
    }
    for experiment_id, case in family_cases.items():
        run_id = "nwpu_dense_local_delta_learner_s42_48_16"
        eval_ref = _eval_ref(run_id, case["result_file"], case["result_sha"], case["result_size"])
        specs[experiment_id] = {
            "run_id": run_id,
            "completion": "completed",
            "scope": case["scope"],
            "git_commit": case["git_commit"],
            "git_dirty": False,
            "environment": {"host_alias": "remote:manifold"},
            "resolved_config": _version_config_ref(experiment_id, case["config_sha"]),
            "inputs": [],
            "outputs": [eval_ref],
            "runtime_manifest_sha256": case["result_sha"],
            "metrics_summary": case["metrics_summary"],
            "gates": {},
            "source_evidence_pointer": "docs/autonomous_exploration_ledger.md",
            "missing_evidence": [
                ME_RUNTIME_SUBSTITUTION,
                ME_INVOCATION,
                ME_ENV_VERSIONS,
                (
                    f"run-time git commit copied from the sibling evidence bundle name "
                    f"({case['bundle_name']}); the result JSON records no git commit"
                ),
                "evidence records no gates: post-hoc family diagnostic",
            ],
        }

    # -- family 7: terminal re-ROI / AWR closure --------------------------
    closure_report = "docs/reports/nwpu_re_roi_and_awr_closure_2026-07-21.md"
    remote_split = (
        "split_manifest",
        "remote:manifold/spectral_detection_posttrain/configs/splits/nwpu_re_roi_counterfactual_s42_nested.json",
        "5c4222e033f7ed333835eab7f7786498b84bba11856c2d362ec8895845942244",
        9286,
    )
    fit_cache = (
        "action_cache",
        f"{REMOTE_RUNS}/nwpu_re_roi_cache_fit_s42_strongbest_e58806b_v4/re_roi_cache_fit.pt",
        "7fcb3774467aeae7b6592e20485ce93565946c1b96fc4cd7469ce63ddae288f7",
        20724910,
    )
    tune_cache = (
        "action_cache",
        f"{REMOTE_RUNS}/nwpu_re_roi_cache_tune_s42_strongbest_e58806b_v1/re_roi_cache_tune.pt",
        "6110a25c6f666c3d86e1ce4635ce8d66046c3483452d5541a273171234a07db7",
        5599416,
    )
    calibration_cache = (
        "action_cache",
        f"{REMOTE_RUNS}/nwpu_re_roi_cache_calibration_s42_strongbest_e58806b_v1/re_roi_cache_calibration.pt",
        "6421f83c2b6dd68e2862bfc8702586c7e042ebf01a7300d773c68f34bee13c8c",
        5491662,
    )
    re_roi_run_id = "nwpu_re_roi_evidence_s42_strongbest_ce49b57_v1"
    re_roi_run = f"{REMOTE_RUNS}/{re_roi_run_id}"
    re_roi_runtime_manifest = (
        "runtime_manifest",
        f"{re_roi_run}/manifest.json",
        "d011653bc9997b2d0f5a0d206229417cf1c76c6a8bc5baa0b3db46f85d154591",
        4939,
    )
    specs["det.energy.re_roi_counterfactual_evidence.001"] = {
        "run_id": re_roi_run_id,
        "completion": "completed",
        "scope": ("train_only", 386, None, None),
        "git_commit": "ce49b5747800e0a7b76e1f5be8e2b46f6500dbea",
        "git_dirty": False,
        "environment": {
            "host_alias": "remote:manifold",
            "cuda_visible_devices": "2",
        },
        "resolved_config": (
            "resolved_config",
            re_roi_runtime_manifest[1],
            re_roi_runtime_manifest[2],
            re_roi_runtime_manifest[3],
        ),
        "inputs": [
            ANNOTATION_REF,
            STRONG_CHECKPOINT_REF,
            remote_split,
            fit_cache,
            tune_cache,
            calibration_cache,
            _repo_ref(
                "protocol_document",
                "docs/re_roi_counterfactual_evidence_protocol.md",
                "1d7ee89219a595c73fc9dbc93cfb6f9e2ce447060dd85cac75749dd3ff1282e7",
            ),
        ],
        "outputs": [
            re_roi_runtime_manifest,
            (
                "eval_metrics",
                f"{re_roi_run}/eval_metrics.json",
                "4ac0da6a065bc58486006326abbcf32fbe25ca4df0152d76ce075a9511affd85",
                13436,
            ),
            (
                "model_checkpoint",
                f"{re_roi_run}/arm_B_final.pt",
                "9d6985798bcbfcbb54fc7ad092679097c8a6abb881e0bdaa2a23f437b7d1df47",
                640185,
            ),
            (
                "model_checkpoint",
                f"{re_roi_run}/arm_C_final.pt",
                "68fc54c68e2a56982da49c8fd8789c2e58c7aa385ceaafa73c4e8c8ceed3ac2f",
                902329,
            ),
            (
                "model_checkpoint",
                f"{re_roi_run}/arm_D_final.pt",
                "f2880626ec7595a8c0aba7a71056a3f2d590d50f5f50ce22577e1d327722373e",
                902329,
            ),
        ],
        "runtime_manifest_sha256": re_roi_runtime_manifest[2],
        "metrics_summary": {
            "scientific_status": "train_only_re_roi_mechanism_frozen",
            "all_gates_passed": False,
            "outer_heldout_read": False,
            "detector_validation_read": False,
            "fit_images": 250,
            "tune_images": 68,
            "calibration_images": 68,
            "tune_b_residual_pairwise_accuracy": 0.5782024132091781,
            "tune_c_residual_pairwise_accuracy": 0.5382165670268348,
            "tune_d_residual_pairwise_accuracy": 0.5258315711770595,
            "c_vs_b_point_delta": -0.04144620522856712,
            "c_vs_b_lcb": -0.08438427746295929,
            "c_vs_d_point_delta": 0.009758192114531994,
            "c_vs_d_lcb": -0.029235413298010826,
            "tune_c_reconstructed_relative_mae_gain": 0.28023859693289943,
            "calibration_positive_rate": 0.029411764705882353,
            "calibration_coverage": 1.0,
            "calibration_correction": 268.4031982421875,
            "generalization_residual_mae_gain_gap": 0.2728413826718951,
        },
        "gates": {
            "support": True,
            "identity": True,
            "re_roi_gain": False,
            "bundle_integrity": False,
            "static_baseline": True,
            "calibration": False,
            "generalization": False,
            "all_passed": False,
        },
        "source_evidence_pointer": closure_report,
        "missing_evidence": [],
        "observed_at_utc": CLOSURE_OBSERVED_AT_UTC,
        "observed_workspace_revision": CLOSURE_OBSERVED_WORKSPACE_REVISION,
    }

    awr_run_id = "nwpu_oracle_utility_boxhead_strongbest_e58806b_s42_Z"
    awr_log = (
        "launcher_log",
        f"{REMOTE_RUNS}/{awr_run_id}_launcher.log",
        "ab95634556eba177c1c6ec39c478171dba6b14daf8c7fbddb88a2236739159c5",
        591,
    )
    specs["det.energy.oracle_utility_boxhead.001"] = {
        "run_id": awr_run_id,
        "completion": "completed",
        "scope": ("cache_only", 250, None, None),
        "git_commit": "e58806b7ca4c1a21f66d2505512cbf4d9cf601c3",
        "git_dirty": False,
        "environment": {
            "host_alias": "remote:manifold",
            "cuda_visible_devices": "2",
        },
        "resolved_config": _version_config_ref(
            "det.energy.oracle_utility_boxhead.001",
            "7764bfedc3d941aa27477c7cedde98118a9df864a1aa752e89c9c0a6b269259c",
        ),
        "inputs": [
            ANNOTATION_REF,
            STRONG_CHECKPOINT_REF,
            remote_split,
            fit_cache,
            _repo_ref(
                "protocol_document",
                "docs/awr_weighted_boxhead_protocol.md",
                "970590126411bb9bb2e26ea173dc911497ba95ae7aeff1ff4baa42ea41fb7038",
            ),
        ],
        "outputs": [awr_log],
        "runtime_manifest_sha256": awr_log[2],
        "metrics_summary": {
            "scientific_status": "support_gate_blocked_before_training",
            "fit_images": 250,
            "positive_images": 62,
            "positive_image_rate": 0.248,
            "positive_candidates": 93,
            "minimum_positive_candidates": 500,
            "effective_sample_size": 197.0039,
            "weight_saturation": 0.0,
            "normalized_mean_weight": 1.0,
            "training_started": False,
            "downstream_arms_started": False,
            "outer_heldout_read": False,
            "detector_validation_read": False,
        },
        "gates": {
            "positive_image_support": True,
            "positive_candidate_support": False,
            "support": False,
            "weight_health": True,
            "training_not_started": True,
            "all_passed": False,
        },
        "source_evidence_pointer": closure_report,
        "missing_evidence": [
            "no runtime manifest.json exists for this support preflight; runtime_manifest_sha256 binds the launcher log instead",
            "the original CLI invocation and exact cache argument were not serialized in the launcher log",
            "numeric preflight diagnostics survive in the reviewed handoff and closure report, not in the launcher log",
            "no training, detector evaluation, or AP artifact exists because the support gate failed before training",
        ],
        "observed_at_utc": CLOSURE_OBSERVED_AT_UTC,
        "observed_workspace_revision": CLOSURE_OBSERVED_WORKSPACE_REVISION,
    }

    return specs


# ---------------------------------------------------------------------------
# Build / write / check
# ---------------------------------------------------------------------------


def build_manifest(experiment_id: str, spec: dict[str, Any]) -> ArtifactManifest:
    kind, image_count, limit_train, limit_val = spec["scope"]
    return ArtifactManifest(
        schema_version=SCHEMA_VERSION,
        manifest_kind="reviewed",
        manifest_id=f"{experiment_id}:{spec['run_id']}",
        experiment_id=experiment_id,
        run_id=spec["run_id"],
        completion=spec["completion"],
        invocation=tuple(spec.get("invocation", ())),
        resolved_config=(
            spec["resolved_config"]
            if isinstance(spec["resolved_config"], ArtifactRef)
            else _const_ref(spec["resolved_config"])
        ),
        inputs=tuple(
            ref if isinstance(ref, ArtifactRef) else _const_ref(ref) for ref in spec["inputs"]
        ),
        outputs=tuple(_const_ref(ref) for ref in spec["outputs"]),
        git_commit=spec["git_commit"],
        git_dirty=spec["git_dirty"],
        environment=dict(spec["environment"]),
        evaluation_scope=EvaluationScope(
            kind=kind,
            image_count=image_count,
            limit_train=limit_train,
            limit_val=limit_val,
        ),
        metrics_summary=dict(spec["metrics_summary"]),
        gates=dict(spec["gates"]),
        runtime_manifest_sha256=spec["runtime_manifest_sha256"],
        observed_at_utc=spec.get("observed_at_utc", OBSERVED_AT_UTC),
        observed_host_alias=OBSERVED_HOST_ALIAS,
        observed_workspace_revision=spec.get(
            "observed_workspace_revision", OBSERVED_WORKSPACE_REVISION
        ),
        source_evidence_pointer=spec["source_evidence_pointer"],
        missing_evidence=tuple(spec["missing_evidence"]),
        unavailable_reason=None,
        supersedes=None,
    )


def generate_manifests() -> dict[str, str]:
    """Return ``{file_name: content}`` for the 17 reviewed manifests."""
    specs = _specs()
    documents = {}
    for experiment_id in sorted(specs):
        manifest = build_manifest(experiment_id, specs[experiment_id])
        documents[f"{experiment_id}.json"] = serialize_artifact_manifest(manifest)
    return documents


def write_manifests(out_dir: Path) -> dict[str, str]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    documents = generate_manifests()
    for name, content in documents.items():
        (out_dir / name).write_text(content, encoding="utf-8", newline="\n")
    return documents


def check_drift(artifacts_dir: Path = DEFAULT_ARTIFACTS_DIR) -> tuple[bool, str]:
    artifacts_dir = Path(artifacts_dir)
    documents = generate_manifests()
    problems = []
    for name, content in documents.items():
        path = artifacts_dir / name
        if not path.is_file():
            problems.append(f"missing reviewed manifest: {name}")
            continue
        if path.read_text(encoding="utf-8") != content:
            problems.append(f"drift: {name}")
    extra = sorted(
        path.name for path in artifacts_dir.glob("*.json") if path.name not in documents
    ) if artifacts_dir.is_dir() else []
    for name in extra:
        problems.append(f"unexpected manifest not produced by the generator: {name}")
    if problems:
        return False, (
            "; ".join(problems)
            + f" — regenerate with `python {GENERATOR_REPO_PATH} --write`"
        )
    return True, "backfilled artifact manifests are up to date"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--write", action="store_true", help="emit the 17 reviewed manifests")
    group.add_argument("--check", action="store_true", help="exit 1 when shipped manifests drift")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help=f"write/compare against this directory instead of {ARTIFACTS_DIR_REPO}/",
    )
    args = parser.parse_args(argv)

    out_dir = args.out_dir if args.out_dir is not None else DEFAULT_ARTIFACTS_DIR
    if args.write:
        documents = write_manifests(out_dir)
        for name in sorted(documents):
            size = len(documents[name].encode("utf-8"))
            print(f"wrote {out_dir / name} ({size} bytes)")
        return 0
    ok, message = check_drift(artifacts_dir=out_dir)
    print(message)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
