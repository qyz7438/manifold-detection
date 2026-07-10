"""Frozen-detector M0 oracle evaluation for NWPU set coordination.

M0 is deliberately a GT-oracle reachability experiment.  Ground truth builds
the bounded candidate pool and scores candidate sets after native detector
postprocessing.  It is not a learned policy or generalization result.
"""

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, dataclass, is_dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spectral_detection_posttrain.datasets import build_nwpu_vhr10_loaders
from spectral_detection_posttrain.eval.detection_metrics import evaluate_detection_predictions
from spectral_detection_posttrain.methods.energy_transport import (
    ActionCandidate,
    ROIActionState,
    ROITransportActions,
    SetOutcome,
    beam_search,
    build_candidate_quality_targets,
    build_symmetric_box_candidates,
    deterministic_delta_permutation,
    greedy_positive_marginal_selection,
    paired_bootstrap_summary,
    set_outcome_from_prediction,
)
from spectral_detection_posttrain.models import build_detector
from spectral_detection_posttrain.trainers.detection.action_local_transport import (
    ProposalActionBatch,
    action_batch_to_predictions,
    extract_proposal_action_batch,
)
from spectral_detection_posttrain.utils.git_state import get_git_state
from spectral_detection_posttrain.utils.io import load_checkpoint
from spectral_detection_posttrain.utils.seed import resolve_device, set_seed


LOCKED_CONFIG = ROOT / "spectral_detection_posttrain" / "configs" / "versions" / "det.energy.set_oracle.m0.001.json"
LOCKED_CONFIG_SHA256 = "b2c485fc485bb605bffc776329beebbf42b556a5b79ed1123d22954465702f0e"
MODES = (
    "identity",
    "local",
    "set_greedy",
    "set_beam",
    "delta_permuted_beam",
    "local_nms_off",
    "set_beam_nms_off",
)
LOCKED_UTILITY = {
    "tp75_weight": 1.0,
    "fp75_weight": -0.25,
    "fp50_weight": -0.1,
    "action_energy_weight": -0.05,
    "action_count_weight": -0.02,
}


@dataclass(frozen=True)
class LocalCandidate:
    """One oracle action retained for a proposal in an image-local pool."""

    proposal_index: int
    candidate_index: int
    box_delta: torch.Tensor
    local_value: float
    quality: float
    base_quality: float
    action_energy: float
    action_id: str

    @property
    def candidate(self) -> ActionCandidate:
        return ActionCandidate(self.action_id, action_energy=self.action_energy)

    @property
    def delta(self) -> torch.Tensor:
        return self.box_delta

    @property
    def utility(self) -> float:
        return self.local_value


@dataclass(frozen=True)
class ParitySummary:
    mismatched_images: int = 0
    max_box_abs_error: float = 0.0
    max_score_abs_error: float = 0.0
    max_label_mismatch: int = 0

    @property
    def passed(self) -> bool:
        return (
            self.mismatched_images == 0
            and self.max_box_abs_error <= 0.0001
            and self.max_score_abs_error <= 0.000002
            and self.max_label_mismatch == 0
        )


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_locked_config(config: dict[str, Any]) -> None:
    if config.get("version_id") != "det.energy.set_oracle.m0.001":
        raise ValueError("M0 runner requires the locked det.energy.set_oracle.m0.001 config")
    if config.get("status") != "preregistered":
        raise ValueError("M0 config must remain preregistered")
    if tuple(config.get("modes", ())) != MODES:
        raise ValueError("locked M0 modes do not match the runner")
    if config.get("set_utility") != LOCKED_UTILITY:
        raise ValueError("locked M0 utility does not match the implemented SetOutcome utility")


def load_locked_config(path: str | Path = LOCKED_CONFIG) -> dict[str, Any]:
    config_path = Path(path).resolve()
    if config_path != LOCKED_CONFIG.resolve():
        raise ValueError(f"M0 requires the canonical locked config: {LOCKED_CONFIG}")
    actual_hash = sha256_file(config_path)
    if actual_hash != LOCKED_CONFIG_SHA256:
        raise ValueError(
            f"canonical M0 config SHA256 mismatch: expected {LOCKED_CONFIG_SHA256}, got {actual_hash}"
        )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_locked_config(config)
    return config


def _resolved_path(value: str | Path, *, base: Path = ROOT) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path)


def verify_locked_hashes(
    config: dict[str, Any],
    checkpoint: str | Path,
    annotation: str | Path,
) -> dict[str, str]:
    """Verify the two immutable inputs named by the preregistered config."""

    checkpoint_path = Path(checkpoint)
    annotation_path = Path(annotation)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"locked checkpoint does not exist: {checkpoint_path}")
    if not annotation_path.is_file():
        raise FileNotFoundError(f"locked annotation does not exist: {annotation_path}")
    actual_checkpoint = sha256_file(checkpoint_path)
    actual_annotation = sha256_file(annotation_path)
    expected_checkpoint = str(config["detector"]["checkpoint_sha256"])
    expected_annotation = str(config["dataset"]["annotation_sha256"])
    if actual_checkpoint != expected_checkpoint:
        raise ValueError(
            f"checkpoint SHA256 mismatch: expected {expected_checkpoint}, got {actual_checkpoint}"
        )
    if actual_annotation != expected_annotation:
        raise ValueError(
            f"annotation SHA256 mismatch: expected {expected_annotation}, got {actual_annotation}"
        )
    return {
        "checkpoint_sha256": actual_checkpoint,
        "annotation_sha256": actual_annotation,
    }


def candidate_id(proposal_index: int, candidate_index: int) -> str:
    """Return a stable, sortable ID that carries both local indices."""

    if proposal_index < 0 or candidate_index < 0:
        raise ValueError("proposal and candidate indices must be non-negative")
    return f"p{int(proposal_index):04d}_c{int(candidate_index):02d}"


def _candidate_energy(delta: torch.Tensor, action_energy_scale: float) -> float:
    if action_energy_scale <= 0.0:
        raise ValueError("action_energy_scale must be positive")
    return float(delta.detach().pow(2).sum().item()) / float(action_energy_scale)


def build_local_candidate_pool(
    candidate_deltas: torch.Tensor,
    candidate_quality: torch.Tensor,
    foreground_dominant: torch.Tensor,
    *,
    min_iou_gain: float = 0.002,
    boundary_iou: float = 0.75,
    boundary_temperature: float = 0.05,
    action_energy_scale: float = 0.04,
    local_energy_weight: float = 0.05,
    local_move_cost: float = 0.02,
    max_candidates: int = 12,
) -> tuple[LocalCandidate, ...]:
    """Keep the best positive boundary-weighted action for each proposal."""

    if candidate_deltas.ndim != 2 or candidate_deltas.shape[1] != 4:
        raise ValueError("candidate_deltas must have shape (K, 4)")
    if candidate_deltas.shape[0] == 0 or not torch.allclose(
        candidate_deltas[0], torch.zeros_like(candidate_deltas[0])
    ):
        raise ValueError("candidate index 0 must be identity")
    if candidate_quality.ndim != 2 or candidate_quality.shape[1] != candidate_deltas.shape[0]:
        raise ValueError("candidate_quality must have shape (N, K)")
    if foreground_dominant.shape != (candidate_quality.shape[0],):
        raise ValueError("foreground_dominant must have shape (N,)")
    if boundary_temperature <= 0.0:
        raise ValueError("boundary_temperature must be positive")
    if action_energy_scale < 0.0 or local_energy_weight < 0.0 or local_move_cost < 0.0:
        raise ValueError("candidate utility weights must be non-negative")
    if max_candidates < 0:
        raise ValueError("max_candidates must be non-negative")

    base = candidate_quality[:, 0]
    base_sigmoid = torch.sigmoid((base - float(boundary_iou)) / float(boundary_temperature))
    candidates: list[LocalCandidate] = []
    for proposal_index in range(candidate_quality.shape[0]):
        if not bool(foreground_dominant[proposal_index].item()):
            continue
        eligible = (candidate_quality[proposal_index] - base[proposal_index]) > float(min_iou_gain)
        eligible[0] = False
        best: LocalCandidate | None = None
        for candidate_index in torch.nonzero(eligible, as_tuple=False).flatten().tolist():
            quality = float(candidate_quality[proposal_index, candidate_index].item())
            base_quality = float(base[proposal_index].item())
            delta = candidate_deltas[candidate_index].detach().clone()
            energy = _candidate_energy(delta, action_energy_scale)
            sigmoid_gain = float(
                torch.sigmoid(
                    (candidate_quality[proposal_index, candidate_index] - float(boundary_iou))
                    / float(boundary_temperature)
                ).item()
                - base_sigmoid[proposal_index].item()
            )
            local_value = sigmoid_gain - float(local_energy_weight) * energy - float(local_move_cost)
            if local_value <= 0.0:
                continue
            item = LocalCandidate(
                proposal_index=proposal_index,
                candidate_index=int(candidate_index),
                box_delta=delta,
                local_value=local_value,
                quality=quality,
                base_quality=base_quality,
                action_energy=energy,
                action_id=candidate_id(proposal_index, int(candidate_index)),
            )
            if best is None or (item.local_value, -item.action_energy, -item.candidate_index) > (
                best.local_value,
                -best.action_energy,
                -best.candidate_index,
            ):
                best = item
        if best is not None:
            candidates.append(best)
    candidates.sort(key=lambda item: (-item.local_value, item.action_energy, item.action_id))
    return tuple(candidates[: int(max_candidates)])


def _zero_actions(state: ROIActionState) -> ROITransportActions:
    return ROITransportActions(
        feature_delta=torch.zeros_like(state.features),
        score_delta=torch.zeros_like(state.scores),
        box_delta=torch.zeros_like(state.boxes),
        keep_logit=torch.zeros_like(state.scores),
    )


def actions_from_selection(
    state: ROIActionState,
    selection: Iterable[LocalCandidate | ActionCandidate],
) -> ROITransportActions:
    """Materialize only the selected proposal-local box actions."""

    actions = _zero_actions(state)
    by_id = {
        item.action_id: item
        for item in selection
        if isinstance(item, LocalCandidate)
    }
    if not by_id:
        return actions
    proposal_rows = {int(item.proposal_index): item for item in by_id.values()}
    for proposal_index, item in proposal_rows.items():
        rows = torch.nonzero(state.proposal_indices == proposal_index, as_tuple=False).flatten()
        # The callback is image-local; proposal indices are therefore unique.
        if rows.numel() == 0:
            continue
        row = int(rows[0].item())
        actions.box_delta[row] = item.box_delta.to(device=state.boxes.device, dtype=state.boxes.dtype)
    return actions


def permute_pool_deltas(
    pool: Sequence[LocalCandidate],
    *,
    seed: int,
) -> tuple[LocalCandidate, ...]:
    """Break action-location alignment while preserving candidate identities."""

    if not pool:
        return ()
    permutation = deterministic_delta_permutation(
        tuple(float(index) for index in range(len(pool))),
        seed=int(seed),
    )
    return tuple(
        replace(
            item,
            box_delta=pool[int(source)].box_delta.clone(),
            action_energy=pool[int(source)].action_energy,
        )
        for item, source in zip(pool, permutation)
    )


def selected_pool_items(
    pool: Sequence[LocalCandidate],
    action_ids: Sequence[str],
) -> tuple[LocalCandidate, ...]:
    by_id = {item.action_id: item for item in pool}
    missing = [action_id for action_id in action_ids if action_id not in by_id]
    if missing:
        raise ValueError(f"unknown action id(s): {missing}")
    return tuple(by_id[action_id] for action_id in action_ids)


def _selection_items(
    selection: Sequence[ActionCandidate],
    by_id: dict[str, LocalCandidate],
) -> tuple[LocalCandidate, ...]:
    return tuple(by_id[action.action_id] for action in selection if action.action_id in by_id)


def _outcome_with_actions(outcome: SetOutcome, actions: Sequence[ActionCandidate]) -> SetOutcome:
    return replace(
        outcome,
        action_energy=sum(float(action.action_energy) for action in actions),
        action_count=len(actions),
    )


def evaluate_preregistered_gates(summary: dict[str, Any], locked: dict[str, Any]) -> dict[str, Any]:
    """Evaluate G0-G5 without silently dropping an absent condition."""

    gates_cfg = locked["gates"]
    parity = summary.get("strict_parity", {})
    g0 = bool(
        parity.get("passed", True)
        and int(parity.get("mismatched_images", -1)) <= int(gates_cfg["G0_strict_zero_parity"]["mismatched_images"])
        and float(parity.get("max_box_abs_error", float("inf"))) <= float(gates_cfg["G0_strict_zero_parity"]["max_box_abs_error"])
        and float(parity.get("max_score_abs_error", float("inf"))) <= float(gates_cfg["G0_strict_zero_parity"]["max_score_abs_error"])
    )
    g1 = float(summary.get("coverage_fraction_two_candidates", 0.0)) >= float(
        gates_cfg["G1_set_problem"]["min_fraction_images_with_two_candidates"]
    )
    coordination = summary.get("beam_minus_local", {})
    nms_coordination = summary.get("beam_minus_local_nms_off_sum", 0.0)
    g2 = bool(
        float(coordination.get("sum", float("-inf"))) >= float(gates_cfg["G2_coordination"]["min_sum_beam_minus_local_utility"])
        and int(coordination.get("improved_images", -1)) >= int(gates_cfg["G2_coordination"]["min_improved_images"])
        and float((coordination.get("bootstrap_ci95") or [float("-inf")])[0]) > float(gates_cfg["G2_coordination"]["paired_bootstrap_lower_bound_gt"])
    )
    native_uplift = float(summary.get("native_to_nms_off_uplift_ratio", float("-inf")))
    if "native_to_nms_off_uplift_ratio" not in summary:
        native_uplift = abs(float(coordination.get("sum", 0.0))) / max(abs(float(nms_coordination)), 1e-12)
    g3 = native_uplift >= float(gates_cfg["G3_nms_specific"]["min_native_to_nms_off_uplift_ratio"])
    permuted = summary.get("beam_minus_permuted", {})
    g4 = float((permuted.get("bootstrap_ci95") or [float("-inf")])[0]) > float(
        gates_cfg["G4_location_alignment"]["beam_minus_permuted_bootstrap_lower_bound_gt"]
    )
    detector = summary.get("detector_deltas", {})
    g5_cfg = gates_cfg["G5_detector_headroom"]
    g5 = bool(
        float(detector.get("ap75", float("-inf"))) >= float(g5_cfg["min_ap75_delta_vs_identity"])
        and float(detector.get("ap50", float("-inf"))) >= float(g5_cfg["min_ap50_delta_vs_identity"])
        and float(detector.get("false_positive_rate", float("inf"))) <= float(g5_cfg["max_fpr_delta_vs_identity"])
        and float(detector.get("num_predictions_relative", float("inf"))) <= float(g5_cfg["max_prediction_relative_delta_vs_identity"])
    )
    gates = {"G0_strict_zero_parity": g0, "G1_set_problem": g1, "G2_coordination": g2, "G3_nms_specific": g3, "G4_location_alignment": g4, "G5_detector_headroom": g5}
    return {"all_passed": all(gates.values()), "gates": gates}


def _json_value(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return _json_value(asdict(value))
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return str(value)


def _prediction_summary(prediction: dict[str, torch.Tensor]) -> dict[str, Any]:
    return {key: _json_value(value) for key, value in prediction.items()}


def validation_manifest(loader) -> dict[str, Any]:
    image_ids = getattr(loader.dataset, "img_ids", None)
    if image_ids is None:
        return {"count": len(loader.dataset)}
    normalized = sorted(int(image_id) for image_id in image_ids)
    encoded = json.dumps(normalized, separators=(",", ":")).encode("ascii")
    return {
        "count": len(normalized),
        "image_ids_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def verify_validation_manifest(manifest: dict[str, Any], locked: dict[str, Any]) -> None:
    dataset = locked["dataset"]
    expected = {
        "count": int(dataset["validation_images"]),
        "image_ids_sha256": str(dataset["val_image_ids_sha256"]),
    }
    if manifest != expected:
        raise ValueError(f"validation split mismatch: expected {expected}, got {manifest}")


def _compare_predictions(expected: dict[str, torch.Tensor], actual: dict[str, torch.Tensor]) -> tuple[float, float, int, bool]:
    count_mismatch = abs(int(expected["boxes"].shape[0]) - int(actual["boxes"].shape[0]))
    common = min(expected["boxes"].shape[0], actual["boxes"].shape[0])
    box_error = float((expected["boxes"][:common] - actual["boxes"][:common]).abs().max().item()) if common else 0.0
    score_error = float((expected["scores"][:common] - actual["scores"][:common]).abs().max().item()) if common else 0.0
    label_mismatch = int((expected["labels"][:common] != actual["labels"][:common]).sum().item()) if common else 0
    numeric_mismatch = int(box_error > 1e-4 or score_error > 2e-6)
    mismatch = count_mismatch + label_mismatch + numeric_mismatch
    return box_error, score_error, mismatch, mismatch == 0


def _as_prediction_dict(prediction: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu() if torch.is_tensor(value) else value for key, value in prediction.items()}


def _make_model_config(locked: dict[str, Any], *, num_classes: int = 11) -> dict[str, Any]:
    detector = locked["detector"]
    return {
        "seed": int(locked["dataset"]["seed"]),
        "data_seed": int(locked["dataset"]["data_seed"]),
        "data": {"root": str(ROOT / "data" / "NWPU VHR-10 dataset"), "annotation": str(ROOT / "data" / "NWPU_VHR10_coco.json"), "max_size": int(detector["max_size"]), "train_fraction": float(locked["dataset"]["train_fraction"]), "num_workers": 0},
        "model": {"name": detector["model_name"], "model_name": detector["model_name"], "pretrained": False, "num_classes": num_classes, "min_size": int(detector["min_size"]), "max_size": int(detector["max_size"])},
        "train": {"batch_size": 1},
        "eval": {"batch_size": 1},
    }


def build_frozen_detector(locked: dict[str, Any], checkpoint: str | Path, device: torch.device) -> torch.nn.Module:
    model = build_detector(_make_model_config(locked)).to(device)
    load_checkpoint(model, checkpoint, device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _candidate_pool_for_batch(batch: ProposalActionBatch, locked: dict[str, Any]) -> tuple[LocalCandidate, ...]:
    pool_cfg = locked["candidate_pool"]
    deltas = build_symmetric_box_candidates(tuple(float(value) for value in pool_cfg["step_sizes"]))
    if int(deltas.shape[0]) != int(pool_cfg["candidate_count"]):
        raise ValueError(f"locked candidate builder produced {deltas.shape[0]} actions, expected {pool_cfg['candidate_count']}")
    quality = build_candidate_quality_targets(
        batch.state,
        deltas.to(device=batch.state.boxes.device, dtype=batch.state.boxes.dtype),
        matched_gt_boxes=batch.matched_gt_boxes,
        matched_gt_labels=batch.matched_gt_labels,
        image_sizes=batch.image_sizes,
        min_iou_gain=float(pool_cfg["min_iou_gain"]),
    ).candidate_quality
    foreground = batch.state.scores >= float(pool_cfg["min_score"])
    if bool(pool_cfg.get("require_foreground_dominant", True)) and batch.state.logits is not None and batch.state.logits.shape[1] > 1:
        probabilities = torch.softmax(batch.state.logits, dim=-1)
        foreground = foreground & (probabilities[:, 1:].amax(dim=1) > probabilities[:, 0])
    return build_local_candidate_pool(
        deltas,
        quality,
        foreground,
        min_iou_gain=float(pool_cfg["min_iou_gain"]),
        boundary_iou=float(pool_cfg["boundary_iou"]),
        boundary_temperature=float(pool_cfg["boundary_temperature"]),
        action_energy_scale=float(pool_cfg["action_energy_scale"]),
        local_energy_weight=float(pool_cfg["local_energy_weight"]),
        local_move_cost=float(pool_cfg["local_move_cost"]),
        max_candidates=int(pool_cfg["max_candidates_per_image"]),
    )


def _run_search(
    batch: ProposalActionBatch,
    target: dict[str, torch.Tensor],
    pool: tuple[LocalCandidate, ...],
    model: torch.nn.Module,
    locked: dict[str, Any],
    *,
    nms_threshold: float,
    permute_deltas: bool = False,
    permutation_seed: int | None = None,
) -> tuple[dict[str, Any], dict[str, torch.Tensor], dict[str, SetOutcome]]:
    search_cfg = locked["search"]
    by_id = {item.action_id: item for item in pool}
    if permute_deltas and pool:
        pool = permute_pool_deltas(
            pool,
            seed=int(search_cfg["permutation_seed"] if permutation_seed is None else permutation_seed),
        )
        by_id = {item.action_id: item for item in pool}
    candidates = tuple(item.candidate for item in pool)
    cache: dict[tuple[float, tuple[str, ...]], tuple[SetOutcome, dict[str, torch.Tensor]]] = {}

    def evaluate(actions: tuple[ActionCandidate, ...]) -> SetOutcome:
        key = (float(nms_threshold), tuple(action.action_id for action in actions))
        if key not in cache:
            selected = _selection_items(actions, by_id)
            materialized = actions_from_selection(batch.state, selected)
            prediction = action_batch_to_predictions(
                batch,
                materialized,
                score_threshold=float(locked["detector"]["score_threshold"]),
                nms_threshold=float(nms_threshold),
                detections_per_img=int(locked["detector"]["detections_per_image"]),
                native_model=model,
            )[0]
            outcome = set_outcome_from_prediction(prediction, target, score_threshold=float(locked["detector"]["score_threshold"]))
            cache[key] = (_outcome_with_actions(outcome, actions), prediction)
        return cache[key][0]

    identity_actions: tuple[ActionCandidate, ...] = ()
    local_items = tuple(sorted(pool, key=lambda item: (-item.local_value, item.action_id))[: int(search_cfg["action_budget"])])
    local_result = tuple(item.candidate for item in local_items)
    greedy = greedy_positive_marginal_selection(candidates, evaluate, max_actions=int(search_cfg["action_budget"]))
    beam = beam_search(candidates, evaluate, beam_width=int(search_cfg["beam_width"]), max_depth=int(search_cfg["beam_depth"]))
    local_outcome = evaluate(local_result)
    identity_outcome = evaluate(identity_actions)
    selected = {
        "identity": identity_actions,
        "local": local_result,
        "set_greedy": greedy.actions,
        "set_beam": beam.actions,
        "delta_permuted_beam": beam.actions if permute_deltas else beam.actions,
    }
    predictions: dict[str, dict[str, torch.Tensor]] = {}
    outcomes: dict[str, SetOutcome] = {}
    for mode, actions in selected.items():
        key = (float(nms_threshold), tuple(action.action_id for action in actions))
        result = cache.get(key)
        if result is None:
            evaluate(actions)
            result = cache[key]
        outcomes[mode] = result[0]
        predictions[mode] = result[1]
    return {
        "selected": {mode: [action.action_id for action in actions] for mode, actions in selected.items()},
        "cache_evaluations": len(cache),
        "candidate_count": len(pool),
        "identity_utility": identity_outcome.utility,
        "local_utility": local_outcome.utility,
    }, predictions, outcomes


def _evaluate_fixed_selection(
    batch: ProposalActionBatch,
    target: dict[str, torch.Tensor],
    pool: Sequence[LocalCandidate],
    action_ids: Sequence[str],
    model: torch.nn.Module,
    locked: dict[str, Any],
    *,
    nms_threshold: float,
) -> tuple[dict[str, torch.Tensor], SetOutcome]:
    selected = selected_pool_items(pool, action_ids)
    action_candidates = tuple(item.candidate for item in selected)
    prediction = action_batch_to_predictions(
        batch,
        actions_from_selection(batch.state, selected),
        score_threshold=float(locked["detector"]["score_threshold"]),
        nms_threshold=float(nms_threshold),
        detections_per_img=int(locked["detector"]["detections_per_image"]),
        native_model=model,
    )[0]
    outcome = set_outcome_from_prediction(
        prediction,
        target,
        score_threshold=float(locked["detector"]["score_threshold"]),
    )
    return prediction, _outcome_with_actions(outcome, action_candidates)


def run(args: argparse.Namespace) -> dict[str, Any]:
    locked = load_locked_config(args.config)
    checkpoint = _resolved_path(args.checkpoint or locked["detector"]["checkpoint"])
    annotation = _resolved_path(args.annotation or ROOT / "data" / "NWPU_VHR10_coco.json")
    hashes = verify_locked_hashes(locked, checkpoint, annotation)
    set_seed(int(locked["dataset"]["seed"]))
    device = resolve_device({"device": args.device} if args.device else {"device": "cuda" if torch.cuda.is_available() else "cpu"})
    model = build_frozen_detector(locked, checkpoint, device)
    config = _make_model_config(locked)
    config["data"]["root"] = str(_resolved_path(args.data_root or config["data"]["root"]))
    config["data"]["annotation"] = str(annotation)
    _, loader = build_nwpu_vhr10_loaders(config, limit_val=args.limit_val, batch_size=1)
    if args.require_full_validation and args.limit_val is not None:
        raise ValueError("--require-full-validation cannot be combined with --limit-val")
    split_manifest = validation_manifest(loader)
    if args.limit_val is None:
        verify_validation_manifest(split_manifest, locked)

    mode_set = tuple(args.mode or MODES)
    unknown = set(mode_set) - set(MODES)
    if unknown:
        raise ValueError(f"unsupported M0 mode(s): {sorted(unknown)}")
    predictions_by_mode = {mode: [] for mode in mode_set}
    targets: list[dict[str, torch.Tensor]] = []
    records: list[dict[str, Any]] = []
    parity = ParitySummary()
    beam_local_utilities: list[float] = []
    beam_local_nms_off_utilities: list[float] = []
    beam_permuted_utilities: list[float] = []
    two_candidate_images = 0

    with torch.no_grad():
        for image_index, (images, batch_targets) in enumerate(tqdm(loader, desc="M0 set oracle")):
            images_device = [image.to(device) for image in images]
            target = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch_targets[0].items()}
            batch = extract_proposal_action_batch(model, images_device, [target], box_base="decoded")
            native_prediction = _as_prediction_dict(model(images_device)[0])
            identity_prediction = action_batch_to_predictions(
                batch,
                _zero_actions(batch.state),
                score_threshold=float(locked["detector"]["score_threshold"]),
                nms_threshold=float(locked["detector"]["nms_threshold"]),
                detections_per_img=int(locked["detector"]["detections_per_image"]),
                native_model=model,
            )[0]
            box_error, score_error, mismatch, same = _compare_predictions(native_prediction, identity_prediction)
            parity = ParitySummary(
                mismatched_images=parity.mismatched_images + (0 if same else 1),
                max_box_abs_error=max(parity.max_box_abs_error, box_error),
                max_score_abs_error=max(parity.max_score_abs_error, score_error),
                max_label_mismatch=max(parity.max_label_mismatch, mismatch),
            )
            pool = _candidate_pool_for_batch(batch, locked)
            if len(pool) >= 2:
                two_candidate_images += 1
            search_record, searched_predictions, searched_outcomes = _run_search(batch, target, pool, model, locked, nms_threshold=float(locked["detector"]["nms_threshold"]))
            image_id = int(target["image_id"].flatten()[0].item())
            perm_record, perm_predictions, perm_outcomes = _run_search(
                batch,
                target,
                pool,
                model,
                locked,
                nms_threshold=float(locked["detector"]["nms_threshold"]),
                permute_deltas=True,
                permutation_seed=int(locked["search"]["permutation_seed"]) + image_id,
            )
            selected_predictions = dict(searched_predictions)
            selected_outcomes = dict(searched_outcomes)
            local_nms_prediction, local_nms_outcome = _evaluate_fixed_selection(
                batch,
                target,
                pool,
                search_record["selected"]["local"],
                model,
                locked,
                nms_threshold=float(locked["search"]["nms_off_threshold"]),
            )
            beam_nms_prediction, beam_nms_outcome = _evaluate_fixed_selection(
                batch,
                target,
                pool,
                search_record["selected"]["set_beam"],
                model,
                locked,
                nms_threshold=float(locked["search"]["nms_off_threshold"]),
            )
            nms_record = {
                "policy": "fixed_native_selected_actions",
                "local": search_record["selected"]["local"],
                "set_beam": search_record["selected"]["set_beam"],
                "postprocess_evaluations": 2,
            }
            selected_predictions["local_nms_off"] = local_nms_prediction
            selected_predictions["set_beam_nms_off"] = beam_nms_prediction
            selected_outcomes["local_nms_off"] = local_nms_outcome
            selected_outcomes["set_beam_nms_off"] = beam_nms_outcome
            selected_predictions["delta_permuted_beam"] = perm_predictions["set_beam"]
            selected_outcomes["delta_permuted_beam"] = perm_outcomes["set_beam"]
            for mode in mode_set:
                predictions_by_mode[mode].append(selected_predictions[mode])
            targets.append({key: value.detach().cpu() if torch.is_tensor(value) else value for key, value in target.items()})
            beam_local_utilities.append(float(selected_outcomes["set_beam"].utility - selected_outcomes["local"].utility))
            beam_local_nms_off_utilities.append(float(beam_nms_outcome.utility - local_nms_outcome.utility))
            beam_permuted_utilities.append(float(selected_outcomes["set_beam"].utility - selected_outcomes["delta_permuted_beam"].utility))
            records.append({
                "image_index": image_index,
                "image_id": image_id,
                "candidate_pool": [{key: _json_value(value) for key, value in item.__dict__.items()} for item in pool],
                "search": search_record,
                "nms_off_search": nms_record,
                "permuted_search": perm_record,
                "outcomes": {mode: _json_value(outcome) for mode, outcome in selected_outcomes.items()},
                "strict_parity": {"box_abs_error": box_error, "score_abs_error": score_error, "same": same},
            })

    metrics = {
        mode: evaluate_detection_predictions(
            predictions_by_mode[mode],
            targets,
            score_threshold=float(locked["detector"]["score_threshold"]),
            per_class=True,
            num_classes=11,
            per_size=True,
        )
        for mode in mode_set
    }
    identity_metrics = metrics.get("identity") or metrics[mode_set[0]]
    beam_metrics = metrics.get("set_beam", identity_metrics)
    detector_deltas = {
        "ap75": float(beam_metrics["ap75"] - identity_metrics["ap75"]),
        "ap50": float(beam_metrics["ap50"] - identity_metrics["ap50"]),
        "false_positive_rate": float(beam_metrics["false_positive_rate"] - identity_metrics["false_positive_rate"]),
        "num_predictions_relative": float((beam_metrics["num_predictions"] - identity_metrics["num_predictions"]) / max(1, identity_metrics["num_predictions"])),
    }
    bootstrap = paired_bootstrap_summary(beam_local_utilities, [0.0] * len(beam_local_utilities), num_resamples=int(locked["search"]["bootstrap_repetitions"]), seed=int(locked["search"]["bootstrap_seed"])) if beam_local_utilities else None
    bootstrap_perm = paired_bootstrap_summary([0.0] * len(beam_permuted_utilities), [-value for value in beam_permuted_utilities], num_resamples=int(locked["search"]["bootstrap_repetitions"]), seed=int(locked["search"]["bootstrap_seed"])) if beam_permuted_utilities else None
    summary = {
        "strict_parity": _json_value(parity.__dict__ | {"passed": parity.passed}),
        "coverage_fraction_two_candidates": two_candidate_images / max(1, len(records)),
        "beam_minus_local": {"sum": sum(beam_local_utilities), "improved_images": sum(value > 0.0 for value in beam_local_utilities), "bootstrap_ci95": [bootstrap.ci_low, bootstrap.ci_high] if bootstrap else []},
        "beam_minus_local_nms_off_sum": sum(beam_local_nms_off_utilities),
        "native_to_nms_off_uplift_ratio": abs(sum(beam_local_utilities))
        / max(abs(sum(beam_local_nms_off_utilities)), 1e-12),
        "beam_minus_permuted": {"bootstrap_ci95": [bootstrap_perm.ci_low, bootstrap_perm.ci_high] if bootstrap_perm else []},
        "detector_deltas": detector_deltas,
        "paired_bootstrap": _json_value(bootstrap.__dict__) if bootstrap else None,
        "paired_bootstrap_beam_minus_permuted": _json_value(bootstrap_perm.__dict__) if bootstrap_perm else None,
    }
    result = {
        "completed": True,
        "version_id": locked["version_id"],
        "eval_scope": "full_val" if args.limit_val is None else "smoke",
        "config": locked,
        "inputs": {
            "checkpoint": str(checkpoint.resolve()),
            "annotation": str(annotation.resolve()),
            **hashes,
            "device": str(device),
            "frozen_detector": True,
            "validation_images": len(records),
            "validation_manifest": split_manifest,
            "git_state": get_git_state(ROOT),
        },
        "metrics": _json_value(metrics),
        "summary": _json_value(summary),
        "gates": evaluate_preregistered_gates(summary, locked),
        "per_image": _json_value(records),
    }
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=LOCKED_CONFIG)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--annotation", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "runs" / "det.energy.set_oracle.m0.001" / "m0_results.json")
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit-val", type=int)
    parser.add_argument("--require-full-validation", action="store_true")
    parser.add_argument("--require-clean-git", action="store_true")
    parser.add_argument("--mode", action="append", choices=MODES)
    args = parser.parse_args(argv)
    if args.require_clean_git:
        import subprocess

        status = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, check=True, capture_output=True, text=True)
        if status.stdout.strip():
            raise RuntimeError("--require-clean-git requested but the repository is dirty")
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(_json_value(result), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "gates": result["gates"], "validation_images": result["inputs"]["validation_images"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
