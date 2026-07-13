"""Image-balanced diagnostics for fixed-step sparse box actions."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch


@dataclass(frozen=True)
class StepActionRows:
    target: torch.Tensor
    image_ids: torch.Tensor
    family: tuple[str, ...]
    direction: tuple[str, ...]
    class_ids: torch.Tensor
    scale_bin: tuple[str, ...]


def extract_step_action_rows(
    records: Sequence[tuple[dict[str, Any], dict[str, Any]]],
    *,
    step: float,
    small_area_max: float = 0.02,
    medium_area_max: float = 0.10,
) -> StepActionRows:
    if not records or step <= 0.0 or not 0.0 < small_area_max < medium_area_max:
        raise ValueError("records, step, and scale thresholds must be valid")
    targets: list[torch.Tensor] = []
    image_ids: list[torch.Tensor] = []
    families: list[str] = []
    directions: list[str] = []
    classes: list[torch.Tensor] = []
    scales: list[str] = []
    for detector, trace in records:
        deltas = torch.as_tensor(trace["box_deltas"], dtype=torch.float32)
        if deltas.numel() == 0:
            continue
        mask = torch.isclose(
            deltas.abs().amax(dim=1),
            torch.tensor(float(step)),
            atol=1e-6,
            rtol=0.0,
        )
        if not mask.any():
            continue
        proposal_indices = torch.as_tensor(
            trace["proposal_indices"], dtype=torch.long
        )[mask]
        selected_deltas = deltas[mask]
        selected_targets = torch.as_tensor(
            trace["singleton_delta_u"], dtype=torch.float32
        )[mask]
        labels = torch.as_tensor(detector["predicted_labels"], dtype=torch.long)[
            proposal_indices
        ]
        boxes = torch.as_tensor(detector["boxes"], dtype=torch.float32)[
            proposal_indices
        ]
        image_h, image_w = (float(value) for value in detector["image_size"])
        widths = (boxes[:, 2] - boxes[:, 0]).clamp_min(0.0)
        heights = (boxes[:, 3] - boxes[:, 1]).clamp_min(0.0)
        area_fraction = widths * heights / (image_h * image_w)
        targets.append(selected_targets)
        image_ids.append(
            torch.full(
                (selected_targets.numel(),),
                int(detector["image_id"]),
                dtype=torch.long,
            )
        )
        classes.append(labels)
        for delta, area in zip(selected_deltas, area_fraction):
            has_xy = bool(delta[:2].abs().gt(0.0).any())
            has_wh = bool(delta[2:].abs().gt(0.0).any())
            families.append(
                "joint" if has_xy and has_wh else "translation" if has_xy else "scale"
            )
            signs = torch.sign(delta).to(torch.int64).tolist()
            directions.append(",".join(f"{value:+d}" if value else "0" for value in signs))
            area_value = float(area.item())
            scales.append(
                "small"
                if area_value <= small_area_max
                else "medium"
                if area_value <= medium_area_max
                else "large"
            )
    if not targets:
        raise ValueError("no actions match the requested step")
    rows = StepActionRows(
        target=torch.cat(targets),
        image_ids=torch.cat(image_ids),
        family=tuple(families),
        direction=tuple(directions),
        class_ids=torch.cat(classes),
        scale_bin=tuple(scales),
    )
    _validate_rows(rows)
    return rows


def strata_masks(rows: StepActionRows) -> dict[str, torch.Tensor]:
    _validate_rows(rows)
    masks: dict[str, torch.Tensor] = {}
    string_axes = {"family": rows.family, "direction": rows.direction, "scale": rows.scale_bin}
    for axis, values in string_axes.items():
        for value in sorted(set(values)):
            masks[f"{axis}={value}"] = torch.tensor(
                [item == value for item in values], dtype=torch.bool
            )
    for value in torch.unique(rows.class_ids, sorted=True).tolist():
        masks[f"class={int(value)}"] = rows.class_ids.eq(int(value)).cpu()
    return masks


def summarize_stratum(
    rows: StepActionRows,
    mask: torch.Tensor,
    *,
    manifest_image_ids: Sequence[int],
    target_epsilon: float,
    lcb_z: float,
) -> dict[str, float | int]:
    _validate_rows(rows)
    selected = torch.as_tensor(mask, dtype=torch.bool).flatten().cpu()
    if selected.shape != rows.target.shape:
        raise ValueError("stratum mask shape mismatch")
    manifest = sorted(set(int(value) for value in manifest_image_ids))
    if not manifest or target_epsilon < 0.0 or lcb_z < 0.0:
        raise ValueError("manifest and summary parameters must be valid")
    target = rows.target.cpu()
    image_ids = rows.image_ids.cpu()
    selected_values = target[selected]
    per_manifest: list[float] = []
    paired_gains: list[float] = []
    positive_fractions: list[float] = []
    baseline_positive_fractions: list[float] = []
    selected_image_values: list[float] = []
    for image_id in manifest:
        image_mask = selected & image_ids.eq(int(image_id))
        if image_mask.any():
            image_target = target[image_mask]
            baseline_target = target[image_ids.eq(int(image_id))]
            value = float(image_target.mean().item())
            baseline_value = float(baseline_target.mean().item())
            selected_image_values.append(value)
            positive_fractions.append(
                float(image_target.gt(float(target_epsilon)).float().mean().item())
            )
            baseline_positive_fractions.append(
                float(
                    baseline_target.gt(float(target_epsilon)).float().mean().item()
                )
            )
            per_manifest.append(value)
            paired_gains.append(value - baseline_value)
        else:
            per_manifest.append(0.0)
            paired_gains.append(0.0)
    values = torch.tensor(per_manifest, dtype=torch.float64)
    mean_delta = float(values.mean().item())
    standard_error = (
        float(values.std(unbiased=True).item()) / math.sqrt(values.numel())
        if values.numel() > 1
        else 0.0
    )
    precision = sum(positive_fractions) / max(1, len(positive_fractions))
    baseline_precision = sum(baseline_positive_fractions) / max(
        1, len(baseline_positive_fractions)
    )
    paired_values = torch.tensor(paired_gains, dtype=torch.float64)
    paired_mean = float(paired_values.mean().item())
    paired_standard_error = (
        float(paired_values.std(unbiased=True).item()) / math.sqrt(paired_values.numel())
        if paired_values.numel() > 1
        else 0.0
    )
    return {
        "candidate_count": int(selected.sum().item()),
        "image_count": len(selected_image_values),
        "manifest_image_count": len(manifest),
        "action_image_rate": len(selected_image_values) / len(manifest),
        "candidate_positive_prevalence": float(
            selected_values.gt(float(target_epsilon)).float().mean().item()
        )
        if selected_values.numel()
        else 0.0,
        "selected_positive_precision": precision,
        "same_image_uniform_positive_precision": baseline_precision,
        "positive_precision_lift": precision - baseline_precision,
        "mean_delta_u_per_manifest_image": mean_delta,
        "mean_delta_u_selected_images": sum(selected_image_values)
        / max(1, len(selected_image_values)),
        "mean_delta_u_standard_error": standard_error,
        "mean_delta_u_lcb": mean_delta - float(lcb_z) * standard_error,
        "paired_uniform_gain_per_manifest_image": paired_mean,
        "paired_uniform_gain_standard_error": paired_standard_error,
        "paired_uniform_gain_lcb": paired_mean
        - float(lcb_z) * paired_standard_error,
    }


def discover_stable_strata(
    fit: Mapping[str, Mapping[str, Any]],
    tune: Mapping[str, Mapping[str, Any]],
    *,
    min_fit_candidates: int,
    min_fit_images: int,
    min_tune_candidates: int,
    min_tune_images: int,
) -> list[str]:
    selected = []
    for key in sorted(set(fit) & set(tune)):
        fit_row = fit[key]
        tune_row = tune[key]
        if (
            int(fit_row["candidate_count"]) >= int(min_fit_candidates)
            and int(fit_row["image_count"]) >= int(min_fit_images)
            and float(fit_row["mean_delta_u_lcb"]) > 0.0
            and float(fit_row["paired_uniform_gain_lcb"]) > 0.0
            and int(tune_row["candidate_count"]) >= int(min_tune_candidates)
            and int(tune_row["image_count"]) >= int(min_tune_images)
            and float(tune_row["mean_delta_u_lcb"]) > 0.0
            and float(tune_row["paired_uniform_gain_lcb"]) > 0.0
        ):
            selected.append(key)
    return selected


def select_primary_stratum(
    selected: Sequence[str], tune: Mapping[str, Mapping[str, Any]]
) -> str | None:
    if not selected:
        return None
    return max(
        selected,
        key=lambda key: (
            float(tune[key]["paired_uniform_gain_lcb"]),
            float(tune[key]["mean_delta_u_lcb"]),
            int(tune[key]["image_count"]),
            key,
        ),
    )


def rate_matched_uniform_control(
    rows: StepActionRows,
    *,
    manifest_image_ids: Sequence[int],
    action_image_count: int,
    trials: int,
    seed: int,
) -> dict[str, float | int]:
    _validate_rows(rows)
    manifest_count = len(set(int(value) for value in manifest_image_ids))
    eligible = torch.unique(rows.image_ids.cpu(), sorted=True)
    if not 0 < action_image_count <= eligible.numel() or trials <= 0 or manifest_count <= 0:
        raise ValueError("invalid rate-matched control parameters")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    samples = []
    target = rows.target.cpu()
    image_ids = rows.image_ids.cpu()
    for _ in range(int(trials)):
        chosen_images = eligible[
            torch.randperm(eligible.numel(), generator=generator)[:action_image_count]
        ]
        total = 0.0
        for image_id in chosen_images.tolist():
            candidates = torch.nonzero(
                image_ids.eq(int(image_id)), as_tuple=False
            ).flatten()
            chosen = candidates[
                int(torch.randint(candidates.numel(), (1,), generator=generator).item())
            ]
            total += float(target[chosen].item())
        samples.append(total / manifest_count)
    return _distribution_summary(samples)


def target_permutation_control(
    rows: StepActionRows,
    mask: torch.Tensor,
    *,
    manifest_image_ids: Sequence[int],
    trials: int,
    seed: int,
) -> dict[str, float | int]:
    _validate_rows(rows)
    selected = torch.as_tensor(mask, dtype=torch.bool).flatten().cpu()
    if selected.shape != rows.target.shape or trials <= 0:
        raise ValueError("invalid target permutation control parameters")
    manifest_count = len(set(int(value) for value in manifest_image_ids))
    if manifest_count <= 0:
        raise ValueError("manifest must be non-empty")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    target = rows.target.cpu()
    image_ids = rows.image_ids.cpu()
    samples = []
    for _ in range(int(trials)):
        permuted = target.clone()
        for image_id in torch.unique(image_ids, sorted=True).tolist():
            image_rows = torch.nonzero(
                image_ids.eq(int(image_id)), as_tuple=False
            ).flatten()
            if image_rows.numel() > 1:
                order = torch.randperm(image_rows.numel(), generator=generator)
                permuted[image_rows] = target[image_rows[order]]
        paired_total = 0.0
        for image_id in torch.unique(image_ids[selected], sorted=True).tolist():
            image_mask = selected & image_ids.eq(int(image_id))
            baseline_mask = image_ids.eq(int(image_id))
            paired_total += float(
                (permuted[image_mask].mean() - permuted[baseline_mask].mean()).item()
            )
        samples.append(paired_total / manifest_count)
    summary = _distribution_summary(samples)
    return {
        "trials": summary["trials"],
        "paired_uniform_gain_mean": summary["mean_delta_u_mean"],
        "paired_uniform_gain_p05": summary["mean_delta_u_p05"],
        "paired_uniform_gain_p95": summary["mean_delta_u_p95"],
    }


def evaluate_step_strata_gates(
    payload: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, Any]:
    gates = config["gates"]
    support = payload["outer_support"]
    winners: list[str] = []
    diagnostics: dict[str, Any] = {}
    primary = payload.get("primary_stratum")
    for key in [primary] if primary is not None else []:
        row = payload["strata"][key]
        outer = row["outer"]
        permutation_p95 = float(
            row["target_permutation"]["paired_uniform_gain_p95"]
        )
        mean_delta = float(outer["mean_delta_u_per_manifest_image"])
        paired_gain = float(outer["paired_uniform_gain_per_manifest_image"])
        passed = (
            float(row["fit"]["mean_delta_u_lcb"]) > 0.0
            and float(row["tune"]["mean_delta_u_lcb"]) > 0.0
            and int(outer["candidate_count"])
            >= int(gates["min_outer_stratum_candidates"])
            and int(outer["image_count"]) >= int(gates["min_outer_stratum_images"])
            and mean_delta >= float(gates["min_outer_mean_delta_u"])
            and float(outer["mean_delta_u_lcb"])
            > float(gates["min_outer_mean_delta_u_lcb"])
            and paired_gain >= float(gates["min_paired_uniform_gain"])
            and float(outer["paired_uniform_gain_lcb"])
            > float(gates["min_paired_uniform_gain_lcb"])
            and float(outer["positive_precision_lift"])
            >= float(gates["min_positive_precision_lift"])
            and paired_gain - permutation_p95
            >= float(gates["min_gain_over_permutation_p95"])
        )
        diagnostics[key] = {
            "passed": passed,
            "gain_over_permutation_p95": paired_gain - permutation_p95,
        }
        if passed:
            winners.append(key)
    gate_values = {
        "G0_outer_support": (
            int(support["candidate_count"]) >= int(gates["min_outer_candidates"])
            and int(support["image_count"]) >= int(gates["min_outer_images"])
        ),
        "G1_fit_tune_discovery": bool(payload["selected_strata"]),
        "G2_outer_stable_stratum": bool(winners),
        "G3_control_gain": bool(winners),
    }
    return {
        "all_passed": all(gate_values.values()),
        "gates": gate_values,
        "winning_strata": winners,
        "diagnostics": diagnostics,
    }


def _distribution_summary(samples: Sequence[float]) -> dict[str, float | int]:
    values = torch.tensor(list(samples), dtype=torch.float64)
    return {
        "trials": int(values.numel()),
        "mean_delta_u_mean": float(values.mean().item()),
        "mean_delta_u_p05": float(torch.quantile(values, 0.05).item()),
        "mean_delta_u_p95": float(torch.quantile(values, 0.95).item()),
    }


def _validate_rows(rows: StepActionRows) -> None:
    count = rows.target.numel()
    if (
        count == 0
        or rows.target.shape != (count,)
        or rows.image_ids.shape != (count,)
        or rows.class_ids.shape != (count,)
        or len(rows.family) != count
        or len(rows.direction) != count
        or len(rows.scale_bin) != count
    ):
        raise ValueError("step action rows must share non-empty rows")
    if not torch.isfinite(rows.target).all():
        raise ValueError("step action targets must be finite")
