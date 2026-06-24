"""Summarize Round 2.8 results from eval metrics + diagnostics."""
from __future__ import annotations

import json
from pathlib import Path

from spectral_detection_posttrain.utils.io import save_json

GROUPS = [
    "round28_g01_baseline_full",
    "round28_g02_old_afm_full",
    "round28_g03_identity_current_full",
    "round28_g04_identity_delta_full",
    "round28_g05_identity_norm_delta_full",
    "round28_g06_baseline_box_head_only",
    "round28_g07_identity_current_afm_only",
    "round28_g08_identity_current_afm_box_head",
    "round28_g09_identity_delta_afm_box_head",
]


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    rows = []
    for group in GROUPS:
        metrics_path = Path("runs") / group / "eval_metrics.json"
        diagnostics_path = Path("runs") / group / "round28_diagnostics.json"
        metrics = _load_json(metrics_path) if metrics_path.exists() else {}
        diagnostics = _load_json(diagnostics_path) if diagnostics_path.exists() else {}
        threshold_005 = diagnostics.get("0.05", {})
        history = metrics.get("history", [])
        last_scales = history[-1] if history else {}
        rows.append({
            "group": group, "ap50": metrics.get("ap50"), "ap75": metrics.get("ap75"),
            "precision": metrics.get("precision"), "recall": metrics.get("recall"),
            "ece": metrics.get("ece"), "high_conf_fp_count": metrics.get("high_conf_fp_count"),
            "num_predictions": metrics.get("num_predictions"),
            "matched_iou_mean": threshold_005.get("matched_iou_mean"),
            "center_error_mean": threshold_005.get("center_error_mean"),
            "size_error_mean": threshold_005.get("size_error_mean"),
            "duplicate_predictions": threshold_005.get("duplicate_predictions"),
            "mag_scale": last_scales.get("mag_scale"),
            "phase_scale": last_scales.get("phase_scale"),
            "residual_scale": last_scales.get("residual_scale"),
        })

    save_json({"rows": rows}, Path("runs") / "round28_summary.json")
    lines = [
        "# Round 2.8 AFM Diagnostics Results", "",
        "| group | AP50 | AP75 | prec | recall | ECE | high_FP | pred | IoU_m | ctr_err | sz_err | dup | mag_s | pha_s | res_s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        def fmt(key, prec=".4f"):
            v = row.get(key)
            if v is None: return "NA"
            return f"{v:{prec}}"
        lines.append(
            f"| {row['group']} | {fmt('ap50')} | {fmt('ap75')} | {fmt('precision')} | "
            f"{fmt('recall')} | {fmt('ece')} | {fmt('high_conf_fp_count', '.0f')} | {fmt('num_predictions', '.0f')} | "
            f"{fmt('matched_iou_mean')} | {fmt('center_error_mean')} | {fmt('size_error_mean')} | "
            f"{fmt('duplicate_predictions', '.0f')} | {fmt('mag_scale', '.6f')} | {fmt('phase_scale', '.6f')} | {fmt('residual_scale', '.6f')} |"
        )
    lines.extend(["", "## Verdict Checklist", "",
        "- [ ] Frozen parity passed: identity AFM is detector-level no-op before training.",
        "- [ ] Identity delta residual improves AP75/precision over identity current.",
        "- [ ] Old AFM gain, if present, is separated from prediction-count inflation.",
        "- [ ] AFM-only and AFM+box-head training scopes explain whether AFM itself or head adaptation causes drift.",
        "- [ ] Threshold curves identify whether the AP75 drop is localization error or score calibration.",
    ])
    Path("docs/round28_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
