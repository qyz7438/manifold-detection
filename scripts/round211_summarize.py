"""Summarize Round 2.11 results and write decision report."""
from __future__ import annotations

import json
from pathlib import Path

GROUPS = [
    "round211_voc_v1_baseline_eval",
    "round211_voc_v2_posttrain_detection_only",
    "round211_voc_v3_posttrain_spatial",
    "round211_voc_v4_posttrain_spatial_spectral_loggate",
    "round211_voc_v5_posttrain_spatial_shuffled_spectral",
]

SCENES = ["clean", "object_edge", "background_texture", "near_object"]

FIELDS = ["AP50", "AP75", "precision", "recall", "ECE", "high_conf_FP", "num_predictions"]


def _load_metrics(run_name: str) -> dict | None:
    p = Path(f"runs/{run_name}/eval_metrics.json")
    if p.exists():
        with p.open(encoding="utf-8") as f:
            return json.load(f)
    return None


def _fmt(v) -> str:
    if v is None:
        return "N/A"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def main():
    rows = []
    for g in GROUPS:
        m = _load_metrics(g)
        if m:
            rows.append({f: m.get(f) for f in FIELDS + ["mode"]})
            rows[-1]["group"] = g
        else:
            rows.append({"group": g, "mode": "missing"})

    # Clean table
    lines = []
    lines.append("## VOC Clean Metrics")
    lines.append("| Group | Mode | AP50 | AP75 | Prec | Recall | ECE | hiFP | #Pred |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        lines.append(f"| {r.get('group','?')} | {r.get('mode','?')} | {_fmt(r.get('AP50'))} | {_fmt(r.get('AP75'))} | {_fmt(r.get('precision'))} | {_fmt(r.get('recall'))} | {_fmt(r.get('ECE'))} | {_fmt(r.get('high_conf_FP'))} | {_fmt(r.get('num_predictions'))} |")

    # Stress table
    lines.append("")
    lines.append("## VOC Stress Metrics (AP50 only)")
    header = "| Group | " + " | ".join(SCENES) + " |"
    lines.append(header)
    lines.append("|---" * (len(SCENES) + 1) + "|")
    for g in GROUPS:
        cells = [g]
        for scene in SCENES:
            p = Path(f"runs/{g}/{scene}_eval_metrics.json")
            if p.exists():
                with p.open(encoding="utf-8") as f:
                    sm = json.load(f)
                cells.append(_fmt(sm.get("AP50")))
            else:
                cells.append("N/A")
        lines.append("| " + " | ".join(cells) + " |")

    # Decision verdict
    lines.append("")
    lines.append("## Decision Verdict")
    lines.append("")

    v3 = _load_metrics("round211_voc_v3_posttrain_spatial")
    v4 = _load_metrics("round211_voc_v4_posttrain_spatial_spectral_loggate")
    v5 = _load_metrics("round211_voc_v5_posttrain_spatial_shuffled_spectral")

    if v3 and v4 and v5:
        ap50_v3 = v3.get("AP50", 0)
        ap50_v4 = v4.get("AP50", 0)
        ap50_v5 = v5.get("AP50", 0)

        if ap50_v4 <= ap50_v3:
            lines.append("**Verdict**: V4 did not beat V3. Spectral evidence is still not useful on a harder detection subset.")
        elif ap50_v4 > ap50_v3 and abs(ap50_v5 - ap50_v4) < 0.01:
            lines.append("**Verdict**: V4 beat V3 but V5 matches V4. Improvement is not spectral-causal.")
        else:
            lines.append("**Verdict**: V4 beat V3 and V5 does not match V4. Penn-Fudan was too simple — spectral evidence deserves Round 3.x validation.")
    else:
        lines.append("**Verdict**: Pending — run the matrix first.")

    report = "\n".join(lines)
    Path("docs/round211_results.md").write_text(report, encoding="utf-8")

    summary_data = {"groups": rows, "verdict": lines[-1]}
    Path("runs/round211_summary.json").write_text(json.dumps(summary_data, indent=2, ensure_ascii=False), encoding="utf-8")

    print(report)


if __name__ == "__main__":
    main()
