"""Plan 3.7: Aggregate all detection results and compute A/C win rate vs baseline."""
import json, subprocess, sys
from pathlib import Path
from collections import defaultdict

PY = sys.executable
RUNS_DIR = Path("runs")

PLANS = {
    "2.16": ["round216_baseline", "round216_mplseg_mid06", "round216_mplseg_weak03",
             "round216_mplseg_strong10", "round216_mplseg_frozen", "round216_identity",
             "round216p_mid06", "round216p_frozen", "round216pp_mid06"],
    "2.18": ["round218_", "round218b_", "round218c_"],
    "2.19": ["round219_"],
    "2.20": ["round220_"],
    "2.21": ["round221_"],
    "2.22": ["round222_"],
    "2.25": ["round225_"],
    "2.26": ["round226_"],
}

def load_metrics(run_dir):
    p = RUNS_DIR / run_dir / "eval_metrics.json"
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text())
        return {"run": run_dir, "ap50": d.get("ap50", 0), "ap75": d.get("ap75", 0),
                "precision": d.get("precision", 0), "ece": d.get("ece", 0),
                "afm_type": d.get("afm_type", ""), "trainable_mode": d.get("trainable_mode", ""),
                "epochs": d.get("epochs", 0), "seed": d.get("seed", 0)}
    except Exception:
        return None

def classify(metrics):
    """Classify a run as baseline, mid06, A_post, or C_post."""
    r = metrics["run"].lower()
    if "baseline" in r or ("afm_type" in metrics and metrics["afm_type"] == "none"):
        return "baseline"
    if "mid06" in r or "mplseg_mid" in str(metrics.get("afm_type", "")):
        return "mid06"
    if "_a_" in r or "_a_s" in r or "approach" in str(metrics).lower() and "a" in str(metrics).lower():
        return "A_post"
    if "_c_" in r or "_c_s" in r:
        return "C_post"
    return "other"

def main():
    all_m = []
    for run_dir in RUNS_DIR.iterdir():
        if not run_dir.is_dir():
            continue
        for plan, prefixes in PLANS.items():
            if any(run_dir.name.startswith(p) for p in prefixes):
                m = load_metrics(run_dir.name)
                if m:
                    m["plan"] = plan
                    m["category"] = classify(m)
                    all_m.append(m)
                break

    # Win rate summary
    lines = ["## Plan 3.7 Detection Summary", "",
             f"Total experiments: {len(all_m)}", "",
             "### By Category", "",
             "| Category | Count | Mean AP50 | Mean AP75 | Mean ECE |",
             "|---:|---:|---:|---:|---:|"]

    by_cat = defaultdict(list)
    for m in all_m:
        by_cat[m["category"]].append(m)

    for cat in ["baseline", "mid06", "A_post", "C_post", "other"]:
        if not by_cat[cat]:
            continue
        items = by_cat[cat]
        ap50 = sum(i["ap50"] for i in items) / len(items)
        ap75 = sum(i["ap75"] for i in items) / len(items)
        ece = sum(i["ece"] for i in items) / len(items)
        lines.append(f"| {cat} | {len(items)} | {ap50:.4f} | {ap75:.4f} | {ece:.4f} |")

    msg = "\n".join(lines)
    print(msg)
    subprocess.run([PY, "scripts/notify_feishu.py", msg[:800]], capture_output=True)

if __name__ == "__main__":
    main()
