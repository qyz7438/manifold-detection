"""Run Plans 2.13-2.15: 15 groups total, send results to Feishu."""
import subprocess
import sys
import json
from pathlib import Path

PYTHON = sys.executable
SCRIPT = "scripts/round28_train_eval.py"

PLAN213 = [
    # (run_name, afm_type, seed, epochs)
    ("round213_baseline_s42", "none", 42, 1),
    ("round213_baseline_s123", "none", 123, 1),
    ("round213_baseline_s456", "none", 456, 1),
    ("round213_identity_s42", "identity", 42, 1),
    ("round213_identity_s123", "identity", 123, 1),
    ("round213_identity_s456", "identity", 456, 1),
    ("round213_mplseg_s42", "mplseg", 42, 1),
    ("round213_mplseg_s123", "mplseg", 123, 1),
    ("round213_mplseg_s456", "mplseg", 456, 1),
]

PLAN214 = [
    ("round214_mplseg_trained", "mplseg"),
    ("round214_mplseg_frozen", "mplseg_frozen"),
    ("round214_mplseg_notune", "mplseg_notune"),
]

PLAN215 = [
    ("round215_weak_03", "mplseg_weak"),
    ("round215_mid_06", "mplseg_mid"),
    ("round215_strong_10", "mplseg"),
]


def run_group(run_name, afm_type, seed=42, epochs=1):
    cmd = [
        PYTHON, SCRIPT,
        "--run-name", run_name,
        "--afm-type", afm_type,
        "--trainable-mode", "full",
        "--epochs", str(epochs),
        "--seed", str(seed),
    ]
    print(f"  {run_name} ...", end=" ", flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"FAIL")
        return {"run": run_name, "status": "FAIL", "error": r.stderr[-200:]}
    m = Path(f"runs/{run_name}/eval_metrics.json")
    if m.exists():
        d = json.loads(m.read_text())
        h = d.get("history", [{}])[-1]
        result = {
            "run": run_name,
            "status": "OK",
            "ap50": d.get("ap50"),
            "ap75": d.get("ap75"),
            "residual_scale": h.get("residual_scale", "N/A"),
            "train_loss": h.get("train_loss", "N/A"),
        }
        print(f"AP50={result['ap50']:.4f} AP75={result['ap75']:.4f}")
        return result
    print("no metrics")
    return {"run": run_name, "status": "NO_METRICS"}


def format_feishu_table(results, title):
    lines = [title, ""]
    lines.append("| Group | AP50 | AP75 | residual_scale | train_loss |")
    lines.append("|---|---:|---:|---:|---:|")
    for r in results:
        ap50 = f"{r['ap50']:.4f}" if r.get('ap50') else r['status']
        ap75 = f"{r['ap75']:.4f}" if r.get('ap75') else r['status']
        rs = f"{r['residual_scale']:.4f}" if isinstance(r.get('residual_scale'), float) else str(r.get('residual_scale', '-'))
        tl = f"{r['train_loss']:.4f}" if isinstance(r.get('train_loss'), float) else str(r.get('train_loss', '-'))
        lines.append(f"| {r['run']} | {ap50} | {ap75} | {rs} | {tl} |")
    return "\n".join(lines)


def send_feishu(text):
    script = Path(__file__).parent / "notify_feishu.py"
    subprocess.run([PYTHON, str(script), text], capture_output=True)


def main():
    all_results = {}

    # Plan 2.13
    print("\n=== Plan 2.13: Multi-seed validation (9 groups) ===")
    results_213 = [run_group(*args) for args in PLAN213]
    all_results["2.13"] = results_213

    # Plan 2.14
    print("\n=== Plan 2.14: Gate ablation (3 groups) ===")
    results_214 = [run_group(name, afm_type) for name, afm_type in PLAN214]
    all_results["2.14"] = results_214

    # Plan 2.15
    print("\n=== Plan 2.15: Gate strength sweep (3 groups) ===")
    results_215 = [run_group(name, afm_type) for name, afm_type in PLAN215]
    all_results["2.15"] = results_215

    # Format and send to Feishu
    print("\n=== Sending to Feishu ===")
    for plan, results in all_results.items():
        msg = format_feishu_table(results, f"## Plan {plan}")
        print(msg)
        print()
        send_feishu(msg)

    # Final summary
    summary_parts = ["## Summary: Plans 2.13-2.15"]
    for plan, results in all_results.items():
        oks = sum(1 for r in results if r["status"] == "OK")
        fails = sum(1 for r in results if r["status"] != "OK")
        summary_parts.append(f"Plan {plan}: {oks} OK, {fails} failed")
    send_feishu("\n".join(summary_parts))
    print("\nDone. All results sent to Feishu.")


if __name__ == "__main__":
    main()
