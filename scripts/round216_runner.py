"""Plan 2.16: Strict deterministic paired rerun. cudnn.benchmark=False, git hash saved."""
import subprocess
import sys
import json
from pathlib import Path

PYTHON = sys.executable
SCRIPT = "scripts/round28_train_eval.py"

GIT_HASH = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()

CONFIGS = [
    ("round216_baseline", "none"),
    ("round216_identity", "identity"),
    ("round216_mplseg_weak03", "mplseg_weak"),
    ("round216_mplseg_mid06", "mplseg_mid"),
    ("round216_mplseg_strong10", "mplseg"),
    ("round216_mplseg_frozen", "mplseg_frozen"),
]
SEEDS = [42, 123, 456]

RESULTS = {}


def run_group(run_name, afm_type, seed):
    cmd = [
        PYTHON, SCRIPT,
        "--run-name", run_name,
        "--afm-type", afm_type,
        "--trainable-mode", "full",
        "--epochs", "1",
        "--seed", str(seed),
    ]
    print(f"  {run_name} ...", end=" ", flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"CRASH")
        return {"run": run_name, "afm": afm_type, "seed": seed, "status": "CRASH", "error": r.stderr[-300:]}

    m = Path(f"runs/{run_name}/eval_metrics.json")
    if not m.exists():
        return {"run": run_name, "afm": afm_type, "seed": seed, "status": "NO_METRICS"}

    d = json.loads(m.read_text())
    d["git_hash"] = GIT_HASH
    d["cli_args"] = " ".join(cmd)
    d["cudnn_benchmark"] = "False"
    d["cudnn_deterministic"] = "True"
    d["afm_config"] = afm_type
    d["seed_used"] = seed
    m.write_text(json.dumps(d, indent=2, ensure_ascii=False))

    h = d.get("history", [{}])[-1]
    result = {
        "run": run_name, "afm": afm_type, "seed": seed, "status": "OK",
        "ap50": d.get("ap50"), "ap75": d.get("ap75"),
        "precision": d.get("precision"), "recall": d.get("recall"),
        "ece": d.get("ece"), "pred": d.get("num_predictions"),
        "high_fp": d.get("high_conf_fp_count"),
        "residual_scale": h.get("residual_scale", "N/A"),
        "train_loss": h.get("train_loss", "N/A"),
    }
    print(f"AP50={result['ap50']:.4f} AP75={result['ap75']:.4f} ECE={result['ece']:.4f}")
    return result


def format_table(results, title):
    lines = [title, ""]
    lines.append("| Run | AP50 | AP75 | Prec | Recall | ECE | hiFP | Pred | r_scale |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in results:
        ap50 = f"{r['ap50']:.4f}" if r.get('ap50') else '-'
        ap75 = f"{r['ap75']:.4f}" if r.get('ap75') else '-'
        prec = f"{r['precision']:.4f}" if r.get('precision') else '-'
        rec = f"{r['recall']:.4f}" if r.get('recall') else '-'
        ece = f"{r['ece']:.4f}" if r.get('ece') else '-'
        hfp = str(r.get('high_fp', '-'))
        pred = str(r.get('pred', '-'))
        rs = f"{r['residual_scale']:.4f}" if isinstance(r.get('residual_scale'), float) else '-'
        lines.append(f"| {r['run']} | {ap50} | {ap75} | {prec} | {rec} | {ece} | {hfp} | {pred} | {rs} |")
    return "\n".join(lines)


def format_summary(all_results):
    lines = ["## Plan 2.16 Summary (per-config mean over 3 seeds)", ""]
    lines.append("| Config | AP50 | AP75 | Prec | Recall | ECE | hiFP | Pred | r_scale |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    for cfg_name, afm_type in CONFIGS:
        matching = [r for r in all_results if r["afm"] == afm_type and r["status"] == "OK"]
        if len(matching) < 3:
            lines.append(f"| {cfg_name} | INCOMPLETE | | | | | | | |")
            continue
        def mean(key):
            vals = [r[key] for r in matching if r.get(key) is not None]
            return sum(vals) / len(vals) if vals else None
        ap50 = mean("ap50"); ap75 = mean("ap75")
        prec = mean("precision"); rec = mean("recall"); ece = mean("ece")
        hfp = mean("high_fp"); pred = mean("pred")
        r_scale_vals = [r["residual_scale"] for r in matching if isinstance(r.get("residual_scale"), float)]
        rs = sum(r_scale_vals) / len(r_scale_vals) if r_scale_vals else None
        lines.append(f"| {cfg_name} | {ap50:.4f} | {ap75:.4f} | {prec:.4f} | {rec:.4f} | {ece:.4f} | {hfp:.1f} | {pred:.0f} | {rs:.4f} |")

    return "\n".join(lines)


def send_feishu(text):
    script = Path(__file__).parent / "notify_feishu.py"
    subprocess.run([PYTHON, str(script), text], capture_output=True)


def main():
    print(f"Plan 2.16: 6 configs x 3 seeds = 18 groups")
    print(f"git: {GIT_HASH}")
    print(f"cudnn.benchmark=False  cudnn.deterministic=True\n")

    all_results = []
    for cfg_name, afm_type in CONFIGS:
        print(f"\n-- {cfg_name} ({afm_type}) --")
        for seed in SEEDS:
            run_name = f"{cfg_name}_s{seed}"
            result = run_group(run_name, afm_type, seed)
            all_results.append(result)

    if all(r["status"] == "OK" for r in all_results):
        summary = format_summary(all_results)
        send_feishu(f"Plan 2.16 complete: 18/18 OK. git={GIT_HASH[:8]}")
        send_feishu(summary)
        print(f"\n{summary}")
    else:
        crashed = [r for r in all_results if r["status"] != "OK"]
        send_feishu(f"Plan 2.16: {len(crashed)}/{len(all_results)} FAILED")
        for c in crashed:
            print(f"  {c['run']}: {c['status']}")

    print(f"\nDone. Results sent to Feishu.")


if __name__ == "__main__":
    main()
