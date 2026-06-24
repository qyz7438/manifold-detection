"""Plan 2.25: Phase contribution ablation.

3 variants (mag_only / phase_only / both) x 3 seeds x PF+MobV3 x 3ep full fine-tune.
Tests whether magnitude gate or phase residual is the critical component.
"""
import subprocess, sys, json, time
from pathlib import Path

PY = sys.executable
SCRIPT = "scripts/round28_train_eval.py"
GIT = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
SEEDS = [42, 123, 456]
MAX_RETRIES = 3

GROUPS = [
    ("round225_mag_only_s{}", "mplseg_mag_only", "Mag-only"),
    ("round225_phase_only_s{}", "mplseg_phase_only", "Phase-only"),
    ("round225_both_s{}", "mplseg_mid", "Both (control)"),
]


def run_one(run_name, afm_type, seed):
    for attempt in range(1, MAX_RETRIES + 1):
        cmd = [
            PY, SCRIPT,
            "--run-name", run_name,
            "--afm-type", afm_type,
            "--trainable-mode", "full",
            "--epochs", "3",
            "--seed", str(seed),
        ]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode == 0:
            break
        print(f"  RETRY {attempt}/{MAX_RETRIES}: {r.stderr[-200:]}")
        time.sleep(5)
    m = Path(f"runs/{run_name}/eval_metrics.json")
    if m.exists():
        d = json.loads(m.read_text())
        d["git_hash"] = GIT
        m.write_text(json.dumps(d, indent=2, ensure_ascii=False))
        return {"run": run_name, "status": "OK", "ap50": d["ap50"], "ap75": d["ap75"]}
    return {"run": run_name, "status": "CRASH", "ap50": 0, "ap75": 0}


def main():
    all_r = []
    for name_tpl, afm_type, label in GROUPS:
        print(f"\n-- {label} ({afm_type}) --")
        for seed in SEEDS:
            run_name = name_tpl.format(seed)
            print(f"  {run_name} ...", end=" ", flush=True)
            r = run_one(run_name, afm_type, seed)
            all_r.append(r)
            ap50 = f"{r['ap50']:.4f}" if r["status"] == "OK" else r["status"]
            print(f"AP50={ap50}")

    lines = ["## Plan 2.25 Phase Ablation", "",
             "| Run | AP50 | AP75 |", "|---:|---:|---:|"]
    for r in all_r:
        ap50 = f"{r['ap50']:.4f}" if r["status"] == "OK" else r["status"]
        ap75 = f"{r['ap75']:.4f}" if r["status"] == "OK" else ""
        lines.append(f"| {r['run']} | {ap50} | {ap75} |")
    msg = "\n".join(lines)
    print(f"\n{msg}")
    subprocess.run([PY, "scripts/notify_feishu.py", msg[:800]], capture_output=True)


if __name__ == "__main__":
    main()
