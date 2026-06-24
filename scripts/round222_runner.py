"""Plan 2.22: Data volume sweep.

4 volume levels x 2 configs (baseline + A post-training) x seed42 x PF+MobV3 x 3ep.
Baseline trains from scratch; A starts from mid06_5ep checkpoint.
"""
import subprocess, sys, json, time
from pathlib import Path

PY = sys.executable
SCRIPT = "scripts/round28_train_eval.py"
GIT = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
SEED = 42
BASE_CKPT = "runs/round216pp_mid06_s42/checkpoint_last.pth"
MAX_RETRIES = 3
VOLUMES = [30, 60, 90, 136]


def run_one(run_name, afm_type, trainable_mode, epochs, limit_train, checkpoint=None):
    for attempt in range(1, MAX_RETRIES + 1):
        cmd = [
            PY, SCRIPT,
            "--run-name", run_name,
            "--afm-type", afm_type,
            "--trainable-mode", trainable_mode,
            "--epochs", str(epochs),
            "--seed", str(SEED),
            "--limit-train", str(limit_train),
        ]
        if checkpoint:
            cmd.extend(["--checkpoint", checkpoint])
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode == 0:
            break
        print(f"  RETRY {attempt}/{MAX_RETRIES}: {r.stderr[-200:]}")
        time.sleep(5)
    m = Path(f"runs/{run_name}/eval_metrics.json")
    if m.exists():
        d = json.loads(m.read_text())
        d["git_hash"] = GIT
        d["train_size"] = limit_train
        m.write_text(json.dumps(d, indent=2, ensure_ascii=False))
        return {"run": run_name, "status": "OK", "ap50": d["ap50"], "ap75": d["ap75"]}
    return {"run": run_name, "status": "CRASH", "ap50": 0, "ap75": 0}


def main():
    all_r = []
    for vol in VOLUMES:
        # baseline
        r = run_one(f"round222_d{vol}_baseline", "none", "full", 3, vol)
        all_r.append(r)
        print(f"  d{vol}_baseline: AP50={r.get('ap50','?')}")

        # A post-training
        r = run_one(f"round222_d{vol}_A", "mplseg_weak", "afm_only", 3, vol, BASE_CKPT)
        all_r.append(r)
        print(f"  d{vol}_A: AP50={r.get('ap50','?')}")

    # Summary
    lines = ["## Plan 2.22 Data Volume Sweep", "",
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
