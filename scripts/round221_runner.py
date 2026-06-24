"""Plan 2.21: Freeze component ablation with post-training Approach A (weak gate 0.1).

5 groups: freeze_all, freeze_bb, freeze_rpn, freeze_box, freeze_none.
All start from mid06_5ep checkpoint, seed=42, 2 epochs.
"""
import subprocess, sys, json, time
from pathlib import Path

PY = sys.executable
SCRIPT = "scripts/round28_train_eval.py"
GIT = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
SEED = 42
BASE_CKPT = "runs/round216pp_mid06_s42/checkpoint_last.pth"
MAX_RETRIES = 3

GROUPS = [
    ("round221_freeze_all", "afm_only"),           # freeze backbone+RPN+box_head, train AFM
    ("round221_freeze_bb", "all_except_backbone"),   # freeze backbone only
    ("round221_freeze_rpn", "all_except_rpn"),       # freeze RPN only
    ("round221_freeze_box", "all_except_box"),       # freeze box_head only
    ("round221_freeze_none", "full"),                # nothing frozen (neg control)
]


def run_one(run_name, trainable_mode):
    for attempt in range(1, MAX_RETRIES + 1):
        cmd = [
            PY, SCRIPT,
            "--run-name", run_name,
            "--afm-type", "mplseg_weak",
            "--trainable-mode", trainable_mode,
            "--epochs", "2",
            "--seed", str(SEED),
            "--checkpoint", BASE_CKPT,
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
    return {"run": run_name, "status": "CRASH"}


def main():
    all_r = []
    for run_name, mode in GROUPS:
        label = run_name.replace("round221_", "")
        print(f"{label} ({mode}) ...", end=" ", flush=True)
        r = run_one(run_name, mode)
        all_r.append(r)
        ap50 = f"{r['ap50']:.4f}" if r["status"] == "OK" else r["status"]
        print(f"AP50={ap50}")

    lines = ["## Plan 2.21 Freeze Ablation", "",
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
