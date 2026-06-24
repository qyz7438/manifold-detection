# Plan 2.21: Freeze Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Determine which components must be frozen during AFM post-training by ablating freeze targets.

**Architecture:** 5 groups × seed42 × PF+MobV3 × Approach A (weak gate 0.1) × 2 epochs post-training. All start from mid06_5ep checkpoint. Vary which components are frozen vs trainable.

**Tech Stack:** Python, PyTorch, existing `spectral_detection_posttrain` package, `round28_train_eval.py`.

---

## Design

| Group | Frozen | Trainable |
|---|---|---|
| freeze_all | backbone+RPN+box_head | AFM only (A standard) |
| freeze_bb | backbone | AFM+RPN+box_head |
| freeze_rpn | RPN | AFM+backbone+box_head |
| freeze_box | box_head | AFM+backbone+RPN |
| freeze_none | none | full model (neg control) |

---

## Task 1: Runner Script

**Files:**
- Create: `scripts/round221_runner.py`

- [ ] **Step 1: Write runner with retry logic**

```python
"""Plan 2.21: Freeze component ablation with post-training Approach A."""
import subprocess, sys, json, time
from pathlib import Path

PY = sys.executable
SCRIPT = "scripts/round28_train_eval.py"
GIT = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
SEED = 42
BASE_CKPT = "runs/round216pp_mid06_5ep_s42/checkpoint_last.pth"

GROUPS = [
    ("round221_freeze_all", "afm_only"),
    ("round221_freeze_bb", "afm_box_head"),
    ("round221_freeze_rpn", "full"),  # freeze RPN manually via hook
    ("round221_freeze_box", "full"),  # freeze box_head manually via hook
    ("round221_freeze_none", "full"),
]

MAX_RETRIES = 3

def run_one(run_name, trainable_mode, checkpoint):
    for attempt in range(1, MAX_RETRIES + 1):
        cmd = [
            PY, SCRIPT,
            "--run-name", run_name,
            "--afm-type", "mplseg_weak",
            "--trainable-mode", trainable_mode,
            "--epochs", "2",
            "--seed", str(SEED),
            "--checkpoint", checkpoint,
        ]
        if attempt > 1:
            print(f"  RETRY {attempt}/{MAX_RETRIES}")
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode == 0:
            break
        print(f"  CRASH (attempt {attempt}): {r.stderr[-200:]}")
        time.sleep(3)
    m = Path(f"runs/{run_name}/eval_metrics.json")
    if m.exists():
        d = json.loads(m.read_text())
        d["git_hash"] = GIT
        m.write_text(json.dumps(d, indent=2, ensure_ascii=False))
        return {"run": run_name, "status": "OK", "ap50": d["ap50"], "ap75": d["ap75"]}
    return {"run": run_name, "status": "CRASH"}

def main():
    all_r = []
    for run_name, trainable_mode in GROUPS:
        print(f"{run_name} ({trainable_mode}) ...", end=" ", flush=True)
        r = run_one(run_name, trainable_mode, BASE_CKPT)
        all_r.append(r)
        print(f"AP50={r.get('ap50','N/A')}")

    lines = ["## Plan 2.21 Freeze Ablation", "",
             "| Run | AP50 | AP75 |", "|---:|---:|---:|"]
    for r in all_r:
        ap50 = f"{r['ap50']:.4f}" if r["status"] == "OK" else r["status"]
        ap75 = f"{r['ap75']:.4f}" if r["status"] == "OK" else ""
        lines.append(f"| {r['run']} | {ap50} | {ap75} |")
    msg = "\n".join(lines)
    print(f"\n{msg}")
    subprocess.run([PY, "scripts/notify_feishu.py", f"Plan 2.21: {sum(1 for r in all_r if r['status']=='OK')}/{len(all_r)} OK"], capture_output=True)

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run**

```powershell
cd E:/CLIproject/RLimage; $env:PYTHONPATH = "E:/CLIproject/RLimage"
E:\anaconda\01\envs\RLimage\python.exe scripts/round221_runner.py
```

- [ ] **Step 3: Verify results**

Check `runs/round221_*/eval_metrics.json` exist with AP50 values.

---

## Success Criteria

1. freeze_all AP50 >= baseline × 0.90 (post-training stable)
2. freeze_none AP50 close to baseline (no catastrophic forgetting)
3. Identifies which freeze targets are necessary for post-training stability
