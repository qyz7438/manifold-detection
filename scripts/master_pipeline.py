"""Master pipeline: run Plans 2.21-2.26 + 3.x sequentially with notifications.

Each plan runs as a subprocess. On failure, retries up to 3 times.
Sends Feishu notification after each plan completes.
"""
import subprocess, sys, time, json
from datetime import datetime
from pathlib import Path

PY = sys.executable
NOTIFY = [PY, "scripts/notify_feishu.py"]
MAX_PLAN_RETRIES = 2

PIPELINE = [
    # (plan_id, script, description, timeout_sec)
    ("2.21", "scripts/round221_runner.py", "Freeze Ablation", 600),
    ("2.22", "scripts/round222_runner.py", "Data Volume Sweep", 900),
    ("2.23", "scripts/round223_analysis.py", "Gate Visualization", 300),
    ("2.25", "scripts/round225_runner.py", "Phase Ablation", 2400),
    ("2.26", "scripts/round226_runner.py", "Recipe Sweep", 3600),
    ("3.4", "scripts/round34_runner.py", "VOC 20-Class Full", 5400),
]


def notify(msg):
    try:
        subprocess.run(NOTIFY + [msg], capture_output=True, timeout=30)
    except Exception:
        pass


def run_plan(plan_id, script, desc, timeout):
    print(f"\n{'='*60}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Plan {plan_id}: {desc}")
    print(f"{'='*60}")

    for attempt in range(1, MAX_PLAN_RETRIES + 1):
        try:
            r = subprocess.run([PY, script], capture_output=True, text=True, timeout=timeout)
            print(r.stdout[-3000:] if len(r.stdout) > 3000 else r.stdout)
            if r.returncode == 0:
                notify(f"Plan {plan_id} ({desc}) COMPLETED")
                return True
            print(f"FAILED (exit {r.returncode}): {r.stderr[-500:]}")
        except subprocess.TimeoutExpired:
            print(f"TIMEOUT after {timeout}s")
        except Exception as e:
            print(f"ERROR: {e}")

        if attempt < MAX_PLAN_RETRIES:
            print(f"Retrying plan {plan_id} ({attempt}/{MAX_PLAN_RETRIES})...")
            time.sleep(10)
        else:
            notify(f"Plan {plan_id} ({desc}) FAILED after {MAX_PLAN_RETRIES} retries")
            return False

    return False


def main():
    start = datetime.now()
    notify(f"Master pipeline STARTED at {start.strftime('%H:%M:%S')}. {len(PIPELINE)} plans.")
    print(f"Pipeline start: {start.strftime('%Y-%m-%d %H:%M:%S')}")

    results = {}
    for plan_id, script, desc, timeout in PIPELINE:
        ok = run_plan(plan_id, script, desc, timeout)
        results[plan_id] = "OK" if ok else "FAILED"
        elapsed = (datetime.now() - start).total_seconds() / 60
        print(f"Progress: {list(results.values()).count('OK')}/{len(results)} plans OK, {elapsed:.0f}min elapsed")

    elapsed = (datetime.now() - start).total_seconds() / 60
    summary = f"Pipeline DONE. {list(results.values()).count('OK')}/{len(results)} plans OK. {elapsed:.0f}min total."
    notify(summary)
    print(f"\n{'='*60}")
    print(summary)
    for pid, status in results.items():
        print(f"  Plan {pid}: {status}")


if __name__ == "__main__":
    main()
