#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/ps/lzz/manifold-detection-energy-transport"
RUN_DIR="$ROOT/runs/nwpu_e1_post_nms_suppress_smoke_s42"
EXPECTED_CONFIG_SHA="72405a2fdaf19b1b5d9b3ae28b57e21c5c33343e177f8e181273ba4d53ef25bb"
EXPECTED_SOURCE_D4="6f437f7c9e05c718f80dda8ec5f9d1d4e230288f0d2f73207d8e452356c92223"
cd "$ROOT"
free_mb=$(nvidia-smi -i 2 --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
estimated_peak_mib=6144
if (( free_mb - estimated_peak_mib <= 8192 )); then
  echo "GPU2 memory.free=${free_mb} MiB; estimated peak ${estimated_peak_mib} MiB would violate the 8192 MiB reserve"
  exit 3
fi
if [[ -n "$(git status --porcelain)" ]]; then
  echo "remote worktree is dirty"
  exit 4
fi
if [[ -f "$RUN_DIR/eval_metrics.json" ]]; then
  current_head=$(git rev-parse HEAD)
  /home/ps/anaconda3/envs/RLimage/bin/python - "$RUN_DIR/eval_metrics.json" "$EXPECTED_CONFIG_SHA" "$EXPECTED_SOURCE_D4" "$current_head" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
valid = (
    payload.get("completed") is True
    and payload.get("config_sha256") == sys.argv[2]
    and payload.get("source_d4_result_sha256") == sys.argv[3]
    and payload.get("git_commit") == sys.argv[4]
    and payload.get("git_dirty") is False
)
if not valid:
    raise SystemExit("existing E1 artifact has invalid provenance")
PY
  echo "validated E1 artifact already exists"
  exit 0
fi
export CUDA_VISIBLE_DEVICES=2
mkdir -p "$RUN_DIR"
/home/ps/anaconda3/envs/RLimage/bin/python scripts/train_nwpu_e1_post_nms_suppress.py \
  --run-dir "$RUN_DIR" --require-clean-git 2>&1 | tee "$RUN_DIR/launcher.log"
