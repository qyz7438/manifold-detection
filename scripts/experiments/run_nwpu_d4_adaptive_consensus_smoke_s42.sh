#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/ps/lzz/manifold-detection-energy-transport"
RUN_DIR="$ROOT/runs/nwpu_d4_adaptive_consensus_smoke_s42"
EXPECTED_CONFIG_SHA="17e9d28770f5fab9fe7fe067d20c559c7c4d11c91251472414ef2ffb0de23642"
EXPECTED_SOURCE_D3="dd4d00d2a37e168cecc26482eb715e12fbd846bd71856c3ff323a54e086d1a11"
cd "$ROOT"
free_mb=$(nvidia-smi -i 2 --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
estimated_peak_mib=7680
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
  /home/ps/anaconda3/envs/RLimage/bin/python - "$RUN_DIR/eval_metrics.json" "$EXPECTED_CONFIG_SHA" "$EXPECTED_SOURCE_D3" "$current_head" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
valid = (
    payload.get("completed") is True
    and payload.get("config_sha256") == sys.argv[2]
    and payload.get("source_d3_result_sha256") == sys.argv[3]
    and payload.get("git_commit") == sys.argv[4]
    and payload.get("git_dirty") is False
)
if not valid:
    raise SystemExit("existing D4 artifact has invalid provenance")
PY
  echo "validated D4 artifact already exists"
  exit 0
fi
export CUDA_VISIBLE_DEVICES=2
mkdir -p "$RUN_DIR"
/home/ps/anaconda3/envs/RLimage/bin/python scripts/train_nwpu_d4_adaptive_consensus.py \
  --run-dir "$RUN_DIR" --require-clean-git 2>&1 | tee "$RUN_DIR/launcher.log"
