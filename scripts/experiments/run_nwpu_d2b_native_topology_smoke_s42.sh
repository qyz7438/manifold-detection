#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/ps/lzz/manifold-detection-energy-transport"
RUN_DIR="$ROOT/runs/nwpu_d2b_native_topology_smoke_s42"
EXPECTED_CONFIG_SHA="1b07a3f3862cc49945b0a972a8ed96298c1924ced23638f0d09b7debb7836f92"
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
  /home/ps/anaconda3/envs/RLimage/bin/python - "$RUN_DIR/eval_metrics.json" "$EXPECTED_CONFIG_SHA" "$current_head" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
valid = (
    payload.get("completed") is True
    and payload.get("config_sha256") == sys.argv[2]
    and payload.get("git_commit") == sys.argv[3]
    and payload.get("git_dirty") is False
    and payload.get("source_cache_sha256") == "1fa2184945bf5d38b1c47e5b732ffce4bcdd1b57064f4f2eb71d76611304a53e"
    and payload.get("experiment_scope") == "native_topology_correctness_smoke_32_32"
)
if not valid:
    raise SystemExit("existing D2b artifact has invalid provenance")
PY
  echo "validated D2b artifact already exists"
  exit 0
fi
export CUDA_VISIBLE_DEVICES=2
mkdir -p "$RUN_DIR"
/home/ps/anaconda3/envs/RLimage/bin/python scripts/train_nwpu_d2b_native_topology.py \
  --run-dir "$RUN_DIR" --require-clean-git 2>&1 | tee "$RUN_DIR/launcher.log"
