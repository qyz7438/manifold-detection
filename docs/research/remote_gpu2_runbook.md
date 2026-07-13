# Remote GPU2 Execution Runbook

This runbook documents the guarded path for running training or full evaluation
on the remote Linux GPU host.  It is a living procedure document; the
authoritative rule set is enforced by
[scripts/run/guard_gpu2.py](../../scripts/run/guard_gpu2.py) and by the
research-status registry.

## Host and environment

| Item | Local development | Remote execution |
|---|---|---|
| Host | Windows 10, AMD64 | Linux 6.8, x86_64, `ps@122.51.19.136` |
| Workspace | `E:/CLIproject/manifold` | `/home/ps/lzz/manifold-detection-energy-transport` |
| Conda env | `RLimage` | `RLimage` |
| Python | `E:/anaconda/01/envs/RLimage/python.exe` | `/home/ps/anaconda3/envs/RLimage/bin/python` |
| PyTorch | `2.1.0+cu121` | `2.1.0+cu121` |
| GPUs | 1 local consumer GPU | 4 GPUs; **only physical GPU2 may be used** |

The remote workspace is the only permitted remote working directory.  Do not
write to `/home/ps/lzz/RLimage` or any other path.

## Locked launch rules

1. **Physical GPU2 only.**  Set `CUDA_VISIBLE_DEVICES=2`.  Never use GPU0/1/3.
2. **Memory gate.**  After the new process reaches its peak, `memory.free` on
   GPU2 must be **strictly greater than 8192 MiB**.  The guard rejects equality.
3. **No interference.**  Never stop, restart, signal, reprioritize, or kill a
   process owned by another user or project.  The guard contains no kill/stop
   helpers.
4. **Clean Git.**  The working tree must have no uncommitted changes, no staged
   changes, and no untracked files outside runtime directories.
5. **Authorized experiment.**  The run must come from the experiment registry
   (`spectral_detection_posttrain/configs/registry/experiments.json`) and its
   research line must be `active` or `frozen` with `runnable` set accordingly.
   CLI overrides of the launcher are forbidden.
6. **No remote installs.**  Do not install or upgrade packages on the remote
   host.  If a dependency is missing, the correct fix is a local/CI change, not
   an ad-hoc remote `pip install`.

## Pre-flight checks (local or remote)

```bash
# 1. Query GPU2 state (read-only)
python scripts/run/guard_gpu2.py query

# 2. Check all gates for an experiment
python scripts/run/guard_gpu2.py check --experiment det.energy.global_delta_u.c3b.balanced.001

# 3. Dry-run: print the command the guard would run
python scripts/run/guard_gpu2.py dry-run --experiment det.energy.global_delta_u.c3b.balanced.001
```

All three commands are safe to run repeatedly.  Only `dry-run --execute` starts
a subprocess, and it still never signals or interferes with existing processes.

## Remote launch procedure

1. Ensure local changes are committed and pushed or pulled to the remote
   workspace.
2. SSH to the remote host:
   ```bash
   ssh ps@122.51.19.136
   cd /home/ps/lzz/manifold-detection-energy-transport
   source /home/ps/anaconda3/etc/profile.d/conda.sh
   conda activate RLimage
   export PYTHONPATH=/home/ps/lzz/manifold-detection-energy-transport:$PYTHONPATH
   export CUDA_VISIBLE_DEVICES=2
   ```
3. Run the guard in check mode:
   ```bash
   /home/ps/anaconda3/envs/RLimage/bin/python scripts/run/guard_gpu2.py check \
     --experiment <experiment_id>
   ```
4. If the guard passes, run the printed dry-run command inside `nohup` or a
   tmux session, redirecting stdout/stderr to `runs/<run_name>/launcher.log`.
5. Record the resulting PID, command hash, run ID, and manifest path in the
   launch record directory (`runs/gpu2_launches/`).  The guard writes this
   automatically when `--execute` is used.

## Logs and artifacts

- Runtime logs: `runs/<run_name>/launcher.log` (local/runtime only, never Git).
- Launch records: `runs/gpu2_launches/*.json` (local/runtime only).
- Result manifests: `spectral_detection_posttrain/configs/registry/artifacts/*.json`
  (tracked after review).
- Environment snapshots:
  `spectral_detection_posttrain/configs/registry/environments/remote_gpu2_py310_cu121.json`
  (tracked).

## Failure handling

| Failure | Action |
|---|---|
| `memory.free <= 8192 MiB` | Wait for GPU2 load to drop or choose a different time. Do not kill another process. |
| Dirty Git tree | Commit or stash locally, then pull/sync to remote. |
| Experiment not authorized | Check `research_lines.json` status; do not bypass the guard. |
| Missing dependency on remote | Fix locally in `requirements.txt` or environment snapshot, then re-validate in CI. Do not install on the remote host. |
| Process dies or hangs | Investigate logs; cancellation may only target a PID launched by this run and verified by ownership. |

## Link to registries

- Experiments: `spectral_detection_posttrain/configs/registry/experiments.json`
- Research lines: `spectral_detection_posttrain/configs/registry/research_lines.json`
- Guard source: `scripts/run/guard_gpu2.py`
- Tests: `tests/contracts/test_gpu2_guard.py`
