# Execution Environments

This page is maintained prose; it documents the committed machine-readable
snapshots and how to regenerate them. The snapshots are the authoritative
records:

- [local_windows_py310.json](../../spectral_detection_posttrain/configs/registry/environments/local_windows_py310.json) — local Windows development host
- [remote_gpu2_py310_cu121.json](../../spectral_detection_posttrain/configs/registry/environments/remote_gpu2_py310_cu121.json) — remote GPU training host

Both are produced by the read-only collector
[scripts/dev/snapshot_environment.py](../../scripts/dev/snapshot_environment.py)
and are validated by
[tests/contracts/test_environment_snapshot.py](../../tests/contracts/test_environment_snapshot.py).
`captured_at_utc` is an observed value: tests validate its format but never
compare it for equality, so re-capturing a snapshot never breaks the suite.

## Local Windows development host (`host_alias: local:windows-dev`)

- Python 3.10.20 (conda env `RLimage`), Windows 10, AMD64.
- torch 2.1.0+cu121, torchvision 0.16.0+cu121, numpy 1.26.4, Pillow 12.2.0,
  pytest 9.0.3.
- CUDA is available locally (single consumer GPU); the maintained test suite
  is the CPU-safe path and must not depend on a local GPU.
- scikit-learn is present in this interpreter, but it remains an **optional
  legacy analysis dependency** (legacy raw-iFFT verifier and dimensionality
  modules) and is deliberately not part of the maintained install.

## Remote GPU host (`host_alias: remote:manifold`)

- Python 3.10.20 (conda env `RLimage`), Linux 6.8, x86_64.
- torch 2.1.0+cu121, torchvision 0.16.0+cu121, numpy 1.26.4, Pillow 12.2.0,
  pytest 9.1.1; scikit-learn is absent on the remote.
- The host has 4 GPUs, but **training is restricted to physical GPU2
  (index 2, NVIDIA GeForce RTX 4090)**. Only jobs launched with
  `CUDA_VISIBLE_DEVICES=2` may run, and only after a read-only
  `nvidia-smi -i 2` check shows `memory.free > 8192 MiB`. Never stop,
  restart, or signal processes already using GPU2.
- GPU identity in the snapshot comes from a read-only
  `nvidia-smi --query-gpu=index,name,memory.total,memory.free -i 2` query;
  package and platform facts come from an inline python one-liner over SSH.
  No files are written remotely, nothing is installed remotely, and no CUDA
  workload is initialized during capture.

## Privacy and secret policy

Snapshots never record user names or secrets. User-profile path components
are redacted uniformly (`C:\Users\<name>` → `C:\Users\<redacted>`,
`/home/<name>` → `/home/<user>`); host identity is carried by the
`host_alias` field instead. The collector aborts if any collected value
matches a token/password/private-key pattern, and the contract tests re-scan
both committed snapshots for leaks.

## Path aliases used in artifact manifests

Reviewed artifact manifests under
[spectral_detection_posttrain/configs/registry/artifacts/](../../spectral_detection_posttrain/configs/registry/artifacts/)
reference files by host alias instead of absolute paths:

- `remote:manifold` — the remote workspace
  `/home/<user>/lzz/manifold-detection-energy-transport` on the GPU host
  above; paths below the alias are workspace-relative.
- `remote:rlimage` — the separate legacy workspace `/home/<user>/lzz/RLimage`
  on the same host. It is outside the permitted read scope; refs with this
  alias carry hashes copied from evidence plus an explicit `missing_evidence`
  entry, and their bytes are not re-verified.
- `derived:` — a value derived from recorded evidence (for example a
  deterministic split's image-id digest), not a standalone file; such refs
  carry `size_bytes: null`.

## requirements.txt and optional legacy dependencies

[requirements.txt](../../requirements.txt) covers the **maintained runtime /
CPU test path** only. Optional legacy analysis dependencies (scikit-learn)
are documented there as comments and recorded in the snapshots as semantic
notes — they are never installed by default, and no unverified CUDA wheel
URLs are pinned.

## Continuous integration checks

The maintained CPU check workflow `.github/workflows/ci.yml` runs on every
push and pull request. It performs five non-destructive repository checks:

1. `python scripts/dev/validate_repository_state.py` — verifies git state,
   required directories, and environment snapshot presence.
2. `python scripts/dev/check_import_boundaries.py` — enforces that maintained
   packages do not import from legacy paths.
3. `python scripts/dev/check_docs.py` — reports documentation hygiene issues;
   errors fail the check, historical warnings are reported but do not fail.
4. `python scripts/dev/generate_research_docs.py --check` — validates that
   generated research docs are up to date with their sources.
5. `pytest tests -q` — runs the maintained CPU test suite.

The workflow installs from `requirements.txt`, so it covers only the maintained
runtime. Tests that require optional legacy dependencies (for example the
legacy raw-iFFT verifier and dimensionality modules that import scikit-learn)
are skipped automatically when those packages are absent; this is expected and
does not indicate a failure.

## Regenerating the snapshots

Local (Windows, Git Bash):

```bash
PYTHONUTF8=1 E:/anaconda/01/envs/RLimage/python.exe \
  scripts/dev/snapshot_environment.py \
  --host-alias local:windows-dev \
  --output spectral_detection_posttrain/configs/registry/environments/local_windows_py310.json
```

Remote (read-only SSH; index 2 only):

```bash
ssh ps@122.51.19.136 "cd /home/ps/lzz/manifold-detection-energy-transport && \
  /home/ps/anaconda3/envs/RLimage/bin/python -c '<inline collector one-liner>'"
ssh ps@122.51.19.136 \
  "nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader -i 2"
```

Merge the two outputs locally (redact `/home/ps` → `/home/<user>`, keep the
GPU2 restriction notes) and write the remote snapshot with sorted keys,
two-space indent, LF endings, and one trailing newline.
