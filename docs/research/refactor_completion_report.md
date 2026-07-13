# Repository Research-State Refactor — Completion Report

> **Scope:** refactor `manifold-detection` into an auditable research repository
> whose scientific status, experiment definitions, artifact provenance,
> compatibility boundaries, and documentation all agree, without changing
> detector algorithms or rewriting historical evidence.
> **Base commit:** `7e1f2404699df997883678274491e88d3ec253c0`
> **Refactor branch:** `codex/repository-research-state-refactor`
> **No new research result is claimed by this refactor.**

## What changed (commit summary)

| Commit | Task | Summary |
|---|---|---|
| `2533d13` | plan | Add repository research-state refactor plan |
| `0065de4` | baseline | Lock repository refactor baseline |
| `57b198c` | T1 | Lock maintained repository state |
| `987a134` | T2 | Add research status registry |
| `fdf4374` | T4 | Add immutable artifact manifests |
| `fc20975` | T9 | Inventory research scripts and entrypoints |
| `f620072` | T3 | Register versioned research experiments |
| `dbff3a4` | T5 | Record explicit evaluation scope |
| `92855f0` | T8 | Align repository with research registries |
| `c3652e6` | T11 | Record reproducible execution environments |
| `b0a1d73`–`a0ed29e` | T10 | Backfill critical experiment provenance |
| `c405ea1` | T6 | Track canonical experiment lifecycle |
| `3f69594` | T12 | Define energy transport package boundaries |
| `b6d3628` | T7 | Add explicit experiment dispatcher |
| `a12684f`–`636e31b` | T13–T15 | Isolate action/native/policy/endpoint/diagnostic modules |
| `6f52c73` | T15 | Add structured detection trainer protocols |
| `35d5e6f`–`916739c` | T16 | Archive scratch/frozen research scripts |
| `f9c6215` | T17 | Enforce canonical import boundaries |
| `8dd2db0` | T18 | Validate current research documentation |
| `298edc2` | T19 | Add maintained CPU repository checks |
| `cb7b5a6` | T20 | Add guarded GPU2 execution contract |

## Test evidence

### Full maintained suite

```text
E:/anaconda/01/envs/RLimage/python.exe -m pytest tests -q
1528 passed, 3 skipped, 2 xfailed, 1 warning
```

### Targeted compatibility / dispatcher / contract proofs

```text
pytest tests/compatibility tests/experiments tests/contracts/test_trainer_protocols.py \
  tests/contracts/test_trainer_protocols_integration.py \
  tests/contracts/test_energy_transport_import_boundaries.py \
  tests/contracts/test_repository_state.py tests/test_canonical_runner_e2e.py -q
910 passed, 3 skipped, 2 xfailed, 1 warning
```

### Repository hygiene checks

| Check | Command | Result |
|---|---|---|
| Tracked runtime artifacts | `scripts/dev/validate_repository_state.py --json` | 1053 files checked, 0 violations |
| Import boundaries | `scripts/dev/check_import_boundaries.py --json` | 8 allowlist entries, 0 unallowlisted, 0 stale |
| Documentation links | `scripts/dev/check_docs.py` | 0 errors; historical warnings only |
| Generated docs drift | `scripts/dev/generate_research_docs.py --check` | up to date |
| Git whitespace | `git diff --check` | clean |

## Artifact manifest hashes

All reviewed artifact manifests remained byte-stable through the refactor and
carry the following SHA-256 digests at `cb7b5a6`:

```text
spectral_detection_posttrain/configs/registry/artifacts/det.energy.dense_endpoint.absolute.001.json sha256 9b93d5de0b015248c610c6f8d5b2e911bc78274a0e227a7867247c05cfb64c9c
spectral_detection_posttrain/configs/registry/artifacts/det.energy.dense_endpoint.cleanval.001.json sha256 b1a1e35b64397b666cb8aaa8e499f0a71ab8f7a7f52b9da87e98e3c1cfa5955d
spectral_detection_posttrain/configs/registry/artifacts/det.energy.dense_endpoint.geometry_control.001.json sha256 e9a294f1a1707d4a7e6b0b792b6c435717e34b424a7807189593f2b371b693a0
spectral_detection_posttrain/configs/registry/artifacts/det.energy.dense_endpoint.shift_audit.001.json sha256 ffe1349e5f9a5b46965ef633a9e0ba2ab9c85349916b4bec8946d6ac90a07f1a
spectral_detection_posttrain/configs/registry/artifacts/det.energy.dense_local_delta_family_audit.001.json sha256 36657a98ca6535daf7b525a10c64bb3ab368d16e2752f39329e032039ac5ad11
spectral_detection_posttrain/configs/registry/artifacts/det.energy.dense_local_delta_family_prior.001.json sha256 9630e2f3e90a38cb6a537c3fea5306dd99bbf8a5c6d7154af4de01e4375ecc28
spectral_detection_posttrain/configs/registry/artifacts/det.energy.dense_local_delta_learner.001.json sha256 c020e0e998b6ee54e273306d605105739b11a610803fb612ff8b0ee34c5f5653
spectral_detection_posttrain/configs/registry/artifacts/det.energy.dense_local_delta_stats.002.json sha256 3ab0dd3811b288b09b908a2c4b490157ec5e8ccd67618c22a63cfa16a8552769
spectral_detection_posttrain/configs/registry/artifacts/det.energy.global_delta_u.c3b.balanced.001.json sha256 92366b78d4c8fa6a5d65b6e988d0cf666f29bee2d7d469250553ef7fb02028a1
spectral_detection_posttrain/configs/registry/artifacts/det.energy.native_topology.d2b.001.json sha256 3c1ffe28c276de54d81ba9eea10652ddd142cdc1273b6f2e587af5745540683f
spectral_detection_posttrain/configs/registry/artifacts/det.energy.post_nms_suppress.e1.001.json sha256 cc6a35c734b7d9b6013bb92f0c2ff6a6d497affc7d52ec350661de54553c2e5c
spectral_detection_posttrain/configs/registry/artifacts/det.energy.set_policy.m1.001.json sha256 35903ce4991b204b6299160b01d3e9cc6a315b0ae04c818d481de842e38e859c
spectral_detection_posttrain/configs/registry/artifacts/native_zero_parity_baseline.json sha256 0c5712b8e44fb9a0ee54902bbea11410c10c6cd4ac60180719dbb2e4c0730204
spectral_detection_posttrain/configs/registry/artifacts/native_zero_parity_fullft.json sha256 588fab40be106cf04f562d22a7687f4eb6af1f8ebc789633a978f4bbbd171b26
spectral_detection_posttrain/configs/registry/artifacts/nwpu_mob_strong_cosine_s42_bs8_36ep.json sha256 b5e4ea4bb3118a079a7e43852f6bc6d20205d3112106b30192f4caf8c4486c39
```

No tracked checkpoint, cache, dataset payload, log, or secret-pattern file was
introduced (`validate_repository_state.py` reports zero violations).

## Scientific status after refactor

The research-status registry
(`spectral_detection_posttrain/configs/registry/research_lines.json`) still
authorizes **zero** active experiments.  All prior lines retain their previous
status:

- `energy_transport.native_actions.c1_e1` — `frozen`
- `energy_transport.dense_absolute_endpoint` — `frozen`
- `energy_transport.local_delta_q` — `frozen`
- `energy_transport.residual_content_protocol` — `blocked`
- `manifold.prototype_attraction` — `diagnostic`
- `manifold.intrinsic_dimension` — `diagnostic`
- remaining geometry/structure/FFT/verifier lines — `diagnostic` or `historical`

No experiment was re-tuned, widened to new seeds, or promoted to validated.

## Remote verification

Read-only SSH to `ps@122.51.19.136` succeeded:

- Remote workspace: `/home/ps/lzz/manifold-detection-energy-transport`, HEAD
  `7e1f24046` (base commit, not yet updated with the refactor branch).
- `nvidia-smi -i 2` returned: `2, NVIDIA GeForce RTX 4090, 49140, 48628`
  (48628 MiB free on physical GPU2).
- The new guard script `scripts/run/guard_gpu2.py` is not yet present on the
  remote; full remote guard verification (`query`/`check`/`dry-run`) requires
  pushing the refactor branch and pulling it on the remote host.

## Independent review findings

Self-review (main orchestrator) of the full `7e1f240..cb7b5a6` diff:

- **Accepted:** all package moves are accompanied by flat compatibility shims;
  state-dict keys and public signatures are preserved by existing tests;
  import-boundary allowlist is explicit and has owners/removal tasks;
  generated docs match registries; CI is CPU-only and does not claim validation.
- **Rejected:** none.
- **No behavior drift detected** in model arithmetic, random consumption,
  proposal order, BoxCoder, thresholds, NMS, matching, top-K, or metric
  definitions.

## Residual risks

1. **Remote sync.** The remote workspace is still at the base commit. Pushing
   the refactor branch and pulling on the remote is required before the guard
   can be exercised and before remote CPU tests can be re-run.
2. **CI environment.** GitHub-hosted runners install from `requirements.txt`;
   any platform-specific compiled dependency may need adjustment when CI first
   runs.
3. **Historical doc warnings.** `check_docs.py` tolerates warnings in old plans;
   future doc edits that create new broken links are still blocked by the
   contract test.

## Merge recommendation

Open a **draft PR** targeting `codex/energy-guided-roi-transport` (not `main`)
with this branch.  After CI passes and any required reviews, merge with a
non-squash strategy so that the per-task commits and the artifact hash record
remain auditable.  A later consolidation PR from the research branch to `main`
should be planned separately.
