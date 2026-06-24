# Versioning Scheme

The old `round2xx` numbering is overloaded: it mixes dataset changes, architecture changes, post-training objectives, bug fixes, smoke tests, and invalidated runs. New versions should encode method family, task, and maturity.

## New Version Format

Use:

```text
<task>.<method>.<stage>.<sequence>
```

Where:

- `task`: `det` for detection, `seg` for segmentation, `shared` for reusable infrastructure.
- `method`: `afm`, `rlvr`, `dpo`, `signal`, `runner`, `eval`.
- `stage`: `proto`, `smoke`, `clean`, `sweep`, `validated`, `archived`.
- `sequence`: zero-padded integer.

Examples:

```text
det.rlv r.clean.001   # invalid: contains a space, do not use
det.rlvr.clean.001    # valid
det.dpo.smoke.003
seg.afm.proto.001
shared.runner.validated.001
```

## Mapping From Historical Rounds

| Historical range | New family | Status |
|---|---|---|
| Round 1-2.5 | `det.rlvr.proto.*` | external FFT verifier mostly negative |
| Round 2.6-2.18 | `det.afm.validated.*` | Penn-Fudan AFM positive result |
| Round 2.21-2.58 | `det.afm.sweep.*`, `det.dpo.archived.*` | mixed, many audit-invalid results |
| Round 2.100-2.129 | `det.rlvr.clean.*`, `det.dpo.clean.*` | NWPU clean/posttrain reopening |
| Round 2100+ | `det.signal.clean.*`, `det.rlvr.clean.*`, `det.dpo.smoke.*` | NWPU signal and posttrain runs |
| Plan 4.x | `seg.afm.proto.*`, `seg.rlvr.proto.*`, `seg.dpo.proto.*` | segmentation path not yet mature |

## Recommended Current Series

Current active lines:

| Series | Meaning | Starting version |
|---|---|---|
| `shared.runner.*` | canonical runner, schema, metadata, registry | `shared.runner.clean.001` |
| `det.signal.*` | FFT/manifold/geometry verifier diagnostics | `det.signal.clean.001` |
| `det.rlvr.*` | clean NWPU rescue/RLVR post-training | `det.rlvr.clean.001` |
| `det.dpo.*` | clean NWPU DPO/preference post-training | `det.dpo.smoke.001` |
| `det.afm.*` | detection AFM architecture and post-training | `det.afm.validated.001` |
| `seg.afm.*` | segmentation AFM variants | `seg.afm.proto.001` |
| `seg.rlvr.*` | segmentation RLVR with verifiable dense rewards | `seg.rlvr.proto.001` |
| `seg.dpo.*` | segmentation DPO over masks/crops/patches | `seg.dpo.proto.001` |

## Rules

1. A new run must write both the historical run name and the new version id.
2. A version id must map to one resolved config.
3. A validated version must have clean eval, checkpoint hash, git commit, and reproduction command.
4. Historical `round*` names remain allowed only as aliases.
5. Invalid or polluted runs are not deleted; they are marked `archived` or `polluted`.
