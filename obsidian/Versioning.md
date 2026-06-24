# Versioning

Source document:

- [docs/versioning_scheme.md](../docs/versioning_scheme.md)

New format:

```text
<task>.<method>.<stage>.<sequence>
```

Examples:

- `det.rlvr.clean.001`
- `det.dpo.smoke.001`
- `det.afm.validated.001`
- `seg.afm.proto.001`
- `shared.runner.clean.001`

Historical `round*` names become aliases, not the primary version system.

Current assigned aliases:

- `det.dpo.smoke.001` -> `round2219_pre_nms_dpo_rescue_w003_5ep`
