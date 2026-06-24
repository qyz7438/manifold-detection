# Canonical Runner

Canonical runner responsibilities:

- validate config schema,
- validate model/checkpoint compatibility,
- prevent unknown model names,
- prevent random-init fallback in formal experiments,
- force clean eval settings,
- record git commit, config hash, checkpoint hash, torch/cuda versions,
- write registry-ready metadata.

Related source:

- [docs/canonical_runner_hardening_note.md](../docs/canonical_runner_hardening_note.md)
- [docs/refactor_directory_versioning_plan.md](../docs/refactor_directory_versioning_plan.md)
