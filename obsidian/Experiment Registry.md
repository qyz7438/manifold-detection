# Experiment Registry

Purpose:

Index `runs/` so results can be queried without manually opening hundreds of directories.

Minimum fields:

- run name
- version id
- task
- method
- clean eval flag
- pollution status
- baseline metrics
- best metrics
- final metrics
- checkpoint paths
- config hash
- checkpoint hash

Related:

- [[Versioning]]
- [[Canonical Runner]]
- [[NWPU Clean Posttrain]]
