"""Validate that no runtime artifacts are tracked in the repository.

Enumerates tracked files via ``git ls-files`` and reports paths that should
never be tracked: dataset payloads, checkpoints/weights, caches, logs,
anything under ``runs/``, ``.agent_reports/`` or ``data/``, and
secret-pattern files. The check is read-only; it never modifies or deletes
files.

Usage:
    python scripts/dev/validate_repository_state.py          # human-readable
    python scripts/dev/validate_repository_state.py --json   # JSON report

Exit code is 1 iff violations exist, else 0.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# User-owned documents that must never become auto-stage targets even though
# they are untracked. Kept as data so staging tooling and contract tests can
# share a single source of truth.
EXPLICIT_EXCLUSIONS = frozenset(
    {
        "docs/energy_transport_vs_fpn_sm_analysis.md",
    }
)

# Directory-based rules, anchored at the repository root. Generated/runtime
# payload directories only; maintained result records under ``output/`` and
# scratch renders under ``tmp/`` are intentionally not covered here.
FORBIDDEN_DIR_PREFIXES = (
    "runs/",
    ".agent_reports/",
    "data/",
)

# Exact-path allowlist for small, hash-locked test fixtures that would
# otherwise match an extension rule. Every entry must be synthetic, CPU-safe,
# tens of KB at most, and sha256-pinned by
# ``tests/fixtures/checkpoints/energy_transport_checkpoint_manifest.json``.
# Adding an entry is a reviewed, deliberate act — never widen this to a
# directory prefix.
ALLOWED_FIXTURE_PATHS = frozenset(
    {
        "tests/fixtures/checkpoints/action_local_transport_head_seed42.pt",
        "tests/fixtures/checkpoints/spatial_candidate_energy_head_seed43.pt",
        "tests/fixtures/checkpoints/context_only_candidate_energy_head_seed44.pt",
        "tests/fixtures/checkpoints/action_benefit_energy_head_seed45.pt",
    }
)

# Extension-based rules: model weights, caches and logs. Deliberately does
# not include generic ``*.json`` (result records, audit data and configs are
# legitimately tracked) or document formats like ``*.png``/``*.pdf``/``*.csv``.
FORBIDDEN_EXTENSIONS = frozenset(
    {
        ".pth",
        ".pt",
        ".ckpt",
        ".onnx",
        ".safetensors",
        ".pkl",
        ".joblib",
        ".h5",
        ".hdf5",
        ".npz",
        ".npy",
        ".log",
        ".pem",
        ".key",
    }
)

# Name-based rules: secret-pattern files. Exact basenames only, so templates
# such as ``.env.example`` stay allowed.
FORBIDDEN_NAMES = frozenset(
    {
        ".env",
        "id_rsa",
        "id_dsa",
        "id_ecdsa",
        "id_ed25519",
        "credentials",
        "credentials.json",
        "secrets.json",
        "secrets.yaml",
        "secrets.yml",
    }
)


def is_forbidden(path: str) -> bool:
    """Return True if a repo-relative POSIX path should never be tracked."""
    normalized = path.replace("\\", "/")
    if normalized in ALLOWED_FIXTURE_PATHS:
        return False
    for prefix in FORBIDDEN_DIR_PREFIXES:
        if normalized.startswith(prefix):
            return True
    name = normalized.rsplit("/", 1)[-1].lower()
    if name in FORBIDDEN_NAMES:
        return True
    return any(name.endswith(ext) for ext in FORBIDDEN_EXTENSIONS)


def list_tracked_files(root: Path) -> list[str]:
    """Return tracked repo-relative POSIX paths via ``git ls-files``."""
    result = subprocess.run(
        ["git", "-c", "core.quotepath=false", "ls-files"],
        cwd=root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    return [line for line in result.stdout.splitlines() if line]


def find_tracked_runtime_artifacts(root) -> list:
    """Return sorted repo-relative POSIX paths of tracked files that match
    forbidden runtime-artifact patterns. Empty list means the repository
    state is clean."""
    violations = [path for path in list_tracked_files(Path(root)) if is_forbidden(path)]
    return sorted(violations)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Report tracked files that should never be tracked "
        "(dataset payloads, checkpoints, caches, logs, runs/, "
        ".agent_reports/, secrets)."
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="print a deterministic JSON report instead of human-readable text",
    )
    args = parser.parse_args(argv)

    tracked = list_tracked_files(ROOT)
    violations = sorted(path for path in tracked if is_forbidden(path))

    if args.json:
        payload = {"checked": len(tracked), "violations": violations}
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"Checked {len(tracked)} tracked files under {ROOT}")
        if violations:
            print(f"Found {len(violations)} tracked runtime artifact(s):")
            for path in violations:
                print(f"  {path}")
        else:
            print("OK: no forbidden runtime artifacts are tracked.")

    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main())
