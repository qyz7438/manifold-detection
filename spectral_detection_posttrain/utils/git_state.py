from __future__ import annotations

import subprocess
from pathlib import Path


def _run(cmd: list[str], cwd: Path | str | None = None) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return proc.returncode, proc.stdout.strip(), proc.stderr.strip()
    except Exception as exc:  # pragma: no cover
        return -1, "", str(exc)


def get_git_state(repo_root: Path | str | None = None) -> dict:
    """Return a reproducibility stamp for the current git working tree.

    The dictionary contains:
      - ``commit``: full SHA of HEAD (or empty string if not a repo).
      - ``dirty``: True if the working tree has uncommitted changes.
      - ``branch``: active branch name (or ``HEAD`` when detached).
      - ``describe``: output of ``git describe --tags --always``.
    """
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[2]
    repo_root = Path(repo_root)

    commit = ""
    dirty = False
    branch = ""
    describe = ""

    rc, out, _ = _run(["git", "rev-parse", "HEAD"], cwd=repo_root)
    if rc == 0:
        commit = out

    rc, out, _ = _run(["git", "status", "--porcelain"], cwd=repo_root)
    if rc == 0:
        dirty = bool(out.strip())

    rc, out, _ = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_root)
    if rc == 0:
        branch = out

    rc, out, _ = _run(["git", "describe", "--tags", "--always"], cwd=repo_root)
    if rc == 0:
        describe = out

    return {
        "commit": commit,
        "dirty": dirty,
        "branch": branch,
        "describe": describe,
    }


def require_clean_tree(repo_root: Path | str | None = None, strict: bool = False) -> bool:
    """Return True if the git tree is clean; optionally raise if dirty."""
    state = get_git_state(repo_root)
    if state["dirty"]:
        if strict:
            raise RuntimeError(
                f"Git working tree is dirty (commit {state['commit']}). "
                "Commit or stash changes before a canonical run."
            )
        return False
    return True
