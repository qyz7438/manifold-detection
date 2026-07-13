"""Canonical import-boundary contract for the whole repository (Task 17).

AST-only static analysis — modules under test are never imported, so import-time
side effects cannot hide violations.  Rules are the six boundary constraints from
the refactor plan.  Known violations live in the checker allowlist with an owner
and removal task; the ratchet fails on any new violation or any stale allowlist
entry.

The checker module in ``scripts/dev/check_import_boundaries.py`` is the single
source of truth for rule tables; this test only asserts the ratchet and the CLI
shape.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
CHECKER_PATH = ROOT / "scripts" / "dev" / "check_import_boundaries.py"


def load_checker():
    """Load the standalone boundary checker as a module."""
    assert CHECKER_PATH.exists(), f"{CHECKER_PATH} has not been created"
    spec = importlib.util.spec_from_file_location("check_import_boundaries", CHECKER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def checker():
    return load_checker()


# ---------------------------------------------------------------------------
# Allowlist ratchet
# ---------------------------------------------------------------------------


def test_allowlist_entries_are_current_violations(checker):
    """Every allowlisted violation must still be present in the code."""
    violations = {v.key() for v in checker.find_violations()}
    stale = set(checker.ALLOWLIST) - violations
    assert not stale, (
        "allowlist entries no longer violate the rules; remove them: "
        + "; ".join(f"{p} -> {t}" for p, t in sorted(stale))
    )


def test_no_new_violations_beyond_allowlist(checker):
    """No new cross-boundary imports appear outside the explicit allowlist."""
    violations = checker.find_violations()
    new = [v for v in violations if v.key() not in checker.ALLOWLIST]
    assert not new, (
        "new canonical import-boundary violations (add to allowlist with owner + removal task if justified): "
        + "; ".join(f"{v.path} imports {v.target} ({v.rule})" for v in new)
    )


def test_allowlist_entries_have_owner_and_removal_task(checker):
    """Each allowed exception names an owner and a concrete removal task."""
    for (path, target), meta in sorted(checker.ALLOWLIST.items()):
        owner = meta.get("owner")
        removal = meta.get("removal_task")
        assert owner and owner.strip(), f"{path} -> {target} has no owner"
        assert removal and removal.strip(), f"{path} -> {target} has no removal_task"


def test_allowlist_stays_below_guardrail(checker):
    """The allowlist is a shrinking ratchet, not a backlog."""
    assert len(checker.ALLOWLIST) <= 15, (
        f"allowlist has grown to {len(checker.ALLOWLIST)} entries; "
        "stop and resolve before adding more"
    )


# ---------------------------------------------------------------------------
# Checker CLI
# ---------------------------------------------------------------------------


def test_checker_cli_human_readable_reports_no_violations():
    """scripts/dev/check_import_boundaries.py exits 0 and prints summary."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(ROOT), env.get("PYTHONPATH", "")) if part
    )
    result = subprocess.run(
        [sys.executable, str(CHECKER_PATH)],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    assert result.returncode == 0, (
        "human-readable checker failed:\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "violation" in result.stdout.lower() or "clean" in result.stdout.lower()


def test_checker_cli_json_reports_no_violations():
    """--json emits a parseable object with violations/allowlist/stale keys."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(ROOT), env.get("PYTHONPATH", "")) if part
    )
    result = subprocess.run(
        [sys.executable, str(CHECKER_PATH), "--json"],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    assert result.returncode == 0, (
        "JSON checker failed:\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    data = json.loads(result.stdout)
    assert "violations" in data
    assert "unallowlisted" in data
    assert data["unallowlisted"] == [], data["unallowlisted"]
    assert data.get("stale_allowlist", []) == [], data.get("stale_allowlist")


# ---------------------------------------------------------------------------
# Import smoke: action contracts / dense endpoint / canonical runner / registry
# ---------------------------------------------------------------------------

SMOKE_MODULES = (
    "spectral_detection_posttrain.methods.energy_transport.action.contracts",
    "spectral_detection_posttrain.methods.energy_transport.endpoint.dense_endpoint",
    "spectral_detection_posttrain.experiments.canonical_runner",
    "spectral_detection_posttrain.experiments.registry",
)

SMOKE_SCRIPT = "\n".join([
    "import builtins",
    "import os",
    "import socket",
    "import sys",
    "import tempfile",
    "from pathlib import Path",
    "",
    "# --- trackers ---------------------------------------------------------------",
    "",
    "_open_calls = []",
    "_socket_attempts = []",
    "",
    "_orig_open = builtins.open",
    "",
    "def _tracked_open(file, *args, **kwargs):",
    "    path = str(file)",
    "    mode = None",
    "    if args:",
    "        mode = args[0]",
    "    if 'mode' in kwargs:",
    "        mode = kwargs['mode']",
    "    _open_calls.append((path, mode))",
    "    return _orig_open(file, *args, **kwargs)",
    "",
    "builtins.open = _tracked_open",
    "",
    "_orig_socket_connect = socket.socket.connect",
    "_orig_socket_connect_ex = socket.socket.connect_ex",
    "_orig_socket_sendto = socket.socket.sendto",
    "_orig_getaddrinfo = socket.getaddrinfo",
    "",
    "def _tracked_connect(self, address):",
    "    _socket_attempts.append(('connect', address))",
    "    return _orig_socket_connect(self, address)",
    "",
    "def _tracked_connect_ex(self, address):",
    "    _socket_attempts.append(('connect_ex', address))",
    "    return _orig_socket_connect_ex(self, address)",
    "",
    "def _tracked_sendto(self, data, address):",
    "    _socket_attempts.append(('sendto', address))",
    "    return _orig_socket_sendto(self, data, address)",
    "",
    "def _tracked_getaddrinfo(host, port, *args, **kwargs):",
    "    _socket_attempts.append(('getaddrinfo', (host, port)))",
    "    return _orig_getaddrinfo(host, port, *args, **kwargs)",
    "",
    "socket.socket.connect = _tracked_connect",
    "socket.socket.connect_ex = _tracked_connect_ex",
    "socket.socket.sendto = _tracked_sendto",
    "socket.getaddrinfo = _tracked_getaddrinfo",
    "",
    "# --- imports ----------------------------------------------------------------",
    "",
    "%s",
    "",
    "# --- assertions -------------------------------------------------------------",
    "",
    "import torch",
    'assert not torch.cuda.is_initialized(), "import touched CUDA"',
    "",
    'ignored_data = [p for p, _ in _open_calls if "data/" in p.replace(chr(92)+chr(92), "/")]',
    'ignored_runs = [p for p, _ in _open_calls if "runs/" in p.replace(chr(92)+chr(92), "/")]',
    'writes = [p for p, m in _open_calls if isinstance(m, str) and any(c in m for c in "wax+")]',
    "",
    'assert not ignored_data, f"module import read from data/: {ignored_data}"',
    'assert not ignored_runs, f"module import read from runs/: {ignored_runs}"',
    'assert not writes, f"module import wrote files: {writes}"',
    'assert not _socket_attempts, f"module import used sockets: {_socket_attempts}"',
    "",
    "# cwd is a fresh temp dir; nothing should be written there by import.",
    'cwd_files = [p for p in os.listdir(os.getcwd()) if not p.startswith(".")]',
    'assert not cwd_files, f"module import wrote into cwd: {cwd_files}"',
    "",
    'print("OK")',
])
SMOKE_SCRIPT = SMOKE_SCRIPT % "\n".join(f"import {m}" for m in SMOKE_MODULES)


def test_canonical_import_smoke_performs_no_io_or_cuda_side_effects(tmp_path):
    """Importing key canonical modules in a fresh interpreter is side-effect-free."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(ROOT), env.get("PYTHONPATH", "")) if part
    )
    result = subprocess.run(
        [sys.executable, "-c", SMOKE_SCRIPT],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(tmp_path),
        timeout=180,
    )
    assert result.returncode == 0, (
        "import smoke failed:\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert result.stdout.strip().endswith("OK")
