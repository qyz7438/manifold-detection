"""Contract tests for the committed execution-environment snapshots.

The snapshots under ``spectral_detection_posttrain/configs/registry/environments/``
are observed records of the local Windows dev host and the remote GPU2 host.
They exist so drift is visible and reproducibility claims can be checked.

Drift safety: ``captured_at_utc`` is an observed value. It is validated for
format only and is NEVER compared for equality anywhere in this module, so a
re-captured snapshot at a later time does not break the suite.

The tests are read-only: they never install packages, mutate conda, or touch
the remote host. The collector self-test runs the local collector with no
``--output`` flag, which only prints JSON to stdout.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
COLLECTOR = ROOT / "scripts" / "dev" / "snapshot_environment.py"
REGISTRY = ROOT / "spectral_detection_posttrain" / "configs" / "registry" / "environments"
LOCAL_SNAPSHOT = REGISTRY / "local_windows_py310.json"
REMOTE_SNAPSHOT = REGISTRY / "remote_gpu2_py310_cu121.json"
SNAPSHOTS = (LOCAL_SNAPSHOT, REMOTE_SNAPSHOT)

REQUIRED_TOP_LEVEL = (
    "schema_version",
    "captured_at_utc",
    "host_alias",
    "os",
    "python",
    "cuda",
    "packages",
    "optional_dependencies",
    "notes",
)
REQUIRED_PACKAGES = ("torch", "torchvision", "numpy", "pillow", "pytest")
OPTIONAL_DEP_STATES = {"present", "absent", "unavailable"}

# PEP 440-ish version strings such as "3.10.20", "2.1.0+cu121", "1.26.4".
VERSION_RE = re.compile(r"^\d+(\.\d+){1,3}(\+[0-9A-Za-z._-]+)?$")
TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")

# Secret / identity leakage patterns. User-profile paths are only allowed in
# their redacted forms ("<redacted>" / "<user>").
SECRET_PATTERNS = (
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"(?i)(api[_-]?key|secret|token|password|passwd)\s*[:=]\s*\S+"),
    re.compile(r"(?i)[A-Z]:[\\/]+Users[\\/]+(?!<redacted>)[^\\/]+"),
    re.compile(r"/home/(?!<user>)[^/]+"),
    re.compile(r"(?<!:)/Users/(?!<redacted>)[^/]+"),
)


def _load_snapshot(path: Path) -> dict:
    assert path.exists(), f"environment snapshot is missing: {path}"
    text = path.read_text(encoding="utf-8")
    return json.loads(text)


def _validate_version_field(value, label: str) -> None:
    assert isinstance(value, str), f"{label} must be a string, got {type(value).__name__}"
    assert value == "unavailable" or VERSION_RE.match(value), (
        f"{label} has unexpected version shape: {value!r}"
    )


def validate_snapshot(data: dict, label: str) -> None:
    """Validate schema and key invariants. Timestamp is format-checked only."""
    for field in REQUIRED_TOP_LEVEL:
        assert field in data, f"{label}: missing top-level field {field!r}"

    assert isinstance(data["schema_version"], str) and data["schema_version"]
    assert isinstance(data["captured_at_utc"], str) and TIMESTAMP_RE.match(
        data["captured_at_utc"]
    ), f"{label}: captured_at_utc must be an ISO-8601 UTC timestamp"
    assert isinstance(data["host_alias"], str) and data["host_alias"]

    os_info = data["os"]
    assert isinstance(os_info, dict), f"{label}: os must be a mapping"
    for key in ("system", "release", "machine"):
        assert isinstance(os_info.get(key), str), f"{label}: os.{key} must be a string"

    python_info = data["python"]
    assert isinstance(python_info, dict), f"{label}: python must be a mapping"
    assert isinstance(python_info.get("version"), str) and VERSION_RE.match(
        python_info["version"]
    ), f"{label}: python.version has unexpected shape"
    version_info = python_info.get("version_info")
    assert (
        isinstance(version_info, list)
        and len(version_info) == 3
        and all(isinstance(part, int) for part in version_info)
    ), f"{label}: python.version_info must be [major, minor, micro] ints"
    assert isinstance(python_info.get("executable"), str) and python_info["executable"]

    cuda_info = data["cuda"]
    assert isinstance(cuda_info, dict), f"{label}: cuda must be a mapping"
    assert isinstance(cuda_info.get("available"), bool), f"{label}: cuda.available must be bool"
    device_count = cuda_info.get("device_count")
    assert isinstance(device_count, int) and device_count >= 0, (
        f"{label}: cuda.device_count must be a non-negative int"
    )
    devices = cuda_info.get("devices")
    assert isinstance(devices, list), f"{label}: cuda.devices must be a list"
    for device in devices:
        assert isinstance(device, dict), f"{label}: cuda.devices entries must be mappings"
        assert isinstance(device.get("index"), int), f"{label}: device index must be int"
        name = device.get("name")
        assert name is None or isinstance(name, str), f"{label}: device name must be str or null"

    packages = data["packages"]
    assert isinstance(packages, dict), f"{label}: packages must be a mapping"
    for name in REQUIRED_PACKAGES:
        assert name in packages, f"{label}: packages is missing {name!r}"
        _validate_version_field(packages[name], f"{label}: packages.{name}")

    optional = data["optional_dependencies"]
    assert isinstance(optional, dict), f"{label}: optional_dependencies must be a mapping"
    for dep_name, entry in optional.items():
        assert isinstance(entry, dict), f"{label}: optional dep {dep_name!r} must be a mapping"
        assert entry.get("state") in OPTIONAL_DEP_STATES, (
            f"{label}: optional dep {dep_name!r} has invalid state {entry.get('state')!r}"
        )
        assert isinstance(entry.get("note"), str) and entry["note"], (
            f"{label}: optional dep {dep_name!r} must carry a semantic note"
        )

    notes = data["notes"]
    assert isinstance(notes, list) and all(isinstance(note, str) for note in notes), (
        f"{label}: notes must be a list of strings"
    )


def test_snapshot_files_exist():
    for path in SNAPSHOTS:
        assert path.exists(), f"committed environment snapshot is missing: {path}"


@pytest.mark.parametrize("path", SNAPSHOTS, ids=lambda p: p.name)
def test_snapshots_are_deterministic_json(path):
    raw = path.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf"), f"{path.name}: must not carry a UTF-8 BOM"
    text = raw.decode("utf-8")
    assert text.endswith("\n") and not text.endswith("\n\n"), (
        f"{path.name}: must end with exactly one trailing newline"
    )
    assert "\r" not in text, f"{path.name}: must use LF line endings"
    canonical = json.dumps(json.loads(text), indent=2, sort_keys=True) + "\n"
    assert text == canonical, f"{path.name}: must be sorted-keys indent-2 JSON"


@pytest.mark.parametrize("path", SNAPSHOTS, ids=lambda p: p.name)
def test_snapshot_schema(path):
    validate_snapshot(_load_snapshot(path), path.name)


@pytest.mark.parametrize("path", SNAPSHOTS, ids=lambda p: p.name)
def test_snapshots_contain_no_secrets_or_unredacted_user_paths(path):
    text = path.read_text(encoding="utf-8")
    for pattern in SECRET_PATTERNS:
        match = pattern.search(text)
        assert match is None, f"{path.name}: leaked secret/user path matching {pattern.pattern!r}: {match and match.group(0)!r}"


def test_local_snapshot_python_is_310():
    data = _load_snapshot(LOCAL_SNAPSHOT)
    assert data["python"]["version"].startswith("3.10"), (
        "local snapshot must record the maintained Python 3.10 interpreter"
    )


def test_remote_snapshot_records_gpu2_restriction():
    data = _load_snapshot(REMOTE_SNAPSHOT)
    notes = "\n".join(data["notes"])
    assert "GPU2" in notes, "remote snapshot must note the physical GPU2 training restriction"
    assert "8192" in notes, "remote snapshot must note the memory.free > 8192 MiB launch gate"


def test_optional_dependencies_are_semantic_notes_not_install_directives():
    for path in SNAPSHOTS:
        data = _load_snapshot(path)
        sklearn = data["optional_dependencies"].get("scikit-learn")
        assert sklearn is not None, f"{path.name}: scikit-learn state must be recorded"
        assert sklearn["state"] in OPTIONAL_DEP_STATES
        assert "maintained install" in sklearn["note"], (
            f"{path.name}: optional-dependency note must state it is outside the maintained install"
        )


def test_collector_stdout_parses_to_schema():
    assert COLLECTOR.exists(), "environment snapshot collector has not been implemented"
    env = {**os.environ, "PYTHONUTF8": "1"}
    result = subprocess.run(
        [sys.executable, str(COLLECTOR), "--host-alias", "test:collector-self-check"],
        capture_output=True,
        text=True,
        cwd=ROOT,
        env=env,
        timeout=120,
    )
    assert result.returncode == 0, f"collector failed: {result.stderr}"
    data = json.loads(result.stdout)
    validate_snapshot(data, "collector stdout")
    assert data["host_alias"] == "test:collector-self-check"
