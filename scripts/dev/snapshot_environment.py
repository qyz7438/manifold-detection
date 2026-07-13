"""Collect a read-only snapshot of the current execution environment.

The snapshot records interpreter, platform, CUDA availability, GPU identity
(only when CUDA is already available; this never forces CUDA initialization
or runs any workload), and key package versions. It installs nothing,
mutates no conda environment, and only ever queries the GPU index given via
``--gpu-index`` (or all visible indices when omitted, still without running
kernels or allocations).

Output is deterministic JSON: sorted keys, two-space indent, one trailing
newline. ``captured_at_utc`` is an observed value; consumers must validate
its format but never compare it for equality (drift safety).

Secret handling: user-profile path components are redacted
(``C:\\Users\\<name>`` -> ``C:\\Users\\<redacted>``,
``/home/<name>`` -> ``/home/<user>``, ``/Users/<name>`` ->
``/Users/<redacted>``). If a collected value looks like a token, password,
or private key, collection aborts instead of emitting it. Host identity is
conveyed through the ``--host-alias`` option, not through paths.

Usage:
    python scripts/dev/snapshot_environment.py                     # stdout
    python scripts/dev/snapshot_environment.py --output <path>     # write file
    python scripts/dev/snapshot_environment.py --host-alias local:windows-dev
    python scripts/dev/snapshot_environment.py --gpu-index 2
"""
from __future__ import annotations

import argparse
import datetime
import importlib.metadata
import json
import platform
import re
import sys

SCHEMA_VERSION = "1.0"

# Redaction: keep the profile root, drop the user name.
USER_PATH_PATTERNS = (
    (re.compile(r"(?i)([A-Z]:[\\/]+Users[\\/]+)[^\\/]+"), r"\1<redacted>"),
    (re.compile(r"(/home/)[^/]+"), r"\1<user>"),
    (re.compile(r"(?<!:)(/Users/)[^/]+"), r"\1<redacted>"),
)

# Anything matching these in a collected value aborts the snapshot.
SECRET_PATTERNS = (
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"(?i)(api[_-]?key|secret|token|password|passwd)\s*[:=]\s*\S+"),
)

OPTIONAL_DEPENDENCY_NOTE = (
    "optional legacy analysis dependency; not part of the maintained install"
)


def _redact(value: str) -> str:
    for pattern, replacement in USER_PATH_PATTERNS:
        value = pattern.sub(replacement, value)
    return value


def _reject_secrets(data) -> None:
    """Abort if any collected string looks like a secret."""
    stack = [data]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            stack.extend(node.values())
        elif isinstance(node, (list, tuple)):
            stack.extend(node)
        elif isinstance(node, str):
            for pattern in SECRET_PATTERNS:
                if pattern.search(node):
                    raise SystemExit(
                        "snapshot aborted: collected value matches secret pattern "
                        f"{pattern.pattern!r}"
                    )


def _package_version(dist_name: str) -> str:
    try:
        return importlib.metadata.version(dist_name)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def _collect_cuda(gpu_index: int | None) -> dict:
    info: dict = {"available": False, "device_count": 0, "devices": [], "note": None}
    try:
        import torch
    except ImportError:
        info["note"] = "torch is not installed in this interpreter"
        return info
    try:
        available = bool(torch.cuda.is_available())
    except Exception as exc:  # defensive: never let a driver hiccup crash the collector
        info["note"] = f"cuda availability query failed: {exc}"
        return info
    info["available"] = available
    if not available:
        info["note"] = "cuda not available in this interpreter"
        return info
    try:
        device_count = int(torch.cuda.device_count())
    except Exception as exc:
        info["note"] = f"cuda device count query failed: {exc}"
        return info
    info["device_count"] = device_count
    if gpu_index is not None:
        indices = [gpu_index]
    else:
        indices = list(range(device_count))
    for index in indices:
        entry: dict = {"index": index, "name": None}
        if index < 0 or index >= device_count:
            entry["note"] = "requested gpu index is out of range for this host"
        else:
            try:
                # Only queried because CUDA is already available; this creates
                # at most a lightweight context and runs no workload.
                entry["name"] = str(torch.cuda.get_device_name(index))
            except Exception as exc:
                entry["note"] = f"device name query failed: {exc}"
        info["devices"].append(entry)
    return info


def collect_environment(host_alias: str, gpu_index: int | None) -> dict:
    snapshot = {
        "schema_version": SCHEMA_VERSION,
        "captured_at_utc": datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "host_alias": host_alias,
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "python": {
            "version": platform.python_version(),
            "version_info": list(sys.version_info[:3]),
            "executable": _redact(sys.executable),
        },
        "cuda": _collect_cuda(gpu_index),
        "packages": {
            "torch": _package_version("torch"),
            "torchvision": _package_version("torchvision"),
            "numpy": _package_version("numpy"),
            "pillow": _package_version("Pillow"),
            "pytest": _package_version("pytest"),
        },
        "optional_dependencies": {
            "scikit-learn": {
                "state": (
                    "present"
                    if _package_version("scikit-learn") != "unavailable"
                    else "absent"
                ),
                "version": (
                    None
                    if _package_version("scikit-learn") == "unavailable"
                    else _package_version("scikit-learn")
                ),
                "note": OPTIONAL_DEPENDENCY_NOTE,
            }
        },
        "notes": [],
    }
    _reject_secrets(snapshot)
    return snapshot


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--output",
        metavar="PATH",
        default=None,
        help="write the snapshot to PATH (default: print to stdout)",
    )
    parser.add_argument(
        "--host-alias",
        default="local:unknown",
        help="stable alias recorded instead of any host-identifying path",
    )
    parser.add_argument(
        "--gpu-index",
        type=int,
        default=None,
        help="only query this GPU index (default: all visible indices)",
    )
    args = parser.parse_args(argv)

    snapshot = collect_environment(args.host_alias, args.gpu_index)
    text = json.dumps(snapshot, indent=2, sort_keys=True) + "\n"
    if args.output:
        with open(args.output, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
