"""Contract tests for the CPU CI workflow (Task 19).

These tests inspect ``.github/workflows/ci.yml`` and enforce the design rules
from the refactor plan: the workflow must run the maintained CPU checks, must
not claim to validate research runs, must not use GPUs, and must not download
datasets or touch remote secrets.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"

REQUIRED_COMMANDS = [
    "python scripts/dev/validate_repository_state.py",
    "python scripts/dev/check_import_boundaries.py",
    "python scripts/dev/check_docs.py",
    "python scripts/dev/generate_research_docs.py --check",
    "python -m pytest tests -q",
]

FORBIDDEN_PATTERNS = [
    "CUDA_VISIBLE_DEVICES",
    "nvidia-smi",
    "runs/",  # CI must not depend on local experiment artifacts
    "ssh ",
    "SCP",
    "secrets.",
    "dataset",
    "download",
]


def _workflow_text() -> str:
    assert WORKFLOW.exists(), f"CI workflow not found at {WORKFLOW}"
    return WORKFLOW.read_text(encoding="utf-8")


def _workflow() -> dict:
    return yaml.safe_load(_workflow_text())


def test_ci_workflow_exists():
    assert WORKFLOW.exists()


def test_ci_workflow_is_valid_yaml():
    data = _workflow()
    assert isinstance(data, dict)
    assert "jobs" in data


def test_ci_runs_on_linux_cpu():
    data = _workflow()
    for name, job in data["jobs"].items():
        runs_on = job.get("runs-on", "")
        assert "ubuntu" in str(runs_on).lower(), f"job {name} must run on Ubuntu (CPU)"
        assert "gpu" not in str(runs_on).lower(), f"job {name} must not request a GPU runner"


def test_ci_uses_python_three_ten():
    data = _workflow()
    text = _workflow_text()
    assert "python-version: '3.10'" in text or 'python-version: "3.10"' in text


def test_ci_runs_required_repository_checks():
    data = _workflow()
    text = _workflow_text()
    for command in REQUIRED_COMMANDS:
        assert command in text, f"required command missing from CI: {command}"


def test_ci_forbids_gpu_and_remote_patterns():
    text = _workflow_text().lower()
    for pattern in FORBIDDEN_PATTERNS:
        assert pattern.lower() not in text, f"forbidden pattern in CI workflow: {pattern}"


def test_ci_does_not_promote_research_results():
    text = _workflow_text().lower()
    assert "validated" not in text or "promote" not in text, (
        "CI must not claim to validate or promote research runs"
    )


def test_ci_uses_pip_requirements_not_cuda_wheels():
    text = _workflow_text()
    assert "pip install -r requirements.txt" in text
    assert "cu121" not in text or "torch==2.1.0" in text, (
        "CI must not pin CUDA-only wheel URLs"
    )
