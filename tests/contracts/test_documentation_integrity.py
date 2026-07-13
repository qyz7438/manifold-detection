"""Contract tests for documentation integrity (Task 18).

These tests import the checker from ``scripts/dev/check_docs.py`` and enforce
the checks described in the task plan. Errors (broken links/paths in current
docs, current-status contradictions, generated-doc drift, malformed registries)
must be zero. Warnings (historical-status phrasing in old reports, unregistered
historical version configs) are reported for visibility but are not treated as
CI failures because the historical report blobs are kept byte-unchanged.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
CHECKER = ROOT / "scripts" / "dev" / "check_docs.py"


def _load_checker():
    assert CHECKER.exists(), "documentation integrity checker has not been implemented"
    spec = importlib.util.spec_from_file_location("check_docs", CHECKER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def checker():
    return _load_checker()


@pytest.fixture(scope="module")
def result(checker):
    return checker.run_all(checker.ROOT)


def _errors(findings: list[dict]) -> list[dict]:
    return [f for f in findings if f.get("severity") == "error"]


def test_checker_root_points_at_repository(checker):
    assert checker.ROOT == ROOT
    assert (checker.ROOT / ".gitignore").exists()


def test_cli_human_runs_without_crashing():
    proc = subprocess.run(
        [sys.executable, str(CHECKER), "--root", str(ROOT)],
        capture_output=True,
        text=True,
    )
    assert "Documentation integrity" in proc.stdout or "No documentation integrity" in proc.stdout


def test_cli_json_runs_without_crashing():
    proc = subprocess.run(
        [sys.executable, str(CHECKER), "--root", str(ROOT), "--json"],
        capture_output=True,
        text=True,
    )
    data = json.loads(proc.stdout)
    assert "findings" in data
    assert "summary" in data


def test_all_markdown_docs_are_valid_utf8(checker):
    findings = checker.check_encoding(checker.ROOT)
    assert _errors(findings) == []


def test_relative_markdown_links_resolve(checker):
    findings = checker.check_links(checker.ROOT)
    assert _errors(findings) == []


def test_referenced_repo_paths_exist(checker):
    findings = checker.check_inline_paths(checker.ROOT)
    assert _errors(findings) == []


def test_generated_research_docs_do_not_drift(checker):
    findings = checker.check_generated_drift(checker.ROOT)
    assert _errors(findings) == []


def test_experiment_registry_and_version_configs_are_consistent(checker):
    findings = checker.check_duplicate_experiment_ids(checker.ROOT)
    assert _errors(findings) == []


def test_historical_docs_avoid_current_status_phrases(checker):
    findings = checker.check_historical_status_phrases(checker.ROOT)
    # Historical reports are kept byte-unchanged; flag warnings only.
    assert _errors(findings) == []


def test_current_docs_match_registry_status(checker):
    findings = checker.check_current_status_consistency(checker.ROOT)
    assert _errors(findings) == []


def test_metric_tables_carry_evaluation_scope_labels(checker):
    findings = checker.check_eval_scope_labels(checker.ROOT)
    assert _errors(findings) == []


def test_overall_documentation_integrity(result):
    errors = _errors(result["findings"])
    assert errors == [], f"errors: {errors}"
