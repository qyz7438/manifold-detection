"""Tests for tools/validate_project_execution_profile.py.

All fixtures are temporary JSON manifests and SQLite workflow DBs built under
pytest tmp_path. No project research modules are imported; the tool is loaded
by file path and exercised both in-process and through the real CLI.
"""

import hashlib
import importlib.util
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL = REPO_ROOT / "tools" / "validate_project_execution_profile.py"

CHAIN = "det.energy.re_roi_closure.001"
KIMI_SESSION = "11111111-2222-3333-4444-555555555555"
QWEN_SESSION = "a8a962e4-0a35-426a-9eef-0bff9e9b00d2"
KIMI_PROFILE = "manifold-kimi-executor"
QWEN_PROFILE = "manifold-qwen-executor"

spec = importlib.util.spec_from_file_location("validate_project_execution_profile", TOOL)
validator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validator)


# --------------------------------------------------------------------------
# Fixture builders
# --------------------------------------------------------------------------

def build_db(path, chain_id=CHAIN, lanes=None, include_chain=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute(
        "CREATE TABLE chains ("
        " chain_id TEXT PRIMARY KEY, main_profile TEXT NOT NULL,"
        " main_thread TEXT NOT NULL, phase TEXT NOT NULL, next_decision TEXT,"
        " created_at TEXT NOT NULL, updated_at TEXT NOT NULL)"
    )
    connection.execute(
        "CREATE TABLE lanes ("
        " chain_id TEXT NOT NULL, role TEXT NOT NULL, kind TEXT NOT NULL,"
        " profile TEXT NOT NULL, session_id TEXT NOT NULL,"
        " created_at TEXT NOT NULL, updated_at TEXT NOT NULL,"
        " PRIMARY KEY (chain_id, role))"
    )
    if include_chain:
        connection.execute(
            "INSERT INTO chains VALUES (?, 'manifold-research-main', 'thread',"
            " 'resume_or_init', NULL, 'now', 'now')",
            (chain_id,),
        )
    if lanes is None:
        lanes = [
            ("kimi-builder", "kimi-execution", KIMI_PROFILE, KIMI_SESSION),
            ("qwen-builder", "qwen-execution", QWEN_PROFILE, QWEN_SESSION),
        ]
    for role, kind, profile, session in lanes:
        connection.execute(
            "INSERT INTO lanes VALUES (?, ?, ?, ?, ?, 'now', 'now')",
            (chain_id, role, kind, profile, session),
        )
    connection.commit()
    connection.close()


def build_manifest(root, db_path):
    return {
        "schema_version": 1,
        "selector": {
            "absolute_git_root": str(root),
            "chain_id": CHAIN,
            "selection_key": ["absolute_git_root", "chain_id"],
            "window_name_is_selector": False,
        },
        "workflow_db": str(db_path),
        "token_ledger_root": str(root / "orchestrator" / "token-ledger"),
        "main": {"profile": "manifold-research-main", "thread_id": "thread"},
        "lanes": {
            "protocol_review": {"profile": "manifold-protocol-review", "write": False},
            "kimi_execution": {
                "profile": KIMI_PROFILE,
                "role": "kimi-builder",
                "kind": "kimi-execution",
                "provider": "kimi",
                "model": "kimi-code/k3",
                "session_id": KIMI_SESSION,
                "session_state": "active",
                "worktree": str(root.parent / "kimi-worktree"),
                "write": True,
            },
            "qwen_execution": {
                "profile": QWEN_PROFILE,
                "role": "qwen-builder",
                "kind": "qwen-execution",
                "provider": "aliyun-maas-via-local-proxy",
                "model": "qwen3.8-max-preview",
                "session_id": QWEN_SESSION,
                "session_state": "queued",
                "worktree": str(root.parent / "qwen-worktree"),
                "write": True,
            },
        },
        "lifecycle": {
            "idle_consumes_tokens": False,
            "dispatch_requires_workflow_start_or_resume": True,
            "status_log_and_short_followup_resume_existing_profile": True,
            "new_agent_for_status_forbidden": True,
            "cross_project_session_reuse_forbidden": True,
            "owned_files_fail_closed": True,
        },
    }


class Binding:
    def __init__(self, tmp_path, manifest=None, db_kwargs=None):
        self.root = tmp_path / "canonical"
        self.root.mkdir(parents=True, exist_ok=True)
        self.db = self.root / "orchestrator" / "workflow.sqlite3"
        build_db(self.db, **(db_kwargs or {}))
        self.manifest = manifest if manifest is not None else build_manifest(self.root, self.db)
        self.manifest_path = tmp_path / "manifest.json"
        self.manifest_path.write_text(json.dumps(self.manifest, indent=2), encoding="utf-8")


def run_cli(binding, role="kimi-builder", chain=CHAIN, root=None, extra_args=None):
    cmd = [
        sys.executable,
        str(TOOL),
        "--manifest", str(binding.manifest_path),
        "--workflow-db", str(binding.db),
        "--expected-root", str(root if root is not None else binding.root),
        "--chain-id", chain,
        "--role", role,
    ]
    if extra_args:
        cmd.extend(extra_args)
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    payload = json.loads(proc.stdout) if proc.stdout.strip() else None
    return proc.returncode, payload


def error_codes(payload):
    return [entry.split(":")[0] for entry in payload["errors"]]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# --------------------------------------------------------------------------
# Valid lanes
# --------------------------------------------------------------------------

def test_valid_kimi_lane(tmp_path):
    binding = Binding(tmp_path)
    code, payload = run_cli(binding, role="kimi-builder")
    assert code == 0
    assert payload["ok"] is True
    assert payload["errors"] == []
    assert payload["exit_code"] == 0
    assert payload["lane"]["kind"] == "kimi-execution"
    assert payload["lane"]["profile"] == KIMI_PROFILE
    assert payload["lane"]["session_id"] == KIMI_SESSION
    assert payload["lane"]["write"] is True
    assert payload["lane"]["db_lane"]["session_id"] == KIMI_SESSION
    assert payload["identity"]["chain_id"] == CHAIN
    for which in ("manifest", "workflow_db"):
        assert payload["input_hashes"][which]["unchanged"] is True


def test_valid_qwen_lane(tmp_path):
    binding = Binding(tmp_path)
    code, payload = run_cli(binding, role="qwen-builder")
    assert code == 0
    assert payload["ok"] is True
    assert payload["errors"] == []
    assert payload["lane"]["kind"] == "qwen-execution"
    assert payload["lane"]["session_id"] == QWEN_SESSION


# --------------------------------------------------------------------------
# Selector identity
# --------------------------------------------------------------------------

def test_wrong_root_rejected(tmp_path):
    binding = Binding(tmp_path)
    binding.manifest["selector"]["absolute_git_root"] = str(tmp_path / "other-root")
    binding.manifest_path.write_text(json.dumps(binding.manifest), encoding="utf-8")
    code, payload = run_cli(binding)
    assert code == 1
    assert "root_mismatch" in error_codes(payload)
    assert payload["ok"] is False


def test_wrong_chain_rejected(tmp_path):
    binding = Binding(tmp_path)
    code, payload = run_cli(binding, chain="det.energy.other_chain.999")
    assert code == 1
    assert "chain_mismatch" in error_codes(payload)


def test_window_name_selector_rejected(tmp_path):
    binding = Binding(tmp_path)
    binding.manifest["selector"]["window_name_is_selector"] = True
    binding.manifest_path.write_text(json.dumps(binding.manifest), encoding="utf-8")
    code, payload = run_cli(binding)
    assert code == 1
    assert "window_name_routing" in error_codes(payload)


def test_window_name_selection_key_rejected(tmp_path):
    binding = Binding(tmp_path)
    binding.manifest["selector"]["selection_key"] = ["window_name"]
    binding.manifest_path.write_text(json.dumps(binding.manifest), encoding="utf-8")
    code, payload = run_cli(binding)
    assert code == 1
    assert "window_name_routing" in error_codes(payload)


# --------------------------------------------------------------------------
# Project binding containment
# --------------------------------------------------------------------------

def test_workflow_db_outside_project_rejected(tmp_path):
    outside_db = tmp_path / "outside" / "workflow.sqlite3"
    build_db(outside_db)
    binding = Binding(tmp_path)
    binding.db = outside_db
    binding.manifest["workflow_db"] = str(outside_db)
    binding.manifest_path.write_text(json.dumps(binding.manifest), encoding="utf-8")
    code, payload = run_cli(binding)
    assert code == 1
    assert "workflow_db_outside_project" in error_codes(payload)
    assert "workflow_db_mismatch" not in error_codes(payload)


def test_ledger_outside_project_rejected(tmp_path):
    binding = Binding(tmp_path)
    binding.manifest["token_ledger_root"] = str(tmp_path / "outside-ledger")
    binding.manifest_path.write_text(json.dumps(binding.manifest), encoding="utf-8")
    code, payload = run_cli(binding)
    assert code == 1
    assert "ledger_outside_project" in error_codes(payload)


def test_workflow_db_mismatch_rejected(tmp_path):
    binding = Binding(tmp_path)
    binding.manifest["workflow_db"] = str(binding.root / "orchestrator" / "other.sqlite3")
    binding.manifest_path.write_text(json.dumps(binding.manifest), encoding="utf-8")
    code, payload = run_cli(binding)
    assert code == 1
    assert "workflow_db_mismatch" in error_codes(payload)


# --------------------------------------------------------------------------
# Lane presence and agreement
# --------------------------------------------------------------------------

def test_absent_lane_rejected(tmp_path):
    manifest = build_manifest(tmp_path / "canonical", tmp_path / "canonical" / "orchestrator" / "workflow.sqlite3")
    del manifest["lanes"]["kimi_execution"]
    binding = Binding(tmp_path, manifest=manifest)
    code, payload = run_cli(binding, role="kimi-builder")
    assert code == 1
    codes = error_codes(payload)
    assert "lane_absent_for_role" in codes
    assert "execution_lane_absent" in codes


def test_db_lane_absent_rejected(tmp_path):
    binding = Binding(tmp_path, db_kwargs={"lanes": [("qwen-builder", "qwen-execution", QWEN_PROFILE, QWEN_SESSION)]})
    code, payload = run_cli(binding, role="kimi-builder")
    assert code == 1
    assert "lane_absent_in_db" in error_codes(payload)


def test_chain_absent_in_db_rejected(tmp_path):
    binding = Binding(tmp_path, db_kwargs={"chain_id": "det.energy.other.001"})
    code, payload = run_cli(binding)
    assert code == 1
    assert "chain_absent_in_db" in error_codes(payload)


def test_kind_mismatch_rejected(tmp_path):
    lanes = [
        ("kimi-builder", "qwen-execution", KIMI_PROFILE, KIMI_SESSION),
        ("qwen-builder", "qwen-execution", QWEN_PROFILE, QWEN_SESSION),
    ]
    binding = Binding(tmp_path, db_kwargs={"lanes": lanes})
    code, payload = run_cli(binding, role="kimi-builder")
    assert code == 1
    assert "kind_mismatch_db" in error_codes(payload)


def test_profile_mismatch_rejected(tmp_path):
    lanes = [
        ("kimi-builder", "kimi-execution", "manifold-other-profile", KIMI_SESSION),
        ("qwen-builder", "qwen-execution", QWEN_PROFILE, QWEN_SESSION),
    ]
    binding = Binding(tmp_path, db_kwargs={"lanes": lanes})
    code, payload = run_cli(binding, role="kimi-builder")
    assert code == 1
    assert "profile_mismatch_db" in error_codes(payload)


def test_session_mismatch_rejected(tmp_path):
    lanes = [
        ("kimi-builder", "kimi-execution", KIMI_PROFILE, "99999999-0000-0000-0000-000000000000"),
        ("qwen-builder", "qwen-execution", QWEN_PROFILE, QWEN_SESSION),
    ]
    binding = Binding(tmp_path, db_kwargs={"lanes": lanes})
    code, payload = run_cli(binding, role="kimi-builder")
    assert code == 1
    assert "session_mismatch_db" in error_codes(payload)


def test_empty_session_rejected(tmp_path):
    lanes = [
        ("kimi-builder", "kimi-execution", KIMI_PROFILE, ""),
        ("qwen-builder", "qwen-execution", QWEN_PROFILE, QWEN_SESSION),
    ]
    binding = Binding(tmp_path, db_kwargs={"lanes": lanes})
    binding.manifest["lanes"]["kimi_execution"]["session_id"] = ""
    binding.manifest_path.write_text(json.dumps(binding.manifest), encoding="utf-8")
    code, payload = run_cli(binding, role="kimi-builder")
    assert code == 1
    assert "empty_session" in error_codes(payload)


def test_null_session_rejected(tmp_path):
    binding = Binding(tmp_path)
    binding.manifest["lanes"]["kimi_execution"]["session_id"] = None
    binding.manifest_path.write_text(json.dumps(binding.manifest), encoding="utf-8")
    code, payload = run_cli(binding, role="kimi-builder")
    assert code == 1
    assert "empty_session" in error_codes(payload)


# --------------------------------------------------------------------------
# Kimi/Qwen distinctness and worktree isolation
# --------------------------------------------------------------------------

def test_shared_session_rejected(tmp_path):
    binding = Binding(tmp_path)
    binding.manifest["lanes"]["qwen_execution"]["session_id"] = KIMI_SESSION
    binding.manifest_path.write_text(json.dumps(binding.manifest), encoding="utf-8")
    code, payload = run_cli(binding, role="kimi-builder")
    assert code == 1
    assert "shared_session" in error_codes(payload)


def test_shared_worktree_rejected(tmp_path):
    binding = Binding(tmp_path)
    binding.manifest["lanes"]["qwen_execution"]["worktree"] = binding.manifest["lanes"]["kimi_execution"]["worktree"]
    binding.manifest_path.write_text(json.dumps(binding.manifest), encoding="utf-8")
    code, payload = run_cli(binding, role="kimi-builder")
    assert code == 1
    assert "shared_worktree" in error_codes(payload)


def test_worktree_inside_root_rejected(tmp_path):
    binding = Binding(tmp_path)
    binding.manifest["lanes"]["kimi_execution"]["worktree"] = str(binding.root / "inside")
    binding.manifest_path.write_text(json.dumps(binding.manifest), encoding="utf-8")
    code, payload = run_cli(binding, role="kimi-builder")
    assert code == 1
    assert "worktree_not_isolated" in error_codes(payload)


def test_worktree_equal_root_rejected(tmp_path):
    binding = Binding(tmp_path)
    binding.manifest["lanes"]["kimi_execution"]["worktree"] = str(binding.root)
    binding.manifest_path.write_text(json.dumps(binding.manifest), encoding="utf-8")
    code, payload = run_cli(binding, role="kimi-builder")
    assert code == 1
    assert "worktree_not_isolated" in error_codes(payload)


# --------------------------------------------------------------------------
# Lifecycle flags and writable review lanes
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "flag,value",
    [
        ("idle_consumes_tokens", True),
        ("dispatch_requires_workflow_start_or_resume", False),
        ("status_log_and_short_followup_resume_existing_profile", False),
        ("new_agent_for_status_forbidden", False),
        ("cross_project_session_reuse_forbidden", False),
        ("owned_files_fail_closed", False),
        ("owned_files_fail_closed", "missing"),
    ],
)
def test_lifecycle_flags_rejected(tmp_path, flag, value):
    binding = Binding(tmp_path)
    if value == "missing":
        del binding.manifest["lifecycle"][flag]
    else:
        binding.manifest["lifecycle"][flag] = value
    binding.manifest_path.write_text(json.dumps(binding.manifest), encoding="utf-8")
    code, payload = run_cli(binding)
    assert code == 1
    assert "lifecycle_flag_invalid" in error_codes(payload)


def test_writable_review_lane_rejected(tmp_path):
    binding = Binding(tmp_path)
    binding.manifest["lanes"]["protocol_review"]["write"] = True
    binding.manifest_path.write_text(json.dumps(binding.manifest), encoding="utf-8")
    code, payload = run_cli(binding, role="kimi-builder")
    assert code == 1
    assert "writable_non_execution_lane" in error_codes(payload)


def test_review_role_selected_rejected(tmp_path):
    manifest = build_manifest(tmp_path / "canonical", tmp_path / "canonical" / "orchestrator" / "workflow.sqlite3")
    manifest["lanes"]["protocol_review"] = {
        "profile": "manifold-protocol-review",
        "role": "protocol-reviewer",
        "kind": "protocol-review",
        "session_id": "rev-session",
        "worktree": str(tmp_path / "review-worktree"),
        "write": True,
    }
    binding = Binding(tmp_path, manifest=manifest)
    code, payload = run_cli(binding, role="protocol-reviewer")
    assert code == 1
    assert "selected_lane_kind_not_execution" in error_codes(payload)


def test_lane_not_writable_rejected(tmp_path):
    binding = Binding(tmp_path)
    binding.manifest["lanes"]["kimi_execution"]["write"] = False
    binding.manifest_path.write_text(json.dumps(binding.manifest), encoding="utf-8")
    code, payload = run_cli(binding, role="kimi-builder")
    assert code == 1
    assert "lane_not_writable" in error_codes(payload)


# --------------------------------------------------------------------------
# Usage / input failures (exit 2)
# --------------------------------------------------------------------------

def test_malformed_manifest_exit2(tmp_path):
    binding = Binding(tmp_path)
    binding.manifest_path.write_text("{not valid json", encoding="utf-8")
    code, payload = run_cli(binding)
    assert code == 2
    assert "manifest_malformed" in error_codes(payload)
    assert payload["ok"] is False


def test_unreadable_manifest_exit2(tmp_path):
    binding = Binding(tmp_path)
    binding.manifest_path = tmp_path / "does-not-exist.json"
    code, payload = run_cli(binding)
    assert code == 2
    assert "manifest_unreadable" in error_codes(payload)


def test_manifest_not_object_exit2(tmp_path):
    binding = Binding(tmp_path)
    binding.manifest_path.write_text("[1, 2, 3]", encoding="utf-8")
    code, payload = run_cli(binding)
    assert code == 2
    assert "manifest_malformed" in error_codes(payload)


def test_malformed_db_exit2(tmp_path):
    binding = Binding(tmp_path)
    binding.db.write_bytes(b"this is not a sqlite database")
    code, payload = run_cli(binding)
    assert code == 2
    assert "db_malformed" in error_codes(payload)


def test_missing_db_exit2(tmp_path):
    binding = Binding(tmp_path)
    binding.db.unlink()
    code, payload = run_cli(binding)
    assert code == 2
    assert "db_unreadable" in error_codes(payload)


def test_db_missing_tables_exit2(tmp_path):
    binding = Binding(tmp_path)
    connection = sqlite3.connect(binding.db)
    connection.execute("DROP TABLE lanes")
    connection.commit()
    connection.close()
    code, payload = run_cli(binding)
    assert code == 2
    assert "db_malformed" in error_codes(payload)


def test_missing_cli_argument_exit2(tmp_path):
    proc = subprocess.run(
        [sys.executable, str(TOOL), "--manifest", "x.json"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert proc.returncode == 2


# --------------------------------------------------------------------------
# Determinism and read-only proof
# --------------------------------------------------------------------------

def test_deterministic_output(tmp_path):
    binding = Binding(tmp_path)
    first = subprocess.run(
        [sys.executable, str(TOOL), "--manifest", str(binding.manifest_path),
         "--workflow-db", str(binding.db), "--expected-root", str(binding.root),
         "--chain-id", CHAIN, "--role", "kimi-builder"],
        capture_output=True,
    )
    second = subprocess.run(
        [sys.executable, str(TOOL), "--manifest", str(binding.manifest_path),
         "--workflow-db", str(binding.db), "--expected-root", str(binding.root),
         "--chain-id", CHAIN, "--role", "kimi-builder"],
        capture_output=True,
    )
    assert first.returncode == second.returncode == 0
    assert first.stdout == second.stdout
    parsed = json.loads(first.stdout)
    assert list(parsed.keys()) == sorted(parsed.keys())


def test_deterministic_failure_output(tmp_path):
    binding = Binding(tmp_path)
    binding.manifest["selector"]["absolute_git_root"] = str(tmp_path / "other")
    binding.manifest_path.write_text(json.dumps(binding.manifest), encoding="utf-8")
    outputs = []
    for _ in range(2):
        proc = subprocess.run(
            [sys.executable, str(TOOL), "--manifest", str(binding.manifest_path),
             "--workflow-db", str(binding.db), "--expected-root", str(binding.root),
             "--chain-id", CHAIN, "--role", "kimi-builder"],
            capture_output=True,
        )
        outputs.append(proc.stdout)
        assert proc.returncode == 1
    assert outputs[0] == outputs[1]


def test_input_sha256_preserved(tmp_path):
    binding = Binding(tmp_path)
    manifest_before = sha256(binding.manifest_path)
    db_before = sha256(binding.db)
    code, payload = run_cli(binding, role="qwen-builder")
    assert code == 0
    assert sha256(binding.manifest_path) == manifest_before
    assert sha256(binding.db) == db_before
    hashes = payload["input_hashes"]
    assert hashes["manifest"]["sha256_before"] == manifest_before
    assert hashes["manifest"]["sha256_after"] == manifest_before
    assert hashes["workflow_db"]["sha256_before"] == db_before
    assert hashes["workflow_db"]["sha256_after"] == db_before
    assert hashes["manifest"]["unchanged"] is True
    assert hashes["workflow_db"]["unchanged"] is True


def test_input_sha256_preserved_on_failure(tmp_path):
    binding = Binding(tmp_path)
    binding.manifest["lanes"]["kimi_execution"]["session_id"] = ""
    binding.manifest_path.write_text(json.dumps(binding.manifest), encoding="utf-8")
    manifest_before = sha256(binding.manifest_path)
    db_before = sha256(binding.db)
    code, payload = run_cli(binding, role="kimi-builder")
    assert code == 1
    assert sha256(binding.manifest_path) == manifest_before
    assert sha256(binding.db) == db_before
    assert payload["input_hashes"]["manifest"]["unchanged"] is True
    assert payload["input_hashes"]["workflow_db"]["unchanged"] is True


# --------------------------------------------------------------------------
# In-process unit checks for path handling
# --------------------------------------------------------------------------

def test_normalize_path_case_and_separators():
    forward = validator.normalize_path("E:/CLIproject/WorkTree")
    backward = validator.normalize_path("e:\\cliproject\\worktree")
    assert forward == backward
    assert "/" in forward


def test_is_within():
    root = validator.normalize_path("E:/CLIproject/root")
    assert validator.is_within(root, validator.normalize_path("E:/CLIproject/root/db.sqlite3"))
    assert validator.is_within(root, root)
    assert not validator.is_within(root, validator.normalize_path("E:/CLIproject/rooted/x"))
    assert not validator.is_within(root, validator.normalize_path("E:/CLIproject/other"))
    assert not validator.is_within(root, "")
    assert not validator.is_within("", root)


def test_main_in_process_valid(tmp_path, capsys):
    binding = Binding(tmp_path)
    exit_code = validator.main(
        [
            "--manifest", str(binding.manifest_path),
            "--workflow-db", str(binding.db),
            "--expected-root", str(binding.root),
            "--chain-id", CHAIN,
            "--role", "kimi-builder",
        ]
    )
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
