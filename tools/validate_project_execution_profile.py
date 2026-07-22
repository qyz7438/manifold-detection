#!/usr/bin/env python3
"""Validate a Manifold project execution Profile binding without mutating inputs.

This CLI gates dispatch of a writable execution lane (Kimi or Qwen) for the
Manifold detection project. It checks that a routing manifest
(``profile-routing-v1.json`` shape) and the project workflow SQLite DB agree
on the exact project identity and lane binding:

* selector = canonical absolute Git root + chain_id, never a window name;
* workflow DB and token ledger live inside the project binding;
* manifest and DB agree on role, kind, Profile, session, and worktree data;
* writable lanes are exactly ``kimi-execution`` / ``qwen-execution`` lanes;
* the lane has a non-empty persistent session and an isolated worktree;
* lifecycle flags enforce owned-files fail-closed, dispatch-through-workflow,
  idle zero-token, and resume-only status/log/short-followup behavior;
* Kimi and Qwen sessions and worktrees are distinct.

Both inputs are hashed before and after validation and the DB is opened in
SQLite read-only mode, proving no mutation occurred.

Output is deterministic JSON on stdout containing ``ok``, the normalized
identity, the lane summary, input hashes, and ordered errors.

Exit codes: 0 = valid binding, 1 = contract failure, 2 = usage/input failure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import sqlite3
import sys

EXECUTION_KINDS = ("kimi-execution", "qwen-execution")
REQUIRED_SELECTION_KEY = ["absolute_git_root", "chain_id"]
REQUIRED_TABLES = ("chains", "lanes")

# flag -> required exact value
LIFECYCLE_REQUIREMENTS = (
    ("idle_consumes_tokens", False),
    ("dispatch_requires_workflow_start_or_resume", True),
    ("status_log_and_short_followup_resume_existing_profile", True),
    ("new_agent_for_status_forbidden", True),
    ("cross_project_session_reuse_forbidden", True),
    ("owned_files_fail_closed", True),
)


def normalize_path(value):
    """Normalize a path for case-insensitive, separator-insensitive compare."""
    if not isinstance(value, str) or not value.strip():
        return ""
    return os.path.normcase(os.path.abspath(os.path.normpath(value))).replace(os.sep, "/")


def is_within(root_norm, path_norm):
    """True when normalized path equals or lives under the normalized root."""
    if not root_norm or not path_norm:
        return False
    return path_norm == root_norm or path_norm.startswith(root_norm + "/")


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def open_db_readonly(db_path):
    """Open the workflow DB in SQLite read-only mode with query_only set."""
    uri = pathlib.Path(os.path.abspath(db_path)).as_uri() + "?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    connection.execute("PRAGMA query_only = ON")
    return connection


def build_parser():
    parser = argparse.ArgumentParser(
        prog="validate_project_execution_profile",
        description=(
            "Validate a Manifold project execution Profile binding "
            "(manifest + workflow DB) without mutating either input."
        ),
    )
    parser.add_argument("--manifest", required=True, help="Path to the profile routing JSON manifest.")
    parser.add_argument("--workflow-db", required=True, help="Path to the project workflow SQLite DB.")
    parser.add_argument("--expected-root", required=True, help="Canonical absolute Git root of the project.")
    parser.add_argument("--chain-id", required=True, help="Expected chain ID.")
    parser.add_argument("--role", required=True, help="Lane role to validate (e.g. kimi-builder).")
    return parser


def _lane_rows(lanes):
    """Yield (lane_key, lane_dict) pairs in deterministic order."""
    if not isinstance(lanes, dict):
        return
    for key in sorted(lanes):
        lane = lanes[key]
        if isinstance(lane, dict):
            yield key, lane


def _find_execution_lane(lanes, kind):
    for key, lane in _lane_rows(lanes):
        if lane.get("kind") == kind:
            return key, lane
    return None, None


def validate(manifest, connection, args):
    """Run ordered contract checks. Returns (errors, lane_summary, exec_lanes)."""
    errors = []
    expected_root = normalize_path(args.expected_root)
    workflow_db_cli = normalize_path(args.workflow_db)

    # 1-3. Selector identity: canonical root + chain_id, never window names.
    selector = manifest.get("selector")
    selector = selector if isinstance(selector, dict) else {}
    selector_root_raw = selector.get("absolute_git_root")
    selector_root = normalize_path(selector_root_raw)
    if not selector_root or selector_root != expected_root:
        errors.append(
            "root_mismatch: selector.absolute_git_root=%r does not equal expected root %r"
            % (selector_root_raw, args.expected_root)
        )
    selector_chain = selector.get("chain_id")
    if selector_chain != args.chain_id:
        errors.append(
            "chain_mismatch: selector.chain_id=%r does not equal expected chain %r"
            % (selector_chain, args.chain_id)
        )
    if selector.get("window_name_is_selector") is not False:
        errors.append(
            "window_name_routing: selector.window_name_is_selector must be exactly false"
        )
    if selector.get("selection_key") != REQUIRED_SELECTION_KEY:
        errors.append(
            "window_name_routing: selector.selection_key must be %r, got %r"
            % (REQUIRED_SELECTION_KEY, selector.get("selection_key"))
        )

    # 4-6. Workflow DB and token ledger must stay inside the project binding.
    manifest_db_raw = manifest.get("workflow_db")
    manifest_db = normalize_path(manifest_db_raw)
    if not manifest_db or manifest_db != workflow_db_cli:
        errors.append(
            "workflow_db_mismatch: manifest workflow_db=%r does not equal --workflow-db %r"
            % (manifest_db_raw, args.workflow_db)
        )
    if not is_within(expected_root, manifest_db):
        errors.append(
            "workflow_db_outside_project: workflow_db=%r is outside the project binding %r"
            % (manifest_db_raw, args.expected_root)
        )
    ledger_raw = manifest.get("token_ledger_root")
    ledger = normalize_path(ledger_raw)
    if not ledger or not is_within(expected_root, ledger):
        errors.append(
            "ledger_outside_project: token_ledger_root=%r is outside the project binding %r"
            % (ledger_raw, args.expected_root)
        )

    # 7-8. Execution lanes exist and only execution lanes may be writable.
    lanes = manifest.get("lanes")
    execution_lanes = {}
    for kind in EXECUTION_KINDS:
        key, lane = _find_execution_lane(lanes, kind)
        if lane is None:
            errors.append("execution_lane_absent: no lane with kind %r" % kind)
        else:
            execution_lanes[kind] = {
                "lane_key": key,
                "profile": lane.get("profile"),
                "role": lane.get("role"),
                "session_id": lane.get("session_id"),
                "worktree": normalize_path(lane.get("worktree")),
                "write": lane.get("write"),
            }
    for key, lane in _lane_rows(lanes):
        if lane.get("write") is True and lane.get("kind") not in EXECUTION_KINDS:
            errors.append(
                "writable_non_execution_lane: lane %r has kind %r with write=true"
                % (key, lane.get("kind"))
            )

    # 9-14. Selected lane binding.
    selected_key = None
    selected = None
    for key, lane in _lane_rows(lanes):
        if lane.get("role") == args.role:
            selected_key, selected = key, lane
            break
    lane_summary = None
    db_lane_row = None
    if selected is None:
        errors.append("lane_absent_for_role: no manifest lane has role %r" % args.role)
    else:
        kind = selected.get("kind")
        if kind not in EXECUTION_KINDS:
            errors.append(
                "selected_lane_kind_not_execution: lane %r has kind %r; writable lanes "
                "require kind in %r" % (selected_key, kind, list(EXECUTION_KINDS))
            )
        if selected.get("write") is not True:
            errors.append(
                "lane_not_writable: execution lane %r must have write=true" % selected_key
            )
        session_id = selected.get("session_id")
        if not isinstance(session_id, str) or not session_id.strip():
            errors.append(
                "empty_session: lane %r requires a non-empty persistent session_id" % selected_key
            )
        worktree_raw = selected.get("worktree")
        worktree = normalize_path(worktree_raw)
        if not worktree:
            errors.append(
                "worktree_not_isolated: lane %r has an empty worktree" % selected_key
            )
        elif is_within(expected_root, worktree):
            errors.append(
                "worktree_not_isolated: lane worktree %r equals or lives inside the "
                "canonical root %r" % (worktree_raw, args.expected_root)
            )
        lane_summary = {
            "lane_key": selected_key,
            "profile": selected.get("profile"),
            "role": selected.get("role"),
            "kind": kind,
            "provider": selected.get("provider"),
            "model": selected.get("model"),
            "session_id": session_id,
            "session_state": selected.get("session_state"),
            "worktree": worktree,
            "write": selected.get("write"),
        }

    # 15. Manifest/DB agreement for chain, role, kind, Profile, and session.
    chain_row = connection.execute(
        "SELECT main_profile, main_thread FROM chains WHERE chain_id = ?",
        (args.chain_id,),
    ).fetchone()
    if chain_row is None:
        errors.append("chain_absent_in_db: no chains row for chain_id %r" % args.chain_id)
    db_lane_row = connection.execute(
        "SELECT kind, profile, session_id FROM lanes WHERE chain_id = ? AND role = ?",
        (args.chain_id, args.role),
    ).fetchone()
    if db_lane_row is None:
        errors.append(
            "lane_absent_in_db: no lanes row for chain_id %r and role %r"
            % (args.chain_id, args.role)
        )
    elif selected is not None:
        db_kind, db_profile, db_session = db_lane_row
        if db_kind != selected.get("kind"):
            errors.append(
                "kind_mismatch_db: manifest kind %r != workflow DB kind %r"
                % (selected.get("kind"), db_kind)
            )
        if db_profile != selected.get("profile"):
            errors.append(
                "profile_mismatch_db: manifest profile %r != workflow DB profile %r"
                % (selected.get("profile"), db_profile)
            )
        if db_session != selected.get("session_id"):
            errors.append(
                "session_mismatch_db: manifest session %r != workflow DB session %r"
                % (selected.get("session_id"), db_session)
            )

    # 16. Kimi and Qwen execution sessions and worktrees must be distinct.
    kimi = execution_lanes.get("kimi-execution")
    qwen = execution_lanes.get("qwen-execution")
    if kimi is not None and qwen is not None:
        kimi_session = kimi["session_id"] if isinstance(kimi["session_id"], str) else ""
        qwen_session = qwen["session_id"] if isinstance(qwen["session_id"], str) else ""
        if kimi_session == qwen_session:
            errors.append(
                "shared_session: kimi-execution and qwen-execution share session %r" % kimi_session
            )
        if kimi["worktree"] == qwen["worktree"]:
            errors.append(
                "shared_worktree: kimi-execution and qwen-execution share worktree %r"
                % (kimi["worktree"] or None)
            )

    # 17. Lifecycle flags: exact required values.
    lifecycle = manifest.get("lifecycle")
    lifecycle = lifecycle if isinstance(lifecycle, dict) else {}
    for flag, required in LIFECYCLE_REQUIREMENTS:
        actual = lifecycle.get(flag)
        if actual is not required:
            errors.append(
                "lifecycle_flag_invalid: lifecycle.%s must be exactly %r, got %r"
                % (flag, required, actual)
            )

    if lane_summary is not None:
        lane_summary["db_lane"] = (
            {"kind": db_lane_row[0], "profile": db_lane_row[1], "session_id": db_lane_row[2]}
            if db_lane_row is not None
            else None
        )
    return errors, lane_summary, execution_lanes


def run(args):
    """Execute validation. Returns (exit_code, result_dict)."""
    expected_root = normalize_path(args.expected_root)
    workflow_db_cli = normalize_path(args.workflow_db)
    identity = {
        "expected_root": expected_root,
        "chain_id": args.chain_id,
        "role": args.role,
        "workflow_db": workflow_db_cli,
        "selector_root": None,
        "manifest_workflow_db": None,
        "token_ledger_root": None,
    }
    input_hashes = {
        "manifest": {"path": args.manifest, "sha256_before": None, "sha256_after": None, "unchanged": None},
        "workflow_db": {"path": args.workflow_db, "sha256_before": None, "sha256_after": None, "unchanged": None},
    }

    def result(exit_code, errors, lane=None, execution_lanes=None, schema_version=None):
        return {
            "ok": exit_code == 0,
            "exit_code": exit_code,
            "schema_version": schema_version,
            "identity": identity,
            "lane": lane,
            "execution_lanes": execution_lanes or {},
            "input_hashes": input_hashes,
            "errors": errors,
        }

    # Input phase: manifest must exist, parse, and be a JSON object.
    try:
        manifest_hash_before = sha256_file(args.manifest)
    except OSError as exc:
        return 2, result(2, ["manifest_unreadable: %s" % exc])
    input_hashes["manifest"]["sha256_before"] = manifest_hash_before
    try:
        with open(args.manifest, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return 2, result(2, ["manifest_malformed: %s" % exc])
    if not isinstance(manifest, dict):
        return 2, result(2, ["manifest_malformed: top-level JSON value must be an object"])

    selector = manifest.get("selector")
    if isinstance(selector, dict):
        identity["selector_root"] = normalize_path(selector.get("absolute_git_root"))
    identity["manifest_workflow_db"] = normalize_path(manifest.get("workflow_db"))
    identity["token_ledger_root"] = normalize_path(manifest.get("token_ledger_root"))

    # Input phase: workflow DB must exist and expose the required tables.
    if not os.path.isfile(args.workflow_db):
        return 2, result(2, ["db_unreadable: %r is not a file" % args.workflow_db])
    try:
        db_hash_before = sha256_file(args.workflow_db)
        input_hashes["workflow_db"]["sha256_before"] = db_hash_before
        connection = open_db_readonly(args.workflow_db)
        try:
            tables = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                )
            }
            missing = [name for name in REQUIRED_TABLES if name not in tables]
            if missing:
                return 2, result(2, ["db_malformed: missing required tables %r" % missing])
        finally:
            connection.close()
    except (OSError, sqlite3.Error) as exc:
        return 2, result(2, ["db_malformed: %s" % exc])

    # Contract phase.
    connection = open_db_readonly(args.workflow_db)
    try:
        errors, lane_summary, execution_lanes = validate(manifest, connection, args)
    finally:
        connection.close()

    # Read-only proof: both inputs must be byte-identical after validation.
    manifest_hash_after = sha256_file(args.manifest)
    db_hash_after = sha256_file(args.workflow_db)
    input_hashes["manifest"]["sha256_after"] = manifest_hash_after
    input_hashes["manifest"]["unchanged"] = manifest_hash_after == manifest_hash_before
    input_hashes["workflow_db"]["sha256_after"] = db_hash_after
    input_hashes["workflow_db"]["unchanged"] = db_hash_after == db_hash_before
    for which in ("manifest", "workflow_db"):
        if input_hashes[which]["unchanged"] is not True:
            errors.append("input_mutated: %s SHA256 changed during validation" % which)

    exit_code = 0 if not errors else 1
    return exit_code, result(
        exit_code,
        errors,
        lane=lane_summary,
        execution_lanes=execution_lanes,
        schema_version=manifest.get("schema_version"),
    )


def main(argv=None):
    args = build_parser().parse_args(argv)
    exit_code, payload = run(args)
    sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
