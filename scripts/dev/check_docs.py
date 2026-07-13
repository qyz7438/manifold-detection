"""Documentation integrity checker for the research-state refactor (Task 18).

Checks:

* UTF-8 decodability of every tracked markdown document.
* Relative markdown links resolve to an existing file or directory.
* Inline repo-relative paths referenced in docs exist (with explicit
  allowlist for local/generated artifacts).
* Generated research docs are up to date with their registries.
* Experiment IDs are unique across the experiment registry and version
  configs; version configs are registered.
* Historical/archived docs do not use present-tense current-status phrases.
* Generated/current metric tables carry explicit evaluation-scope labels.
* Current docs do not independently declare active/authorized/in-progress
  research status that contradicts the registry.

CLI:

    python scripts/dev/check_docs.py          # human-readable findings
    python scripts/dev/check_docs.py --json   # machine-readable findings

Exit code 0 when no findings are reported, 1 otherwise.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Docs that are allowed to speak about current research state. Generated
# research docs are derived from the registry and are verified by the drift
# check, so they are excluded from the "independent status declaration" check.
CURRENT_DOCS: set[str] = {
    "README.md",
    "AGENTS.md",
    "docs/architecture.md",
    "docs/reports/index.md",
    "docs/research/refactor_ledger.md",
    "docs/research/environment.md",
    "docs/versioning_scheme.md",
    "docs/refactor_directory_versioning_plan.md",
}

# Hand-maintained current docs that may declare research status. The refactor
# ledger uses the word "authorized" in a task-ownership sense, not a research
# status sense, so it is excluded from the status-consistency check.
CURRENT_STATUS_DOCS: set[str] = CURRENT_DOCS - {"docs/research/refactor_ledger.md"}

GENERATED_RESEARCH_DOCS: set[str] = {
    "docs/research/README.md",
    "docs/research/current_status.md",
    "docs/research/experiment_index.md",
    "docs/research/artifact_index.md",
}

# Present-tense current-status phrases that historical/archived docs must not
# use. Current docs may use them only when they are backed by the registry.
# Present-tense current-status phrases that historical/archived docs must not
# use. "authorized" by itself is excluded because historical preregistration
# docs use it in a protocol-approval sense; the current-status check below
# still catches independent declarations of authorization in current docs.
HISTORICAL_STATUS_PHRASES: list[str] = [
    r"currently\s+active",
    r"current\s+active",
    r"\bis\s+active\b",
    r"\bactive\s+line\b",
    r"\bactive\s+claim\b",
    r"\bactive\s+research\b",
    r"\bactive\s+family\b",
    r"\bactive\s+path\b",
    r"\bactive\s+question\b",
    r"\bauthorized\s+experiment\b",
    r"\bin\s+progress\b",
    r"\bongoing\b",
    r"\bcurrently\s+authorized\b",
    r"\bcurrently\s+queued\b",
]

# Docs that use task-planning language ("in progress", "authorized") and are
# not historical reports. They are excluded from the historical-phrase scan.
HISTORICAL_STATUS_EXCLUDED_DOCS: set[str] = {
    "docs/superpowers/plans/2026-07-13-manifold-repository-research-state-refactor.md",
}

# Local/generated path prefixes that are allowed even when the filesystem
# target does not exist.
ALLOWED_MISSING_PATH_PREFIXES: tuple[str, ...] = (
    "runs/",
    "runs\\",
    "data/",
    "data\\",
    "output/",
    "output\\",
    "http://",
    "https://",
    "mailto:",
)

# Top-level repo directories that an inline code repo path might reference.
KNOWN_TOP_LEVEL_DIRS: tuple[str, ...] = (
    "scripts",
    "spectral_detection_posttrain",
    "docs",
    "legacy",
    "tests",
    "configs",
    "mfvpt",
    "assets",
    "obsidian",
)

# File extensions that make an inline path look like a repo file reference.
KNOWN_FILE_EXTENSIONS: tuple[str, ...] = (
    ".py",
    ".json",
    ".md",
    ".sh",
    ".bat",
    ".txt",
    ".yml",
    ".yaml",
    ".toml",
)

# Tokens that identify a markdown table row as containing metrics.
METRIC_TOKENS: tuple[str, ...] = (
    "ap50",
    "ap75",
    "map",
    "ece",
    "precision",
    "recall",
    "fpr",
    "false_positive_rate",
    "predictions",
    "num_predictions",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _iter_docs(root: Path):
    for path in root.rglob("*.md"):
        if ".git" in path.parts:
            continue
        yield path


def _is_allowed_missing(target: str) -> bool:
    low = target.lower()
    if low.startswith(ALLOWED_MISSING_PATH_PREFIXES):
        return True
    if low.startswith("output/") and low.endswith(".json"):
        return True
    return False


def _line_at(text: str, offset: int) -> int:
    return text[:offset].count("\n") + 1


def _load_module(root: Path, repo_path: str):
    """Load a repo module by filesystem path without importing the package."""
    path = root / repo_path
    spec = importlib.util.spec_from_file_location(
        repo_path.replace("/", ".").replace("\\", "."), path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Markdown link resolution
# ---------------------------------------------------------------------------


def _markdown_links(text: str):
    """Yield (offset, target) for inline and reference-style link targets."""
    for m in re.finditer(r"\[([^\]]*)\]\(([^)]+)\)", text):
        yield m.start(), m.group(2)
    for m in re.finditer(r"^\[([^\]]+)\]:\s*(\S+)", text, re.MULTILINE):
        yield m.start(), m.group(2)


def _resolve_link(doc_dir: Path, target: str) -> Path | None:
    """Return resolved filesystem path or None if the target is external/allowed."""
    # Strip fragment and query; decode minimal URL escapes.
    target = target.split("#")[0].split("?")[0]
    if not target:
        return None
    target = target.replace("%20", " ")
    # External URL or allowed local/generated artifact.
    if re.match(r"^[a-z][a-z0-9+.-]*:", target, re.IGNORECASE):
        return None
    if _is_allowed_missing(target):
        return None
    return (doc_dir / target).resolve()


# ---------------------------------------------------------------------------
# Inline path extraction
# ---------------------------------------------------------------------------


def _inline_code_paths(text: str):
    """Yield (offset, path) for backtick-quoted repo-relative paths."""
    for m in re.finditer(r"`([^`\n]+)`", text):
        content = m.group(1).strip()
        if not content:
            continue
        if " " in content:
            continue
        if content.startswith(("http://", "https://", "mailto:")):
            continue
        if _is_allowed_missing(content):
            continue
        if "/" not in content and "\\" not in content:
            continue
        # Skip shell-style globs, brace expansion, and ellipsis placeholders.
        if any(ch in content for ch in "*?{}[]<>" ) or "..." in content:
            continue
        first = content.replace("\\", "/").split("/")[0]
        if first not in KNOWN_TOP_LEVEL_DIRS:
            continue
        # Strip trailing line/column references (e.g., path:137 or path::test_name).
        content = re.sub(r":\d+(?::\d+)?$", "", content)
        content = re.sub(r"::\w+$", "", content)
        if not content:
            continue
        yield m.start(), content


# ---------------------------------------------------------------------------
# Registry helpers
# ---------------------------------------------------------------------------


def _registry_dir(root: Path) -> Path:
    return root / "spectral_detection_posttrain" / "configs" / "registry"


def _load_research_lines(root: Path) -> list[dict]:
    return _load_json(_registry_dir(root) / "research_lines.json")["lines"]


def _load_experiment_records(root: Path) -> list[dict]:
    return _load_json(_registry_dir(root) / "experiments.json")["records"]


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def check_encoding(root: Path) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for path in sorted(_iter_docs(root)):
        try:
            path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            findings.append(
                {
                    "category": "encoding",
                    "severity": "error",
                    "path": _relative(root, path),
                    "line": None,
                    "message": f"not valid UTF-8: {exc}",
                }
            )
    return findings


def _doc_severity(rel: str) -> str:
    """Historical/archived docs get warnings; current/generated docs get errors."""
    if rel in CURRENT_DOCS or rel in GENERATED_RESEARCH_DOCS:
        return "error"
    return "warning"


def check_links(root: Path) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for path in sorted(_iter_docs(root)):
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        rel = _relative(root, path)
        doc_dir = path.parent
        for offset, target in _markdown_links(text):
            resolved = _resolve_link(doc_dir, target)
            if resolved is None:
                continue
            if not resolved.exists():
                try:
                    resolved_rel = resolved.relative_to(root.resolve()).as_posix()
                except ValueError:
                    resolved_rel = str(resolved)
                findings.append(
                    {
                        "category": "broken_link",
                        "severity": _doc_severity(rel),
                        "path": rel,
                        "line": _line_at(text, offset),
                        "message": f"broken link target {target!r} (resolved {resolved_rel!r})",
                    }
                )
    return findings


def check_inline_paths(root: Path) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for path in sorted(_iter_docs(root)):
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        rel = _relative(root, path)
        doc_dir = path.parent
        for offset, content in _inline_code_paths(text):
            candidates = [root / content, doc_dir / content]
            if any(p.exists() for p in candidates):
                continue
            findings.append(
                {
                    "category": "broken_path",
                    "severity": _doc_severity(rel),
                    "path": rel,
                    "line": _line_at(text, offset),
                    "message": f"referenced repo path not found: `{content}`",
                }
            )
    return findings


def check_generated_drift(root: Path) -> list[dict[str, Any]]:
    generator = _load_module(root, "scripts/dev/generate_research_docs.py")
    ok, message = generator.check_drift(root=root)
    if ok:
        return []
    return [
        {
            "category": "generated_drift",
            "severity": "error",
            "path": "docs/research/",
            "line": None,
            "message": message,
        }
    ]


def check_duplicate_experiment_ids(root: Path) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    records = _load_experiment_records(root)

    # Duplicate IDs inside the experiment registry.
    seen_record_ids: dict[str, dict] = {}
    for record in records:
        rid = record["id"]
        if rid in seen_record_ids:
            findings.append(
                {
                    "category": "duplicate_experiment_id",
                    "severity": "error",
                    "path": "spectral_detection_posttrain/configs/registry/experiments.json",
                    "line": None,
                    "message": f"duplicate experiment id {rid!r}",
                }
            )
        seen_record_ids[rid] = record

    # Duplicate IDs inside version configs and registration checks.
    versions_dir = root / "spectral_detection_posttrain" / "configs" / "versions"
    config_ids: dict[str, Path] = {}
    for cfg_path in sorted(versions_dir.glob("*.json")):
        try:
            payload = _load_json(cfg_path)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            findings.append(
                {
                    "category": "malformed_version_config",
                    "severity": "error",
                    "path": _relative(root, cfg_path),
                    "line": None,
                    "message": f"cannot parse version config: {exc}",
                }
            )
            continue
        vid = payload.get("version_id")
        if not vid:
            findings.append(
                {
                    "category": "missing_version_id",
                    "severity": "error",
                    "path": _relative(root, cfg_path),
                    "line": None,
                    "message": "version config has no 'version_id' field",
                }
            )
            continue
        if vid in config_ids:
            findings.append(
                {
                    "category": "duplicate_experiment_id",
                    "severity": "error",
                    "path": _relative(root, cfg_path),
                    "line": None,
                    "message": f"version id {vid!r} also appears in {_relative(root, config_ids[vid])}",
                }
            )
        config_ids[vid] = cfg_path

    # Version configs should be referenced by the registry unless they are
    # explicitly historical artifacts.
    record_ids = set(seen_record_ids)
    for vid, cfg_path in config_ids.items():
        if vid not in record_ids:
            findings.append(
                {
                    "category": "unregistered_version_config",
                    "severity": "warning",
                    "path": _relative(root, cfg_path),
                    "line": None,
                    "message": f"version config {vid!r} is not referenced in experiments.json",
                }
            )

    # Record config_path existence and version_id consistency.
    for record in records:
        config_path = record.get("config_path")
        if not config_path:
            continue
        cfg = root / config_path
        if not cfg.exists():
            findings.append(
                {
                    "category": "missing_record_config",
                    "severity": "error",
                    "path": "spectral_detection_posttrain/configs/registry/experiments.json",
                    "line": None,
                    "message": f"record {record['id']!r} config_path {config_path!r} does not exist",
                }
            )
            continue
        try:
            payload = _load_json(cfg)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            findings.append(
                {
                    "category": "malformed_record_config",
                    "severity": "error",
                    "path": _relative(root, cfg),
                    "line": None,
                    "message": f"record {record['id']!r} config is unreadable: {exc}",
                }
            )
            continue
        if payload.get("version_id") != record["id"]:
            findings.append(
                {
                    "category": "version_id_mismatch",
                    "severity": "error",
                    "path": _relative(root, cfg),
                    "line": None,
                    "message": (
                        f"record {record['id']!r} points to config whose "
                        f"version_id is {payload.get('version_id')!r}"
                    ),
                }
            )

    return findings


def check_historical_status_phrases(root: Path) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    pattern = re.compile(
        "|".join(f"(?P<p{i}>{p})" for i, p in enumerate(HISTORICAL_STATUS_PHRASES)),
        re.IGNORECASE,
    )
    for path in sorted(_iter_docs(root)):
        rel = _relative(root, path)
        if rel in CURRENT_DOCS or rel in GENERATED_RESEARCH_DOCS or rel in HISTORICAL_STATUS_EXCLUDED_DOCS:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for m in pattern.finditer(text):
            findings.append(
                {
                    "category": "historical_status_phrase",
                    "severity": "warning",
                    "path": rel,
                    "line": _line_at(text, m.start()),
                    "message": f"forbidden current-status phrase in historical doc: {m.group(0)!r}",
                }
            )
    return findings


def check_eval_scope_labels(root: Path) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    metric_header_words = ("ap", "metric", "score", "predictions", "map", "ece", "precision", "recall")
    for path in sorted(_iter_docs(root)):
        rel = _relative(root, path)
        if rel not in CURRENT_DOCS and rel not in GENERATED_RESEARCH_DOCS:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        lines = text.splitlines()
        header = ""
        header_has_metric = False
        in_table = False
        for i, line in enumerate(lines):
            if line.startswith("|"):
                if not in_table:
                    header = line
                    header_has_metric = any(word in header.lower() for word in metric_header_words)
                    in_table = True
                else:
                    row_lower = line.lower()
                    if header_has_metric and any(token in row_lower for token in METRIC_TOKENS):
                        if "scope" not in header.lower():
                            findings.append(
                                {
                                    "category": "missing_scope_label",
                                    "severity": "warning",
                                    "path": rel,
                                    "line": i + 1,
                                    "message": "metric table row lacks a Scope column",
                                }
                            )
            else:
                in_table = False
                header = ""
                header_has_metric = False
    return findings


def _is_negated_status(line: str, match_start: int) -> bool:
    """Return True if the match is negated in the same line/sentence."""
    prefix = line[:match_start].lower()
    return bool(re.search(r"\b(no|not|never)\b", prefix))


def check_current_status_consistency(root: Path) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    lines = _load_research_lines(root)
    records = _load_experiment_records(root)
    active_lines = {line["id"] for line in lines if line["status"] in ("active", "queued")}
    authorized = {
        record["id"]
        for record in records
        if record.get("runnable") == "authorized"
        and record.get("research_line") in active_lines
    }

    pattern = re.compile(
        r"\b(currently\s+active|current\s+active|is\s+active|active\s+(?:line|claim|research|family|path|question)|"
        r"authorized(?:\s+experiment)?|in\s+progress|ongoing|currently\s+authorized|currently\s+queued)\b",
        re.IGNORECASE,
    )
    for path in sorted(_iter_docs(root)):
        rel = _relative(root, path)
        if rel not in CURRENT_STATUS_DOCS:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        line_starts = [m.start() for m in re.finditer(r"^", text, re.MULTILINE)]
        for m in pattern.finditer(text):
            line_no = _line_at(text, m.start())
            line_start = line_starts[line_no - 1] if line_no - 1 < len(line_starts) else 0
            line_end = text.find("\n", m.start())
            if line_end == -1:
                line_end = len(text)
            line = text[line_start:line_end]
            match_offset_in_line = m.start() - line_start
            if _is_negated_status(line, match_offset_in_line):
                continue
            findings.append(
                {
                    "category": "current_status_mismatch",
                    "severity": "error",
                    "path": rel,
                    "line": line_no,
                    "message": (
                        f"current doc declares status {m.group(0)!r} independently of the registry; "
                        f"active_lines={sorted(active_lines)}, authorized={sorted(authorized)}"
                    ),
                }
            )
    return findings


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_all(root: Path | None = None) -> dict[str, Any]:
    root = Path(root or ROOT)
    findings: list[dict[str, Any]] = []
    checks = [
        check_encoding,
        check_links,
        check_inline_paths,
        check_generated_drift,
        check_duplicate_experiment_ids,
        check_historical_status_phrases,
        check_eval_scope_labels,
        check_current_status_consistency,
    ]
    for fn in checks:
        findings.extend(fn(root))

    summary: dict[str, int] = {}
    for finding in findings:
        summary[finding["category"]] = summary.get(finding["category"], 0) + 1

    return {
        "root": str(root),
        "findings": findings,
        "summary": summary,
        "ok": len(findings) == 0,
    }


def _format_finding(finding: dict[str, Any]) -> str:
    line = f":{finding['line']}" if finding["line"] else ""
    return f"[{finding['severity']}] {finding['category']} {finding['path']}{line}\n    {finding['message']}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json", action="store_true", help="emit findings as JSON instead of human text"
    )
    parser.add_argument(
        "--root", type=Path, default=ROOT, help="repository root to check"
    )
    args = parser.parse_args(argv)

    result = run_all(args.root)

    errors = [f for f in result["findings"] if f.get("severity") == "error"]
    error_free = len(errors) == 0

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        if result["ok"]:
            print("No documentation integrity findings.")
        elif error_free:
            print("Documentation integrity findings (warnings only):")
            for finding in result["findings"]:
                print(_format_finding(finding))
            print("\nSummary:", result["summary"])
        else:
            print("Documentation integrity findings:")
            for finding in result["findings"]:
                print(_format_finding(finding))
            print("\nSummary:", result["summary"])

    return 0 if error_free else 1


if __name__ == "__main__":
    sys.exit(main())
