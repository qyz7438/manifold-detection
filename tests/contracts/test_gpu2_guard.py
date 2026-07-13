"""Contract tests for the GPU2 launch guard (Task 20).

All tests are CPU-only and mock ``nvidia-smi``.  No test trains, queries a
real GPU, or signals any process.
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
GUARD_PATH = ROOT / "scripts" / "run" / "guard_gpu2.py"


def _load_guard():
    spec = importlib.util.spec_from_file_location("guard_gpu2", GUARD_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


guard = _load_guard()


def _make_nvidia_smi_script(tmp_path: Path, stdout: str, returncode: int = 0) -> Path:
    script = tmp_path / "nvidia-smi"
    if sys.platform == "win32":
        script = tmp_path / "nvidia-smi.bat"
        script.write_text(
            f"@echo off\necho {stdout}\nexit /b {returncode}\n", encoding="utf-8"
        )
    else:
        script.write_text(
            f"#!/bin/sh\nprintf '%s\\n' '{stdout}'\nexit {returncode}\n",
            encoding="utf-8",
        )
        script.chmod(0o755)
    return script


def _make_registries(tmp_path: Path, experiment: dict, line: dict) -> tuple[Path, Path]:
    experiments = {"schema_version": "experiments.v1", "records": [experiment]}
    lines = {"lines": [line]}
    exp_path = tmp_path / "experiments.json"
    line_path = tmp_path / "research_lines.json"
    exp_path.write_text(json.dumps(experiments), encoding="utf-8")
    line_path.write_text(json.dumps(lines), encoding="utf-8")
    return exp_path, line_path


def _active_experiment() -> dict:
    return {
        "id": "test.active.001",
        "research_line": "test.line.active",
        "capability": "test",
        "config_path": "",
        "legacy_entrypoint": "scripts/dummy.py",
        "handler": "native_contract",
        "runnable": "active",
        "evaluation_scope": {"kind": "smoke", "image_count": 1},
        "required_inputs": [],
        "expected_outputs": [],
        "gpu_policy": "remote_gpu2_guarded",
        "decision_document": "docs/test.md",
    }


def _frozen_experiment() -> dict:
    exp = _active_experiment()
    exp["id"] = "test.frozen.001"
    exp["runnable"] = "frozen_reproduction_only"
    return exp


def test_guard_module_exists():
    assert GUARD_PATH.exists()


def test_guard_has_no_kill_or_signal_helpers():
    """The guard must never contain convenience functions for killing or signalling."""
    names = {name.lower() for name in dir(guard) if not name.startswith("_")}
    forbidden = {"kill", "terminate", "stop", "signal", "reprioritize", "renice"}
    assert not (names & forbidden), f"guard exposes forbidden names: {names & forbidden}"


class TestGpu2Query:
    def test_query_parses_gpu2_memory(self, tmp_path: Path):
        nvidia_smi = _make_nvidia_smi_script(tmp_path, "2, NVIDIA GeForce RTX 4090, 24564, 10000")
        info = guard.query_gpu2(str(nvidia_smi))
        assert info["index"] == 2
        assert "4090" in info["name"]
        assert info["memory_free_mib"] == 10000

    def test_query_rejects_wrong_gpu_index(self, tmp_path: Path):
        nvidia_smi = _make_nvidia_smi_script(tmp_path, "0, NVIDIA RTX A4000, 8192, 8192")
        with pytest.raises(RuntimeError, match="returned GPU 0"):
            guard.query_gpu2(str(nvidia_smi))

    def test_query_rejects_malformed_output(self, tmp_path: Path):
        nvidia_smi = _make_nvidia_smi_script(tmp_path, "not,a,csv,line")
        with pytest.raises(RuntimeError, match="non-integer GPU index"):
            guard.query_gpu2(str(nvidia_smi))

    def test_query_rejects_non_numeric_memory(self, tmp_path: Path):
        nvidia_smi = _make_nvidia_smi_script(tmp_path, "2, RTX 4090, lots, none")
        with pytest.raises(RuntimeError, match="non-numeric memory"):
            guard.query_gpu2(str(nvidia_smi))

    def test_query_rejects_nvidia_smi_failure(self, tmp_path: Path):
        nvidia_smi = _make_nvidia_smi_script(tmp_path, "", returncode=9)
        with pytest.raises(RuntimeError, match="nvidia-smi failed"):
            guard.query_gpu2(str(nvidia_smi))


class TestGitClean:
    def test_clean_git_tree(self, tmp_path: Path):
        file = tmp_path / "tracked.txt"
        file.write_text("x", encoding="utf-8")
        _init_and_commit_repo(tmp_path)
        clean, reason = guard.is_git_clean(tmp_path)
        assert clean and not reason

    def test_dirty_tree_rejected(self, tmp_path: Path):
        file = tmp_path / "tracked.txt"
        file.write_text("x", encoding="utf-8")
        _init_and_commit_repo(tmp_path)
        file.write_text("changed", encoding="utf-8")
        clean, reason = guard.is_git_clean(tmp_path)
        assert not clean
        assert "uncommitted" in reason

    def test_untracked_in_runtime_dir_ignored(self, tmp_path: Path):
        _run(["git", "init"], cwd=tmp_path)
        (tmp_path / "runs").mkdir()
        (tmp_path / "runs" / "stuff.txt").write_text("x", encoding="utf-8")
        clean, reason = guard.is_git_clean(tmp_path)
        assert clean and not reason

    def test_untracked_outside_runtime_dir_rejected(self, tmp_path: Path):
        _run(["git", "init"], cwd=tmp_path)
        (tmp_path / "new.py").write_text("x", encoding="utf-8")
        clean, reason = guard.is_git_clean(tmp_path)
        assert not clean


def _run(cmd, cwd):
    subprocess.run(
        cmd,
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def _init_and_commit_repo(tmp_path: Path) -> None:
    _run(["git", "init"], cwd=tmp_path)
    _run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path)
    _run(["git", "config", "user.name", "Test User"], cwd=tmp_path)
    _run(["git", "add", "-A"], cwd=tmp_path)
    _run(["git", "commit", "-m", "init"], cwd=tmp_path)


class TestLoadExperiment:
    def test_loads_active_experiment_with_active_line(self, tmp_path: Path):
        exp, line = _make_registries(
            tmp_path,
            _active_experiment(),
            {"id": "test.line.active", "status": "active"},
        )
        record, _ = guard.load_experiment("test.active.001", exp, line)
        assert record["id"] == "test.active.001"

    def test_loads_frozen_reproduction_experiment(self, tmp_path: Path):
        exp, line = _make_registries(
            tmp_path,
            _frozen_experiment(),
            {"id": "test.line.active", "status": "frozen"},
        )
        record, _ = guard.load_experiment("test.frozen.001", exp, line)
        assert record["runnable"] == "frozen_reproduction_only"

    def test_rejects_disabled_experiment(self, tmp_path: Path):
        experiment = _active_experiment()
        experiment["runnable"] = "disabled"
        exp, line = _make_registries(
            tmp_path,
            experiment,
            {"id": "test.line.active", "status": "active"},
        )
        with pytest.raises(ValueError, match="only active/active or frozen_reproduction_only/frozen"):
            guard.load_experiment("test.active.001", exp, line)

    def test_rejects_non_gpu2_policy(self, tmp_path: Path):
        experiment = _active_experiment()
        experiment["gpu_policy"] = "local_gpu"
        exp, line = _make_registries(
            tmp_path,
            experiment,
            {"id": "test.line.active", "status": "active"},
        )
        with pytest.raises(ValueError, match="expected remote_gpu2_guarded"):
            guard.load_experiment("test.active.001", exp, line)


class TestCheckAll:
    def test_check_passes_when_all_gates_pass(self, tmp_path: Path):
        nvidia_smi = _make_nvidia_smi_script(tmp_path, "2, RTX 4090, 24564, 9000")
        exp, line = _make_registries(
            tmp_path,
            _active_experiment(),
            {"id": "test.line.active", "status": "active"},
        )

        # monkeypatch the guard to use our temp registries and git root
        original_experiments = guard.EXPERIMENTS_REGISTRY
        original_lines = guard.RESEARCH_LINES_REGISTRY
        guard.EXPERIMENTS_REGISTRY = exp
        guard.RESEARCH_LINES_REGISTRY = line
        try:
            # Use the temp repo for git clean check
            _init_and_commit_repo(tmp_path)
            (tmp_path / "spectral_detection_posttrain").mkdir(parents=True)
            clean, _ = guard.is_git_clean(tmp_path)
            assert clean
            result = guard.check_all("test.active.001", str(nvidia_smi), tmp_path)
        finally:
            guard.EXPERIMENTS_REGISTRY = original_experiments
            guard.RESEARCH_LINES_REGISTRY = original_lines

        assert result["passed"] is True
        assert result["gpu2"]["memory_free_mib"] == 9000

    def test_check_fails_when_memory_at_threshold(self, tmp_path: Path):
        nvidia_smi = _make_nvidia_smi_script(tmp_path, "2, RTX 4090, 24564, 8192")
        exp, line = _make_registries(
            tmp_path,
            _active_experiment(),
            {"id": "test.line.active", "status": "active"},
        )
        original_experiments = guard.EXPERIMENTS_REGISTRY
        original_lines = guard.RESEARCH_LINES_REGISTRY
        guard.EXPERIMENTS_REGISTRY = exp
        guard.RESEARCH_LINES_REGISTRY = line
        try:
            _init_and_commit_repo(tmp_path)
            result = guard.check_all("test.active.001", str(nvidia_smi), tmp_path)
        finally:
            guard.EXPERIMENTS_REGISTRY = original_experiments
            guard.RESEARCH_LINES_REGISTRY = original_lines

        assert result["passed"] is False
        assert "8192" in result["reason"]

    def test_check_fails_when_memory_below_threshold(self, tmp_path: Path):
        nvidia_smi = _make_nvidia_smi_script(tmp_path, "2, RTX 4090, 24564, 7000")
        exp, line = _make_registries(
            tmp_path,
            _active_experiment(),
            {"id": "test.line.active", "status": "active"},
        )
        original_experiments = guard.EXPERIMENTS_REGISTRY
        original_lines = guard.RESEARCH_LINES_REGISTRY
        guard.EXPERIMENTS_REGISTRY = exp
        guard.RESEARCH_LINES_REGISTRY = line
        try:
            _init_and_commit_repo(tmp_path)
            result = guard.check_all("test.active.001", str(nvidia_smi), tmp_path)
        finally:
            guard.EXPERIMENTS_REGISTRY = original_experiments
            guard.RESEARCH_LINES_REGISTRY = original_lines

        assert result["passed"] is False


class TestBuildCommand:
    def test_command_uses_gpu2_and_pythonpath(self):
        experiment = _active_experiment()
        command = guard.build_command(
            experiment,
            python_path="/opt/python",
            workspace_path="/workspace",
        )
        assert "CUDA_VISIBLE_DEVICES=2" in command
        assert "PYTHONPATH=/workspace:$PYTHONPATH" in command
        assert "/opt/python" in command
        assert "scripts/dummy.py" in command


def test_command_hash_is_sha256():
    h = guard.command_hash("echo hello")
    assert len(h) == 64
    assert h == "584a331fd6b02dcb1ecbe2eba731f609a2e1e3dac0bb73ae998dfad14c309a77"
