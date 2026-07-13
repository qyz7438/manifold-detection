from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
VALIDATOR = ROOT / "scripts" / "dev" / "validate_repository_state.py"


def _load_module():
    assert VALIDATOR.exists(), "repository-state validator has not been implemented"
    spec = importlib.util.spec_from_file_location("validate_repository_state", VALIDATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def validator():
    return _load_module()


def test_pytest_collects_only_maintained_tests():
    config = Path("pyproject.toml").read_text(encoding="utf-8")
    assert 'testpaths = ["tests"]' in config


def test_repository_state_rejects_runtime_artifacts(validator):
    violations = validator.find_tracked_runtime_artifacts(validator.ROOT)
    assert violations == []


def test_user_owned_untracked_document_is_never_an_auto_stage_target(validator):
    assert "docs/energy_transport_vs_fpn_sm_analysis.md" in validator.EXPLICIT_EXCLUSIONS


def test_validator_root_resolves_to_repo_root(validator):
    assert validator.ROOT == ROOT
    assert (validator.ROOT / ".gitignore").exists()


@pytest.mark.parametrize(
    "path",
    [
        "runs/foo/metrics.json",
        ".agent_reports/x.txt",
        "data/raw/img.png",
        "model.pth",
        "weights/model.pt",
        "checkpoints/last.ckpt",
        "cache/features.npz",
        "cache/features.npy",
        "logs/train.log",
        "server.pem",
        "id_rsa",
        ".env",
    ],
)
def test_forbidden_patterns_flag_runtime_artifacts(validator, path):
    assert validator.is_forbidden(path), path


@pytest.mark.parametrize(
    "path",
    [
        "output/roi_dual_energy_controlled_anchor.json",
        "docs/reports/round280_audit.json",
        "cloud_id_twonn.json",
        "spectral_detection_posttrain/configs/versions/det.energy.set_policy.m1.001.json",
        "assets/repochan/app-icon.png",
        "docs/reports/redesign_grid_summary.csv",
        "output/pdf/rlimage_full_experiment_report_2026-06-04.pdf",
        "tmp/pdfs/report_render/page_1.png",
        "scripts/train_nwpu_m1_set_policy.py",
        ".env.example",
    ],
)
def test_forbidden_patterns_allow_maintained_files(validator, path):
    assert not validator.is_forbidden(path), path
