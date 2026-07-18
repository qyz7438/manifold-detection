from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts.experiments.re_roi_counterfactual.gates import GateResult
from scripts.experiments.re_roi_counterfactual.run_protocol import run


def _args(tmp_path: Path) -> argparse.Namespace:
    files = {}
    for name in ("fit_cache", "tune_cache", "calibration_cache", "split_manifest", "checkpoint", "annotation"):
        path = tmp_path / f"{name}.bin"
        path.write_bytes(name.encode("ascii"))
        files[name] = path
    return argparse.Namespace(
        **files,
        output_dir=tmp_path / "run",
        device="cpu",
        seed=42,
    )


def _split() -> dict:
    return {
        "version_id": "det.energy.re_roi_counterfactual.split.001",
        "splits": {
            "fit": {"image_ids": list(range(0, 250))},
            "tune": {"image_ids": list(range(250, 318))},
            "calibration": {"image_ids": list(range(318, 386))},
            "outer_heldout": {"image_ids": list(range(386, 454))},
        },
    }


def _loaded(split_name: str) -> tuple[list[dict], dict, str]:
    return (
        [{"image_id": 1, "actions": []}],
        {
            "split_name": split_name,
            "git_commit": "cache-source-commit",
            "model_config": {"min_size": 480, "max_size": 480},
            "cache_config": {"top_k_candidates": 3},
        },
        f"{split_name}-cache-sha",
    )


def _allow_test_split_hash(args: argparse.Namespace, monkeypatch) -> None:
    digest = hashlib.sha256(args.split_manifest.read_bytes()).hexdigest()
    monkeypatch.setattr(
        "scripts.experiments.re_roi_counterfactual.run_protocol.LOCKED_SPLIT_SHA256",
        digest,
    )


def test_protocol_runner_records_provenance_and_all_gate_results(tmp_path, monkeypatch) -> None:
    args = _args(tmp_path)
    _allow_test_split_hash(args, monkeypatch)
    monkeypatch.setattr(
        "scripts.experiments.re_roi_counterfactual.run_protocol._assert_clean_git",
        lambda: "runner-commit",
    )
    monkeypatch.setattr(
        "scripts.experiments.re_roi_counterfactual.run_protocol.load_split_manifest",
        lambda _path: _split(),
    )
    monkeypatch.setattr(
        "scripts.experiments.re_roi_counterfactual.run_protocol.load_validated_cache",
        lambda path, **kwargs: _loaded(kwargs["split_name"]),
    )
    support = GateResult("support", True, 1.0, "locked", {"tune_images": 68})
    monkeypatch.setattr(
        "scripts.experiments.re_roi_counterfactual.run_protocol.check_support",
        lambda *_args, **_kwargs: support,
    )
    gates = {
        name: GateResult(name, True, 1.0, "locked", {})
        for name in (
            "support",
            "identity",
            "re_roi_gain",
            "bundle_integrity",
            "static_baseline",
            "calibration",
            "generalization",
        )
    }
    monkeypatch.setattr(
        "scripts.experiments.re_roi_counterfactual.run_protocol.run_protocol_evaluation",
        lambda *_args, **_kwargs: SimpleNamespace(
            gates=gates,
            metrics={"tune": {"C": {"residual": {"mae": 0.1}}}},
            controls={"always_noop": {"mean_raw_utility": 0.0}},
            models={"C": torch.nn.Linear(2, 1)},
            protocol_state={"feature_dim": 2, "prior": {"score_up": 0.1}},
        ),
    )

    result = run(args)

    assert result["all_gates_passed"] is True
    assert result["evaluation_scope"] == "nwpu_train_only_fit_tune_calibration"
    assert set(result["gates"]) == set(gates)
    assert result["metrics"]["tune"]["C"]["residual"]["mae"] == 0.1
    manifest = json.loads((args.output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["runner_git_commit"] == "runner-commit"
    assert manifest["cache_source_git_commit"] == "cache-source-commit"
    assert manifest["cache_sha256"]["tune"] == "tune-cache-sha"
    assert manifest["outer_heldout_read"] is False
    assert manifest["cache_paths"]["fit"] == str(args.fit_cache.resolve())
    assert (args.output_dir / "eval_metrics.json").is_file()
    checkpoint = torch.load(args.output_dir / "arm_C_final.pt", map_location="cpu")
    assert checkpoint["protocol_state"]["feature_dim"] == 2


def test_protocol_runner_stops_after_failed_support_gate(tmp_path, monkeypatch) -> None:
    args = _args(tmp_path)
    _allow_test_split_hash(args, monkeypatch)
    monkeypatch.setattr(
        "scripts.experiments.re_roi_counterfactual.run_protocol._assert_clean_git",
        lambda: "runner-commit",
    )
    monkeypatch.setattr(
        "scripts.experiments.re_roi_counterfactual.run_protocol.load_split_manifest",
        lambda _path: _split(),
    )
    monkeypatch.setattr(
        "scripts.experiments.re_roi_counterfactual.run_protocol.load_validated_cache",
        lambda path, **kwargs: _loaded(kwargs["split_name"]),
    )
    support = GateResult("support", False, 0.5, "locked", {"tune_images": 20})
    monkeypatch.setattr(
        "scripts.experiments.re_roi_counterfactual.run_protocol.check_support",
        lambda *_args, **_kwargs: support,
    )

    def should_not_train(*_args, **_kwargs):
        raise AssertionError("training must not start when support fails")

    monkeypatch.setattr(
        "scripts.experiments.re_roi_counterfactual.run_protocol.run_protocol_evaluation",
        should_not_train,
    )

    result = run(args)

    assert result["all_gates_passed"] is False
    assert result["training_started"] is False
    assert set(result["gates"]) == {"support"}


def test_protocol_runner_refuses_reused_output_directory(tmp_path) -> None:
    args = _args(tmp_path)
    args.output_dir.mkdir()

    with pytest.raises(FileExistsError, match="refusing to reuse"):
        run(args)


def test_protocol_runner_locks_seed_42(tmp_path) -> None:
    args = _args(tmp_path)
    args.seed = 2024

    with pytest.raises(ValueError, match="seed must be 42"):
        run(args)


def test_protocol_runner_rejects_same_shape_but_noncanonical_split(tmp_path, monkeypatch) -> None:
    args = _args(tmp_path)
    monkeypatch.setattr(
        "scripts.experiments.re_roi_counterfactual.run_protocol._assert_clean_git",
        lambda: "runner-commit",
    )
    monkeypatch.setattr(
        "scripts.experiments.re_roi_counterfactual.run_protocol.load_split_manifest",
        lambda _path: _split(),
    )

    with pytest.raises(ValueError, match="locked split SHA256"):
        run(args)
