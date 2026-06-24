from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_sweep_module():
    module_path = Path(__file__).resolve().parents[1] / "run_nwpu_active_sweep.py"
    spec = importlib.util.spec_from_file_location("run_nwpu_active_sweep", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_active_sweep_command_keeps_active_correction_flags() -> None:
    sweep = _load_sweep_module()

    cmd = sweep.build_training_command(
        config="cfg.yaml",
        baseline="base.pth",
        run_name="active_run",
        epochs=10,
        num_prototypes=4,
        lambda_tr=0.01,
        lambda_en=0.001,
        lr=1e-5,
        lr_manifold=1e-4,
        active_gamma=0.15,
    )

    assert "--active-manifold-correction" in cmd
    assert cmd[cmd.index("--active-correction-gamma") + 1] == "0.15"
    assert "--lambda-fc1-rank" not in cmd
    assert "--lambda-logit-preserve" not in cmd


def test_build_active_sweep_command_can_select_gated_endpoint_mode() -> None:
    sweep = _load_sweep_module()

    cmd = sweep.build_training_command(
        config="cfg.yaml",
        baseline="base.pth",
        run_name="active_endpoint",
        epochs=2,
        num_prototypes=4,
        lambda_tr=0.01,
        lambda_en=0.001,
        lr=1e-5,
        lr_manifold=1e-4,
        active_gamma=0.2,
        active_correction_mode="gated_endpoint",
        active_endpoint_gate_init=0.4,
    )

    assert "--active-manifold-correction" in cmd
    assert cmd[cmd.index("--active-correction-mode") + 1] == "gated_endpoint"
    assert cmd[cmd.index("--active-endpoint-gate-init") + 1] == "0.4"


def test_build_fc1_preserve_command_adds_geometry_preservation_and_limits() -> None:
    sweep = _load_sweep_module()

    cmd = sweep.build_training_command(
        config="cfg.yaml",
        baseline="base.pth",
        run_name="fc1_keep",
        epochs=2,
        num_prototypes=4,
        lambda_tr=0.0,
        lambda_en=0.0,
        lr=1e-5,
        lr_manifold=1e-4,
        active_gamma=None,
        fc1_rank=0.5,
        fc1_compact=0.05,
        fc1_rank_target=8,
        logit_preserve=0.2,
        bbox_preserve=1.0,
        preserve_temperature=2.0,
        proj_intra=0.05,
        proto_div=0.01,
        proj_inter=0.02,
        projection_inter_margin=0.4,
        proto_div_temperature=0.2,
        limit_train=80,
        limit_val=80,
        warmup_batches=0,
    )

    assert "--active-manifold-correction" not in cmd
    assert cmd[cmd.index("--lambda-fc1-rank") + 1] == "0.5"
    assert cmd[cmd.index("--lambda-fc1-compact") + 1] == "0.05"
    assert cmd[cmd.index("--fc1-rank-target") + 1] == "8"
    assert cmd[cmd.index("--lambda-logit-preserve") + 1] == "0.2"
    assert cmd[cmd.index("--lambda-bbox-preserve") + 1] == "1.0"
    assert cmd[cmd.index("--lambda-proj-intra") + 1] == "0.05"
    assert cmd[cmd.index("--lambda-proto-div") + 1] == "0.01"
    assert cmd[cmd.index("--lambda-proj-inter") + 1] == "0.02"
    assert cmd[cmd.index("--projection-inter-margin") + 1] == "0.4"
    assert cmd[cmd.index("--proto-div-temperature") + 1] == "0.2"
    assert cmd[cmd.index("--limit-train") + 1] == "80"
    assert cmd[cmd.index("--limit-val") + 1] == "80"
    assert cmd[cmd.index("--warmup-batches") + 1] == "0"
