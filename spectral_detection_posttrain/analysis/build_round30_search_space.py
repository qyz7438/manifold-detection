from __future__ import annotations

import argparse
import json
from pathlib import Path

SEEDS = [42, 43]


def _base(name: str, signal: str, seed: int) -> dict:
    return {
        "name": name, "signal": signal, "reward_lambda": 0.0,
        "struct_weight": 0.0, "policy_loss_weight": 0.0003,
        "det_loss_weight": 0.0, "baseline_kl_weight": 10.0,
        "box_loss_weight": 0.0, "unfreeze": "cls", "optimizer": "adamw",
        "temperature": 1.0, "max_candidates": 40,
        "reward_score_threshold": 0.2, "rollout_source": "baseline",
        "policy_objective": "signed", "seed": seed,
    }


def build_phase_a_presets() -> list[dict]:
    presets: list[dict] = []
    for seed in SEEDS:
        null = _base(f"null_no_update_seed{seed}", "none", seed)
        null.update({"policy_loss_weight": 0.0, "baseline_kl_weight": 0.0})
        presets.append(null)
        presets.append(_base(f"signed_iou_seed{seed}", "none", seed))

    reward_lambdas = [0.025, 0.05, 0.1, 0.2]
    policy_weights = [0.0001, 0.0003]
    kl_weights = [5.0, 10.0]
    for seed in SEEDS:
        for signal, label in [("ramp", "amp"), ("shuffled_amp", "shuffled_amp")]:
            for rl in reward_lambdas:
                for pw in policy_weights:
                    for kl in kl_weights:
                        preset = _base(f"signed_{label}_l{rl:g}_pl{pw:g}_kl{kl:g}_seed{seed}", signal, seed)
                        preset.update({"reward_lambda": rl, "policy_loss_weight": pw, "baseline_kl_weight": kl})
                        presets.append(preset)
    return presets


def build_phase_b_presets(best_reward_lambda: float = 0.1) -> list[dict]:
    presets: list[dict] = []
    for seed in SEEDS:
        for pw in [0.0001, 0.0003, 0.0007]:
            for kl in [5.0, 10.0, 20.0]:
                for mc in [20, 40, 80]:
                    preset = _base(f"amp_tune_l{best_reward_lambda:g}_pl{pw:g}_kl{kl:g}_mc{mc}_seed{seed}", "ramp", seed)
                    preset.update({"reward_lambda": best_reward_lambda, "policy_loss_weight": pw,
                                   "baseline_kl_weight": kl, "max_candidates": mc})
                    presets.append(preset)
    return presets


def build_phase_c_presets(best_reward_lambda: float = 0.1) -> list[dict]:
    presets: list[dict] = []
    for seed in SEEDS:
        for signal, label in [
            ("structure", "structure"), ("shuffled_structure", "shuffled_structure"),
            ("amp_structure", "amp_structure"), ("shuffled_amp_structure", "shuffled_amp_structure"),
        ]:
            for sw in [0.05, 0.1, 0.2]:
                preset = _base(f"{label}_sw{sw:g}_seed{seed}", signal, seed)
                preset.update({
                    "reward_lambda": best_reward_lambda if "amp" in signal else 0.0,
                    "struct_weight": sw,
                })
                presets.append(preset)
    return presets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=["A", "B", "C"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--best-reward-lambda", type=float, default=0.1)
    args = parser.parse_args()

    if args.phase == "A":
        presets = build_phase_a_presets()
    elif args.phase == "B":
        presets = build_phase_b_presets(best_reward_lambda=args.best_reward_lambda)
    else:
        presets = build_phase_c_presets(best_reward_lambda=args.best_reward_lambda)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"preset": {"_type": "choice", "_value": presets}}, indent=2), encoding="utf-8")
    print(f"wrote {len(presets)} presets to {args.output}")


if __name__ == "__main__":
    main()
