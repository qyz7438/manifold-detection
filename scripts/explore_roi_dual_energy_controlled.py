"""Controlled experiment for ROI intra/inter dual energy.

This script is intentionally synthetic.  It checks whether the maintained
dual-energy objective separates the two failure modes that motivated the
basic-project transfer:

* intra-class failure: ROI features of the same class are scattered or unstable;
* inter-class failure: class prototypes are compact but have the wrong
  class-relation structure.

The output is written under ``output/`` so it is treated as a local artifact.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spectral_detection_posttrain.methods.energy_transport import (  # noqa: E402
    centered_relation_matrix,
    inter_class_separation_energy,
    relation_cka,
    roi_dual_energy,
)
from spectral_detection_posttrain.methods.energy_transport.cone_projection import normalize_l2  # noqa: E402


@dataclass(frozen=True)
class VariantResult:
    variant: str
    e_compact: float
    e_basin: float
    e_intra: float
    e_inter: float
    dual_energy: float
    inter_reference_alignment: float
    inter_anchor_energy: float
    inter_separation_energy: float
    reference_cls_acc: float
    current_proto_cls_acc: float
    final_loss: float


def make_reference_prototypes(num_classes: int, dim: int, *, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    raw = torch.randn((num_classes, dim), generator=generator)
    raw[:, 0] += torch.linspace(-1.2, 1.2, steps=num_classes)
    raw[:, 1] += torch.sin(torch.linspace(0.0, math.pi, steps=num_classes))
    raw[:, 2] += torch.cos(torch.linspace(0.0, 2.0 * math.pi, steps=num_classes))
    return normalize_l2(raw)


def make_initial_state(
    reference: torch.Tensor,
    *,
    samples_per_class: int,
    feature_noise: float,
    prototype_noise: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    num_classes, dim = reference.shape
    perm = torch.arange(num_classes).roll(1)
    proto = reference[perm] + prototype_noise * torch.randn((num_classes, dim), generator=generator)
    proto = normalize_l2(proto)
    labels = torch.arange(num_classes).repeat_interleave(samples_per_class)
    features = proto[labels] + feature_noise * torch.randn((labels.numel(), dim), generator=generator)
    return features, labels, proto


def evaluate(
    features: torch.Tensor,
    labels: torch.Tensor,
    prototypes: torch.Tensor,
    reference: torch.Tensor,
    *,
    final_loss: float,
    variant: str,
) -> VariantResult:
    energy = roi_dual_energy(
        features,
        labels,
        prototypes=prototypes,
        reference_prototypes=reference,
        perturb_radius=0.0,
        num_perturbations=0,
        separation_weight=0.1,
    )
    z = normalize_l2(features.detach())
    current_proto = normalize_l2(prototypes.detach())
    reference_proto = normalize_l2(reference.detach())
    current_pred = (z @ current_proto.t()).argmax(dim=1)
    reference_pred = (z @ reference_proto.t()).argmax(dim=1)
    alignment = relation_cka(
        centered_relation_matrix(prototypes.detach()),
        centered_relation_matrix(reference.detach()),
    )
    separation = inter_class_separation_energy(prototypes.detach())
    return VariantResult(
        variant=variant,
        e_compact=float(energy.compact_energy.detach().item()),
        e_basin=float(energy.basin_energy.detach().item()),
        e_intra=float(energy.intra_energy.detach().item()),
        e_inter=float(energy.inter_energy.detach().item()),
        dual_energy=float(energy.dual_energy.detach().item()),
        inter_reference_alignment=float(alignment.detach().item()),
        inter_anchor_energy=float(energy.components["inter_anchor_energy"].detach().item()),
        inter_separation_energy=float(separation.detach().item()),
        reference_cls_acc=float((reference_pred.cpu() == labels.cpu()).float().mean().item()),
        current_proto_cls_acc=float((current_pred.cpu() == labels.cpu()).float().mean().item()),
        final_loss=float(final_loss),
    )


def optimize_variant(
    variant: str,
    init_features: torch.Tensor,
    labels: torch.Tensor,
    init_prototypes: torch.Tensor,
    reference: torch.Tensor,
    *,
    steps: int,
    lr: float,
) -> VariantResult:
    if variant == "baseline":
        return evaluate(
            init_features,
            labels,
            init_prototypes,
            reference,
            final_loss=float("nan"),
            variant=variant,
        )

    features = init_features.clone().detach().requires_grad_(True)
    prototypes = init_prototypes.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([features, prototypes], lr=lr)
    final_loss = None
    for _ in range(int(steps)):
        energy = roi_dual_energy(
            features,
            labels,
            prototypes=prototypes,
            reference_prototypes=reference,
            perturb_radius=0.0,
            num_perturbations=0,
            separation_weight=0.1,
        )
        if variant == "intra_only":
            loss = energy.intra_energy
        elif variant == "inter_only":
            loss = energy.inter_energy
        elif variant == "dual":
            loss = energy.dual_energy
        else:
            raise ValueError(f"Unknown variant: {variant}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().item())

    assert final_loss is not None
    return evaluate(
        features.detach(),
        labels,
        prototypes.detach(),
        reference,
        final_loss=final_loss,
        variant=variant,
    )


def print_table(results: list[VariantResult]) -> None:
    columns = [
        ("variant", "variant"),
        ("E_intra", "e_intra"),
        ("E_inter", "e_inter"),
        ("Dual", "dual_energy"),
        ("compact", "e_compact"),
        ("basin", "e_basin"),
        ("rel_align", "inter_reference_alignment"),
        ("anchor", "inter_anchor_energy"),
        ("ref_acc", "reference_cls_acc"),
        ("cur_acc", "current_proto_cls_acc"),
    ]
    header = " | ".join(name.rjust(11) for name, _ in columns)
    print(header)
    print("-" * len(header))
    for result in results:
        cells = []
        payload = asdict(result)
        for _, key in columns:
            value = payload[key]
            if isinstance(value, str):
                cells.append(value.rjust(11))
            else:
                cells.append(f"{value:11.4f}")
        print(" | ".join(cells))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-classes", type=int, default=6)
    parser.add_argument("--dim", type=int, default=12)
    parser.add_argument("--samples-per-class", type=int, default=80)
    parser.add_argument("--feature-noise", type=float, default=0.65)
    parser.add_argument("--prototype-noise", type=float, default=0.15)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.04)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("output/roi_dual_energy_controlled.json"))
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    reference = make_reference_prototypes(args.num_classes, args.dim, seed=args.seed)
    features, labels, prototypes = make_initial_state(
        reference,
        samples_per_class=args.samples_per_class,
        feature_noise=args.feature_noise,
        prototype_noise=args.prototype_noise,
        seed=args.seed + 1,
    )

    results = [
        optimize_variant(
            variant,
            features,
            labels,
            prototypes,
            reference,
            steps=args.steps,
            lr=args.lr,
        )
        for variant in ["baseline", "intra_only", "inter_only", "dual"]
    ]

    print_table(results)

    payload = {
        "config": {
            "num_classes": args.num_classes,
            "dim": args.dim,
            "samples_per_class": args.samples_per_class,
            "feature_noise": args.feature_noise,
            "prototype_noise": args.prototype_noise,
            "steps": args.steps,
            "lr": args.lr,
            "seed": args.seed,
        },
        "results": [asdict(result) for result in results],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved {args.output}")


if __name__ == "__main__":
    main()
