r"""Aggregate multiple layer-wise geometry JSON reports into a single table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True, help="Paths to layerwise_geometry.json files.")
    parser.add_argument("--out", default="runs/layerwise_geometry_summary.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for p in args.inputs:
        data = json.loads(Path(p).read_text(encoding="utf-8"))
        run_name = data["run_name"]
        for layer, geom in data["layer_geometry"].items():
            rows.append({
                "run": run_name,
                "layer": layer,
                "ambient_dim": geom["ambient_dim"],
                "n": geom["n_samples"],
                "n_fg": geom["n_foreground"],
                "id_fg": geom["id_foreground"],
                "intra_mean": geom["intra_mean"],
                "inter_mean": geom["inter_mean"],
            })

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines = ["# Layer-wise geometry summary\n"]
    lines.append("| run | layer | ambient_dim | n | n_fg | id_fg | intra_mean | inter_mean |\n")
    lines.append("|-----|-------|-------------|---|------|-------|------------|------------|\n")
    for r in rows:
        lines.append(
            f"| {r['run']} | {r['layer']} | {r['ambient_dim']} | {r['n']} | {r['n_fg']} | "
            f"{r['id_fg']:.2f} | {r['intra_mean']:.4f} | {r['inter_mean']:.4f} |\n"
        )

    out_path.write_text("".join(lines), encoding="utf-8")
    print("".join(lines))
    print(f"Saved summary to {out_path}")


if __name__ == "__main__":
    main()
