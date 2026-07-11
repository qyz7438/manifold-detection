"""Summarize the utility lattice in detector-action cache artifacts."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import torch


def summarize_records(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    best_values = []
    margins = []
    candidate_values = []
    positive_candidates = 0
    candidate_count = 0
    for record in records:
        delta_u = record["delta_u"].detach().cpu().float()
        observable = record["observable_mask"].detach().cpu().bool()
        values = delta_u[observable, 1:].reshape(-1)
        values = values[torch.isfinite(values)]
        candidate_count += int(values.numel())
        if values.numel():
            candidate_values.extend(float(value) for value in values.tolist())
            positive_candidates += int(values.gt(0.0).sum().item())
            ranked = torch.sort(torch.cat((values, values.new_zeros(1))), descending=True).values
            best_values.append(float(ranked[0].item()))
            margins.append(float((ranked[0] - ranked[1]).item()) if ranked.numel() > 1 else 0.0)
        else:
            best_values.append(0.0)
            margins.append(0.0)
    rounded = Counter(round(value, 6) for value in candidate_values)
    positive = [value for value in best_values if value > 0.0]
    return {
        "images": len(records),
        "candidate_count": candidate_count,
        "positive_candidate_count": positive_candidates,
        "positive_candidate_rate": positive_candidates / max(1, candidate_count),
        "positive_image_count": len(positive),
        "positive_image_rate": len(positive) / max(1, len(records)),
        "best_delta_u_mean": sum(best_values) / max(1, len(best_values)),
        "positive_best_delta_u_mean": sum(positive) / max(1, len(positive)),
        "best_vs_runner_up_margin_mean": sum(margins) / max(1, len(margins)),
        "unique_rounded_delta_u": len(rounded),
        "most_common_delta_u": [
            {"value": value, "count": count}
            for value, count in rounded.most_common(20)
        ],
    }


def summarize_cache(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu")
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError(f"cache {path} has no records list")
    return {
        "path": str(path),
        "format": payload.get("format"),
        "metadata": payload.get("metadata", {}),
        "summary": summarize_records(records),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = {"caches": [summarize_cache(path) for path in args.paths]}
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
