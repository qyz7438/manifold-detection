"""Explicit experiment dispatcher CLI (refactor Task 7).

Thin argparse front-end over
``spectral_detection_posttrain.experiments.dispatcher``. All scientific
state comes from the committed registries; this script never mutates them.

Commands:

* ``python scripts/run_experiment.py list`` — all registry records grouped
  by record kind.
* ``python scripts/run_experiment.py status`` — research-line status and
  authorization summary.
* ``python scripts/run_experiment.py validate --experiment <id>`` —
  registry plus handler-level validation of one record.
* ``python scripts/run_experiment.py dry-run --experiment <id> --run-name
  <name>`` — deterministic, side-effect-free dispatch plan (resolved ID,
  research status, config hash, required inputs, evaluation scope, GPU
  policy, expected outputs, exact command). Frozen records additionally
  require ``--allow-frozen-reproduction``. Never creates a run directory.
* ``python scripts/run_experiment.py run --experiment <id> --run-name
  <name>`` — refused for every record while the registry authorizes zero
  experiments.

Exit codes: 0 on success, 1 on any dispatch guard violation, 2 on argparse
usage errors.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spectral_detection_posttrain.experiments.dispatcher import (
    DispatchError,
    dispatch_dry_run,
    dispatch_run,
    dispatch_validate,
    load_dispatcher_registry,
    render_plan,
    render_record_list,
    render_status,
)

def _parse_overrides(items: list[str] | None) -> dict[str, Any]:
    """Parse repeated ``--override key=value`` items into a plain dict."""
    overrides: dict[str, Any] = {}
    for item in items or []:
        key, separator, value = item.partition("=")
        if not separator or not key:
            raise DispatchError(
                f"--override expects key=value, got {item!r}"
            )
        if key in overrides:
            raise DispatchError(f"duplicate --override key: {key!r}")
        overrides[key] = value
    return overrides


def _add_experiment_arguments(parser: argparse.ArgumentParser, *, run: bool) -> None:
    parser.add_argument("--experiment", required=True, help="registry experiment ID")
    if run:
        parser.add_argument("--run-name", required=True, help="intended run name")
        parser.add_argument(
            "--allow-frozen-reproduction",
            action="store_true",
            help="acknowledge exact-config reproduction of a frozen record",
        )
        parser.add_argument(
            "--override",
            action="append",
            default=None,
            metavar="KEY=VALUE",
            help="config override (repeatable); frozen records reject every override",
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_experiment.py",
        description="Explicit experiment dispatcher over the committed registries.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("list", help="list all registry records")
    subparsers.add_parser("status", help="show research status and authorization summary")
    _add_experiment_arguments(subparsers.add_parser("validate", help="validate one record"), run=False)
    _add_experiment_arguments(subparsers.add_parser("dry-run", help="render a side-effect-free plan"), run=True)
    _add_experiment_arguments(subparsers.add_parser("run", help="execute (currently refused)"), run=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        registry = load_dispatcher_registry()
        if args.command == "list":
            print(render_record_list(registry))
            return 0
        if args.command == "status":
            print(render_status(registry))
            return 0
        if args.command == "validate":
            print(dispatch_validate(registry, args.experiment))
            return 0
        overrides = _parse_overrides(getattr(args, "override", None))
        if args.command == "dry-run":
            plan = dispatch_dry_run(
                registry,
                args.experiment,
                args.run_name,
                allow_frozen_reproduction=args.allow_frozen_reproduction,
                overrides=overrides,
            )
            print(render_plan(plan))
            return 0
        if args.command == "run":
            dispatch_run(
                registry,
                args.experiment,
                args.run_name,
                allow_frozen_reproduction=args.allow_frozen_reproduction,
                overrides=overrides,
            )
            return 0
        raise DispatchError(f"unknown command: {args.command!r}")
    except DispatchError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
