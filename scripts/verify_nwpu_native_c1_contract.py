"""Frozen-reproduction wrapper: implementation archived to ``scripts/archive/historical/nwpu_c1/verify_nwpu_native_c1_contract.py``.

Task 16 wave C archived this frozen experiment entrypoint
(``det.energy.native_listwise.c1.001``, ``frozen_reproduction_only``) to
``scripts/archive/historical/nwpu_c1/``. This wrapper keeps the historical
entrypoint working for exact reproduction:

- importing ``scripts.verify_nwpu_native_c1_contract`` re-exports the archived module's public names;
- running ``python scripts/verify_nwpu_native_c1_contract.py ...`` prints the frozen status and
  delegates to the archived module with identical arguments and exit code,
  preserving its historical ``__file__`` (the archived module resolves
  repository paths and locked-config hashes from it).

Preferred dry-run: ``python scripts/run_experiment.py dry-run --experiment det.energy.native_listwise.c1.001 --run-name nwpu_native_c1_contract_s42 --allow-frozen-reproduction``.
This wrapper will be removed once committed launchers and reports use the dispatcher.
"""

import sys as _sys
from importlib import util as _importlib_util
from pathlib import Path as _Path

_OLD_PATH = _Path(__file__).resolve()
_DESTINATION = _OLD_PATH.parent / "archive" / "historical" / "nwpu_c1" / "verify_nwpu_native_c1_contract.py"


def _load_moved_module(module_name: str):
    _spec = _importlib_util.spec_from_file_location(module_name, _DESTINATION)
    if _spec is None or _spec.loader is None:
        raise ImportError(f"cannot load archived module at {_DESTINATION}")
    _module = _importlib_util.module_from_spec(_spec)
    # Preserve the historical __file__ so Path(__file__).parents[1] inside the
    # archived module still resolves to the repository root.
    _module.__file__ = str(_OLD_PATH)
    _sys.modules[module_name] = _module
    _spec.loader.exec_module(_module)
    return _module


if __name__ == "__main__":
    print(
        "frozen: scripts/verify_nwpu_native_c1_contract.py is a frozen-reproduction entrypoint archived to "
        "scripts/archive/historical/nwpu_c1/verify_nwpu_native_c1_contract.py; delegating with identical arguments "
        "(prefer: python scripts/run_experiment.py dry-run --experiment det.energy.native_listwise.c1.001 "
        "--run-name nwpu_native_c1_contract_s42 --allow-frozen-reproduction)",
        file=_sys.stderr,
    )
    _load_moved_module("__main__")
else:
    _moved = _load_moved_module("scripts.archive.historical.nwpu_c1.verify_nwpu_native_c1_contract")
    globals().update(
        {key: value for key, value in vars(_moved).items() if not key.startswith("__")}
    )
    del _moved

del _sys, _importlib_util, _Path, _OLD_PATH, _DESTINATION, _load_moved_module
