"""Compatibility wrapper: implementation moved to ``scripts/analysis/analyze_dense_teacher_confounding.py``.

Task 16 wave B moved this maintained analysis tool to ``scripts/analysis/``.
This wrapper keeps the historical entrypoint working:

- importing ``scripts.analyze_dense_teacher_confounding`` re-exports the moved module's public names;
- running ``python scripts/analyze_dense_teacher_confounding.py ...`` delegates to the moved module with
  the same arguments, preserving its historical ``__file__`` (the moved module
  resolves repository paths from it).

Update call sites to ``scripts/analysis/analyze_dense_teacher_confounding.py``; this wrapper will be
removed once committed reports and tests use the new path.
"""

import sys as _sys
from importlib import util as _importlib_util
from pathlib import Path as _Path

_OLD_PATH = _Path(__file__).resolve()
_DESTINATION = _OLD_PATH.parent / "analysis" / "analyze_dense_teacher_confounding.py"


def _load_moved_module(module_name: str):
    _spec = _importlib_util.spec_from_file_location(module_name, _DESTINATION)
    if _spec is None or _spec.loader is None:
        raise ImportError(f"cannot load moved module at {_DESTINATION}")
    _module = _importlib_util.module_from_spec(_spec)
    # Preserve the historical __file__ so Path(__file__).parents[1] inside the
    # moved module still resolves to the repository root.
    _module.__file__ = str(_OLD_PATH)
    _sys.modules[module_name] = _module
    _spec.loader.exec_module(_module)
    return _module


if __name__ == "__main__":
    print(
        "deprecated: scripts/analyze_dense_teacher_confounding.py moved to scripts/analysis/analyze_dense_teacher_confounding.py; "
        "delegating with identical arguments",
        file=_sys.stderr,
    )
    _load_moved_module("__main__")
else:
    _moved = _load_moved_module("scripts.analysis.analyze_dense_teacher_confounding")
    globals().update(
        {key: value for key, value in vars(_moved).items() if not key.startswith("__")}
    )
    del _moved

del _sys, _importlib_util, _Path, _OLD_PATH, _DESTINATION, _load_moved_module
