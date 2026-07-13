"""Lazy-facade contract for ``energy_transport`` (refactor plan Task 14 step 4).

The top-level facade loads public names lazily (PEP 562) from a static
name -> defining-module mapping. This test pins:

1. The documented ``__all__`` (143 names, order preserved) and ``dir()``.
2. The facade source has no eager in-repo imports; ``__getattr__``/``__dir__``
   are the only access path.
3. The static mapping covers ``__all__`` exactly once and every name resolves
   to the same object as the defining subpackage module.
4. ``from energy_transport import X`` keeps working; unknown names raise
   ``AttributeError`` (so ``from energy_transport import <submodule>`` can
   still fall back to the submodule import).
5. Import smoke in a fresh subprocess with the optional legacy dependency
   ``sklearn`` blocked: the facade imports and one representative name per
   subpackage resolves. (Investigation: no facade-reachable module imports
   sklearn or any other not-installed optional dependency — only
   ``torchvision.ops``, which is a hard installed dependency — so the block
   proves the facade is insulated rather than documenting a real consumer.)
"""

from __future__ import annotations

import ast
import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

PACKAGE_ROOT = "spectral_detection_posttrain.methods.energy_transport"
REPO_ROOT = Path(__file__).resolve().parents[2]
FACADE_PATH = (
    REPO_ROOT / "spectral_detection_posttrain" / "methods" / "energy_transport" / "__init__.py"
)


def test_facade_all_has_143_unique_names_and_dir_matches():
    facade = importlib.import_module(PACKAGE_ROOT)
    assert len(facade.__all__) == 143
    assert len(set(facade.__all__)) == 143
    assert dir(facade) == sorted(facade.__all__)


def test_facade_source_has_no_eager_in_repo_imports():
    tree = ast.parse(FACADE_PATH.read_text(encoding="utf-8"), filename=str(FACADE_PATH))
    offenders = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            offenders.extend(
                alias.name
                for alias in node.names
                if alias.name.startswith("spectral_detection_posttrain")
            )
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.startswith("spectral_detection_posttrain"):
                offenders.append(node.module)
    assert offenders == [], f"eager in-repo imports in lazy facade: {offenders}"
    functions = {
        statement.name for statement in tree.body if isinstance(statement, ast.FunctionDef)
    }
    assert {"__getattr__", "__dir__"} <= functions


def test_symbol_mapping_covers_all_exactly_once_and_resolves():
    facade = importlib.import_module(PACKAGE_ROOT)
    mapping = facade._SYMBOL_MODULES
    assert set(mapping) == set(facade.__all__)
    assert len(mapping) == len(facade.__all__) == 143
    modules: dict[str, object] = {}
    for name in facade.__all__:
        module_path = mapping[name]
        if module_path not in modules:
            modules[module_path] = importlib.import_module(module_path)
        assert getattr(modules[module_path], name) is getattr(facade, name), name


def test_from_import_form_resolves_and_caches():
    facade = importlib.import_module(PACKAGE_ROOT)
    namespace: dict = {}
    exec(
        f"from {PACKAGE_ROOT} import ActionLocalTransportHead, roi_dual_energy",
        namespace,
    )
    assert namespace["ActionLocalTransportHead"] is facade.ActionLocalTransportHead
    assert namespace["roi_dual_energy"] is facade.roi_dual_energy
    assert "ActionLocalTransportHead" in vars(facade), "resolved name must cache in globals"


def test_unknown_attribute_raises_attribute_error():
    facade = importlib.import_module(PACKAGE_ROOT)
    assert not hasattr(facade, "definitely_not_a_symbol")
    with pytest.raises(AttributeError):
        getattr(facade, "definitely_not_a_symbol")


def test_facade_import_smoke_with_sklearn_blocked(tmp_path):
    """Fresh interpreter, sklearn blocked: facade imports, all five subpackages resolve."""
    script = (
        "import sys\n"
        "sys.modules['sklearn'] = None  # simulate absent optional legacy dependency\n"
        "import spectral_detection_posttrain.methods.energy_transport as facade\n"
        "loaded = [m for m in sys.modules if m.startswith(facade.__name__ + '.')]\n"
        "assert loaded == [], f'eager subpackage imports: {loaded}'\n"
        "assert len(facade.__all__) == 143\n"
        "assert dir(facade) == sorted(facade.__all__)\n"
        "from spectral_detection_posttrain.methods.energy_transport import (\n"
        "    ActionLocalTransportHead,\n"  # action
        "    native_action_nms_topology,\n"  # native
        "    NMSAwareSetPolicyHead,\n"  # policy
        "    DenseSetEnergyEndpoint,\n"  # endpoint
        "    roi_dual_energy,\n"  # diagnostics
        ")\n"
        "assert facade.ActionLocalTransportHead is ActionLocalTransportHead\n"
        "assert facade.roi_dual_energy is roi_dual_energy\n"
        "import torch\n"
        "assert not torch.cuda.is_initialized()\n"
        "assert sys.modules['sklearn'] is None\n"
        "try:\n"
        "    import sklearn\n"
        "    raise SystemExit('sklearn block was not effective')\n"
        "except ImportError:\n"
        "    pass\n"
        "print('OK')\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(REPO_ROOT), env.get("PYTHONPATH", "")) if part
    )
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        capture_output=True,
        text=True,
        env=env,
        timeout=180,
    )
    assert result.returncode == 0, (
        "facade import smoke with sklearn blocked failed:\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert result.stdout.strip().endswith("OK")
