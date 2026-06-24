import importlib


def test_round2120_to_2123_scripts_import_and_expose_main():
    expected = {
        "scripts.round2120_rlvr_fft": ("rlvr", "fft"),
        "scripts.round2121_rlvr_manifold": ("rlvr", "manifold"),
        "scripts.round2122_dpo_fft": ("dpo", "fft"),
        "scripts.round2123_dpo_manifold": ("dpo", "manifold"),
    }
    for module_name, experiment in expected.items():
        imported = importlib.import_module(module_name)
        assert hasattr(imported, "main")
        assert imported.OBJECTIVE == experiment[0]
        assert imported.VERIFIER == experiment[1]
