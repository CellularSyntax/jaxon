"""Import smoke tests. The jax-only modules must import unconditionally; the
Jaxley-backed model builders are checked only when jaxley is installed."""
import importlib
import pytest


def test_import_jax_only_modules():
    for mod in ("jaxon.stim.extracellular_coupled",
                "jaxon.stim.extracellular",
                "jaxon.optim.losses"):
        importlib.import_module(mod)


def test_import_model_layer():
    pytest.importorskip("jaxley")
    for mod in ("jaxon.fibers.mrg", "jaxon.channels.mrg_axnode"):
        importlib.import_module(mod)
