"""Model-builder smoke test (requires jaxley)."""
import pytest

jaxley = pytest.importorskip("jaxley")


def test_build_mrg():
    from jaxon.fibers import mrg
    cell, geom = mrg.build_mrg(diameter=10.0, temperature=37.0)
    assert cell is not None and geom is not None
    centers = mrg.section_centers_um(geom)
    assert len(centers) > 0
