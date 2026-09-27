"""Engineering-scale silhouettes must remain exact and strictly meshable."""

import pytest

from fdtdmesh import Simulation
from fdtdmesh.benchmarks import make_engineered_geometry
from fdtdmesh.geometry_mesher import inspect_mesh


@pytest.mark.parametrize(
    "kind",
    ["swept_aircraft", "propeller_aeroplane", "radio_telescope"],
)
def test_engineered_shape_has_valid_conformal_geometry_mesh(kind):
    recipe = make_engineered_geometry(kind)
    assert recipe.size is None
    assert recipe.bounds is not None
    sim = Simulation(fmin=0.9e9, fmax=1.1e9)
    sim.set_geometry(recipe)
    mesh = sim.apply_mesh("geometry_aware", time_limit=40)
    assert sim.geometry is recipe
    assert mesh.metadata["geometry_aware"]["status"] == "valid"
    assert not inspect_mesh(sim.computational_geometry, mesh)[0]
    assert sim.discretization.dt > 0
