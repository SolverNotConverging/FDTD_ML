"""Shared experiment identities, solver cloning and frequency-balanced errors."""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from ..api import Simulation, SolverSettings
from ..mesh import AxisCollar
from ..pml import PML
from ..result import Result, json_text
from ..strategies import mesh_id
from .features import geometry_anchors

METHOD = "mesh-study-v4"


def identity(value):
    return hashlib.sha256(json_text(value).encode()).hexdigest()


def clone(sim, *, geometry=None, pml=None, layout=None, stop=None):
    settings = asdict(sim.settings)
    settings["stop"] = stop or sim.settings.stop
    geometry = geometry or sim.geometry
    out = Simulation(
        geometry.size,
        sim.fmin,
        sim.fmax,
        settings=SolverSettings(**settings),
        angles=sim.configuration()["angles"],
        pml=pml or sim.pml,
        layout=layout or sim.layout,
    )
    out.set_geometry(geometry)
    return out


def experiment_key(sim):
    return identity(
        dict(method=METHOD, geometry=sim.geometry.as_dict(), config=sim.configuration())
    )


def errors(actual, reference):
    """Equal-frequency relative L2 on the same angular samples, with an absolute floor."""
    if not np.array_equal(actual.frequencies, reference.frequencies) or not np.array_equal(
        actual.angles, reference.angles
    ):
        raise ValueError("Reference and candidate frequency/angle samples differ")
    if not actual.converged or not reference.converged:
        raise ValueError("Cannot score an unconverged result")

    def relative(a, b):
        denominator = np.maximum(np.linalg.norm(b, axis=1), 1e-12 * np.sqrt(b.shape[1]))
        return np.linalg.norm(a - b, axis=1) / denominator

    complex_error = relative(actual.far_field, reference.far_field)
    width_error = relative(actual.width, reference.width)
    return dict(
        error=float(np.sqrt(np.mean(complex_error**2))),
        worst_frequency=float(complex_error.max()),
        per_frequency=complex_error.tolist(),
        width_error=float(np.sqrt(np.mean(width_error**2))),
    )


def run_cached(sim, directory, *, force=False):
    """Cache only converged solves by actual mesh and full physical configuration."""
    if sim.mesh is None:
        raise ValueError("Prepare a mesh before running")
    key = identity(dict(case=experiment_key(sim), mesh=mesh_id(sim.mesh)))
    path = Path(directory) / f"{key}.h5"
    if path.exists() and not force:
        result = Result.load(path)
        if not result.converged:
            raise ValueError("Cached result is unqualified")
        return result, path, True
    result = sim.solve()
    result.save(path)
    return result, path, False


def refined(sim, ppw):
    """Uniform-target, corner-aligned refinement with proportionally refined PML."""
    h = sim.wavelength / ppw
    counts = np.rint(np.array(sim.size) / h).astype(int)
    if not np.allclose(counts * h, sim.size, rtol=1e-12, atol=0):
        raise ValueError("Reference domain must align with the requested uniform spacing")
    collars = []
    for collar in (sim.pml.x, sim.pml.y):
        n = round(collar.thickness / h)
        if not np.isclose(n * h, collar.thickness, rtol=1e-12, atol=0):
            raise ValueError("Reference PML thickness must align with the spacing")
        collars.append(AxisCollar(n, collar.thickness))
    p = asdict(sim.pml)
    p.update(x=collars[0], y=collars[1])
    result = clone(sim, pml=PML(**p))
    anchors = geometry_anchors(sim.geometry)
    result.apply_mesh(
        "uniform", cells=tuple(counts), anchors=anchors, anchor_assignment="local", time_limit=30.0
    )
    return result


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json_text(value), encoding="utf-8")
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))
