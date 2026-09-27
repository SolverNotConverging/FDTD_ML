"""Shared experiment identities, solver cloning and frequency-balanced errors."""

import hashlib
import json
import uuid
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from ..api import Simulation, SolverSettings
from ..mesh import AxisCollar
from ..pml import PML
from ..result import Result, json_text
from .features import geometry_anchors, unresolved_edge_anchors

METHOD = "mesh-study-v7"


def identity(value):
    return hashlib.sha256(json_text(value).encode()).hexdigest()


def clone(sim, *, geometry=None, pml=None, layout=None, stop=None):
    settings = asdict(sim.settings)
    settings["stop"] = stop or sim.settings.stop
    if sim._automatic_domain and geometry is None and pml is None and layout is None:
        out = Simulation(
            fmin=sim.fmin,
            fmax=sim.fmax,
            domain=sim._domain_policy,
            settings=SolverSettings(**settings),
            angles=sim.configuration()["angles"],
        )
        out.set_geometry(sim.geometry)
        return out
    geometry = geometry or sim.computational_geometry
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


def simulation_description(sim):
    """Complete resolved physical inputs, independent of a directory name."""
    root = Path(__file__).parents[1]
    sources = {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.suffix in (".py", ".pyx", ".cu", ".h", ".pyd", ".so")
    }
    return dict(
        method=METHOD,
        implementation=identity(sources),
        geometry=sim.geometry.as_dict(),
        configuration=sim.configuration(),
        source=asdict(sim.source),
        frequencies=sim.frequencies.tolist(),
    )


def experiment_key(sim):
    return identity(simulation_description(sim))


def experiment_directory(root, description):
    """Reuse only an exact JSON match; preserve all incompatible/legacy data."""
    root = Path(root)
    canonical = json.loads(json_text(description))
    key = identity(canonical)
    candidates = [root, root / key]
    if root.exists():
        candidates.extend(sorted(root.glob(key + "-*")))
    for candidate in candidates:
        manifest = candidate / "experiment.json"
        if manifest.exists():
            try:
                if read_json(manifest) == canonical:
                    return candidate
            except (ValueError, OSError):
                pass
    directory = root / key
    if directory.exists():
        directory = root / f"{key}-{uuid.uuid4().hex}"
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "experiment.json", canonical)
    return directory


def archive_directory(directory, filename):
    """Allow loading an unambiguous root, or an explicit experiment directory."""
    directory = Path(directory)
    if (directory / filename).exists():
        return directory
    matches = sorted(p.parent for p in directory.glob(f"*/{filename}"))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(
            "Multiple experiments exist; use the returned object's .directory to select one"
        )
    return directory


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
    description = dict(
        schema=1,
        kind="solver-result",
        simulation=simulation_description(sim),
        mesh=dict(x=sim.mesh.x.tolist(), y=sim.mesh.y.tolist()),
    )
    directory = experiment_directory(directory, description)
    path = directory / "result.h5"
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
    collars = []
    for collar in (sim.pml.x, sim.pml.y):
        n = round(collar.thickness / h)
        collars.append(AxisCollar(max(1, n), collar.thickness))
    p = asdict(sim.pml)
    p.update(x=collars[0], y=collars[1])
    # Numerical references refine the exterior too, preserving physical boxes.
    result = clone(
        sim,
        pml=PML(**p),
        layout=replace(sim.layout, exterior_cells=None, scatterer_margin_cells=None),
    )
    anchors = geometry_anchors(result.geometry)
    from ..solver.conformal import UnresolvedGeometryError
    from ..strategies import generate_mesh

    for attempt in range(9):
        mesh = generate_mesh(
            result.geometry,
            result.layout,
            result.pml,
            tuple(counts),
            "uniform",
            anchors=anchors,
            anchor_assignment="local",
            time_limit=30.0,
        )
        additions = unresolved_edge_anchors(result.geometry, mesh)
        if not any(len(values) for values in additions):
            mesh.metadata.update(reference_anchor_passes=attempt, strategy="uniform")
            result.apply_mesh(mesh)
            break
        if attempt == 8:
            raise UnresolvedGeometryError(
                "Reference edge alignment failed after eight anchor refinements"
            )
        anchors = tuple(np.unique(np.r_[old, new]) for old, new in zip(anchors, additions))
    return result


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json.loads(json_text(value)), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))
