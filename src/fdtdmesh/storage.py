"""Versioned atomic HDF5 archives; all operations occur after GPU execution."""

import json
import os
import tempfile
from pathlib import Path

import h5py
import numpy as np

from .geometry import Geometry
from .mesh import Mesh
from .result import Result, json_text

SCHEMA = "fdtdmesh-h5-v1"


def _write(path, kind, writer):
    path = Path(path)
    if path.suffix.lower() not in (".h5", ".hdf5"):
        raise ValueError("Use .h5 or .hdf5 for the versioned solver archive")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        with h5py.File(name, "w") as h:
            h.attrs.update(schema=SCHEMA, kind=kind, units="SI", method="tmz-conformal-ect-v1")
            writer(h)
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def _base(h, geometry, mesh, config):
    h.create_dataset("config/json", data=json_text(config))
    h.create_dataset("geometry/recipe", data=json_text(geometry.as_dict()))
    h["geometry"].attrs.update(
        geometry_id=geometry.geometry_id, units="m", representation="continuous-ordered-primitives"
    )
    if mesh is not None:
        from .strategies import mesh_id

        h.create_dataset("mesh/x", data=mesh.x)
        h.create_dataset("mesh/y", data=mesh.y)
        h.create_dataset("mesh/metadata", data=json_text(mesh.metadata))
        h["mesh"].attrs.update(mesh_id=mesh_id(mesh), units="m")


def save_simulation(sim, path):
    _write(path, "simulation", lambda h: _base(h, sim.geometry, sim.mesh, sim.configuration()))


def _validate(h, kind):
    if (
        h.attrs.get("schema") != SCHEMA
        or h.attrs.get("kind") != kind
        or h.attrs.get("units") != "SI"
    ):
        raise ValueError("Unsupported HDF5 schema, archive kind or units")


def _read_base(h):
    geometry = Geometry.from_dict(json.loads(h["geometry/recipe"].asstr()[()]))
    if geometry.geometry_id != h["geometry"].attrs["geometry_id"]:
        raise ValueError("Geometry archive identity mismatch")
    config = json.loads(h["config/json"].asstr()[()])
    if tuple(config["size"]) != geometry.size:
        raise ValueError("Geometry and simulation domain sizes differ")
    mesh = None
    if "mesh" in h:
        from .strategies import mesh_id

        mesh = Mesh(h["mesh/x"][:], h["mesh/y"][:], json.loads(h["mesh/metadata"].asstr()[()]))
        if mesh_id(mesh) != h["mesh"].attrs["mesh_id"]:
            raise ValueError("Mesh archive identity mismatch")
    return geometry, mesh, config


def load_simulation(path):
    from .api import DFTConvergence, ScatteringLayout, Simulation, SolverSettings
    from .mesh import AxisCollar
    from .pml import PML

    with h5py.File(path, "r") as h:
        _validate(h, "simulation")
        geometry, mesh, config = _read_base(h)
    settings = dict(config.pop("settings"))
    settings["stop"] = DFTConvergence(**settings["stop"])
    pml = config.pop("pml")
    pml["x"], pml["y"] = AxisCollar(**pml["x"]), AxisCollar(**pml["y"])
    layout = ScatteringLayout(**config.pop("layout"))
    sim = Simulation(**config, settings=SolverSettings(**settings), pml=PML(**pml), layout=layout)
    sim._geometry = geometry
    if mesh is not None:
        sim.apply_mesh(mesh)
    return sim


def save_result(result, path):
    def writer(h):
        _base(h, result.geometry, result.mesh, result.configuration)
        for name, array in (
            ("frequencies", result.frequencies),
            ("angles", result.angles),
            ("amplitude", result.far_field),
            ("scattering_width", result.scattering_width),
        ):
            h.create_dataset("far_field/" + name, data=array, compression="gzip", shuffle=True)
        h["far_field/amplitude"].attrs["convention"] = (
            "dimensionless S; exp(+i omega t); Hankel-2 outgoing"
        )
        h["far_field/scattering_width"].attrs["units"] = "m"
        h["far_field/frequencies"].attrs["units"] = "Hz"
        h["far_field/angles"].attrs["units"] = "rad"
        h.create_dataset("convergence/checkpoints", data=result.history, compression="gzip")
        h["convergence/checkpoints"].attrs["columns"] = (
            "step,time,error_ratio,residual,stable_checks,status,incident_peak,generation"
        )
        for i, name in enumerate(
            ("current_error_ratio", "incident_error_ratio", "incident_strength")
        ):
            h.create_dataset(
                "convergence/" + name, data=result.bin_history[:, :, i], compression="gzip"
            )
        h.create_dataset("diagnostics/json", data=result._diagnostics_json)
        h.create_dataset("source/json", data=json_text(result.diagnostics.get("source", {})))

    _write(path, "result", writer)


def load_result(path):
    with h5py.File(path, "r") as h:
        _validate(h, "result")
        geometry, mesh, config = _read_base(h)
        if mesh is None:
            raise ValueError("Result archive is missing its mesh")
        arrays = [
            h["far_field/" + name][:]
            for name in ("frequencies", "angles", "amplitude", "scattering_width")
        ]
        bins = np.stack(
            [
                h["convergence/" + name][:]
                for name in ("current_error_ratio", "incident_error_ratio", "incident_strength")
            ],
            axis=2,
        )
        return Result(
            *arrays,
            h["convergence/checkpoints"][:],
            bins,
            geometry,
            mesh,
            h["diagnostics/json"].asstr()[()],
            json_text(config),
        )
