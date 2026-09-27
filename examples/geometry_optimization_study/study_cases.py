"""Build comparable study cases and replay the screened conformal seed mesh."""

from __future__ import annotations

import json
from pathlib import Path

from fdtdmesh import Simulation
from fdtdmesh.benchmarks import make_engineered_geometry
from fdtdmesh.benchmarks.shapes import SHAPES, make_geometry
from fdtdmesh.constants import C0

ROOT = Path("artifacts/geometry_optimization_study")
ANGLE_SCREEN = ROOT / "engineered_angle_feasibility.jsonl"


def case_directory(shape: str, angle: float, scale: float) -> Path:
    return ROOT / f"{shape}_{angle:g}_{scale:g}"


def make_simulation(shape: str, angle: float, scale: float) -> Simulation:
    sim = Simulation(fmin=0.9e9, fmax=1.1e9)
    geometry = (
        make_geometry(shape, size=(2.0, 2.0), scale=scale, incidence_deg=angle)
        if shape in SHAPES
        else make_engineered_geometry(shape, scale=scale, incidence_deg=angle)
    )
    sim.set_geometry(geometry)
    return sim


def screened_target_ppw(shape: str, angle: float, scale: float) -> int | None:
    if not ANGLE_SCREEN.exists():
        return None
    records = [json.loads(line) for line in ANGLE_SCREEN.read_text(encoding="utf8").splitlines()]
    matches = [
        row
        for row in records
        if row["shape"] == shape and row["angle"] == angle and row["scale"] == scale
    ]
    match = next(
        (row for row in reversed(matches) if row["status"] == "valid"),
        None,
    )
    if match is None:
        if matches:
            raise ValueError(f"No screened strict-conformal mesh for {shape} {angle:g}° {scale:g}λ")
        return None
    return int(match["selected_target_ppw"])


def apply_seed_mesh(sim: Simulation, shape: str, angle: float, scale: float):
    ppw = screened_target_ppw(shape, angle, scale) if shape not in SHAPES else None
    if ppw is None:
        return sim.apply_mesh("geometry_aware", time_limit=40, max_cells=(512, 512))
    return sim.apply_mesh(
        "geometry_aware",
        target_spacing=C0 / (sim.fmax * ppw),
        time_limit=45,
        max_cells=(768, 768),
        max_passes=30,
    )
