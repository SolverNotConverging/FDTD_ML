"""Mesh proposal strategies; all generated grids pass through the same projector."""

import hashlib
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d

from .constants import C0
from .mesh import Mesh, MeshInfeasibleError, density_mesh


def mesh_id(mesh):
    return hashlib.sha256(mesh.x.tobytes() + mesh.y.tobytes()).hexdigest()


def deterministic_density(geometry, bins=128):
    mask = geometry.rasterize((bins, bins))[0]
    # Project resolved material changes, not primitive edges hidden by overlaps.
    edge_x = np.abs(np.diff(mask, axis=1, prepend=mask[:, :1])).mean(axis=0)
    edge_y = np.abs(np.diff(mask, axis=0, prepend=mask[:1, :])).mean(axis=1)
    result = []
    for edges, occupied in ((edge_x, mask.mean(axis=0)), (edge_y, mask.mean(axis=1))):
        activity = gaussian_filter1d(edges, 2.0)
        activity /= max(float(activity.max()), 1e-12)
        result.append(1 + 3 * activity + occupied)
    return tuple(result)


@lru_cache(maxsize=4)
def _session(path, checksum):
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("CNN inference requires the 'cnn' extra: uv sync --extra cnn") from exc
    return ort.InferenceSession(path, providers=["CPUExecutionProvider"])


def cnn_density(geometry, fmin, fmax, checkpoint):
    """Portable ONNX checkpoint bundle; no pickle or silent heuristic fallback."""
    root = Path(checkpoint)
    manifest = json.loads((root / "manifest.json").read_text())
    if (
        manifest.get("schema") != "fdtdmesh-cnn-v1"
        or manifest.get("method") != "tmz-conformal-ect-v1"
    ):
        raise ValueError("Checkpoint schema/physics is incompatible with this solver")
    if (
        manifest.get("normalization") != "unit-domain"
        or manifest.get("outputs") != "positive-axis-density"
    ):
        raise ValueError("Unsupported checkpoint feature/output normalization")
    model = root / "model.onnx"
    checksum = hashlib.sha256(model.read_bytes()).hexdigest()
    if checksum != manifest.get("sha256"):
        raise ValueError("Checkpoint checksum does not match model.onnx")
    channels = tuple(manifest["channels"])
    features = geometry.rasterize(tuple(manifest["shape"]), channels=channels)[None]
    wavelength = C0 / ((fmin + fmax) / 2)
    physics = np.array(
        [
            [
                geometry.size[0] / wavelength,
                geometry.size[1] / wavelength,
                fmin / ((fmin + fmax) / 2),
                fmax / ((fmin + fmax) / 2),
            ]
        ],
        np.float32,
    )
    runtime = _session(str(model.resolve()), checksum)
    prediction = runtime.run(["rho_x", "rho_y"], {"geometry": features, "physics": physics})
    axes = []
    for values in prediction:
        a = np.asarray(values)
        if a.ndim != 2 or a.shape[0] != 1 or a.shape[1] < 2:
            raise ValueError("CNN outputs must be (1, bins) axis-density vectors")
        axes.append(a[0].astype(float))
    return tuple(axes), dict(
        checkpoint_sha256=checksum,
        checkpoint_schema=manifest["schema"],
        architecture=manifest.get("architecture", "onnx"),
    )


def exterior_lines(length, count, collar, contour, tfsf, allocation):
    """Exact exterior coordinates and cell counts; interior remains movable."""
    if allocation is None:
        return {}
    outer, inner = allocation
    if count - 2 * (collar.cells + outer + inner) <= 4:
        raise MeshInfeasibleError(
            "Cell budget leaves too few cells inside TFSF after exterior allocation"
        )
    p = collar.thickness
    ca, cb = contour
    a, b = tfsf
    if not p < ca < a < b < cb < length - p:
        raise MeshInfeasibleError("Exterior allocation requires nested PML, contour and TFSF")
    fixed = {}
    for index, low, high, cells in (
        (collar.cells, p, ca, outer),
        (collar.cells + outer, ca, a, inner),
        (count - collar.cells - outer - inner, b, cb, inner),
        (count - collar.cells - outer, cb, length - p, outer),
    ):
        fixed.update({index + j: float(v) for j, v in enumerate(np.linspace(low, high, cells + 1))})
    return fixed


def generate_mesh(
    geometry,
    layout,
    pml,
    cells,
    strategy,
    *,
    density=None,
    checkpoint=None,
    strict=False,
    constraints=None,
    anchors=None,
    anchor_assignment="joint",
    time_limit=30.0,
):
    from .mesh import cell_count

    if len(cells) != 2:
        raise ValueError("cells must be (Nx, Ny), including PML")
    nx, ny = (cell_count(n) for n in cells)
    if anchor_assignment not in ("joint", "fixed", "local"):
        raise ValueError("anchor_assignment must be joint, fixed or local")
    if density is not None and strategy != "density":
        raise ValueError("density is only accepted by the density strategy")
    if checkpoint is not None and strategy != "cnn":
        raise ValueError("checkpoint is only accepted by the cnn strategy")
    if strict and strategy != "uniform":
        raise ValueError("strict is only accepted by the uniform strategy")
    ax = np.unique(
        [*layout.tfsf_box[:2], *layout.contour_box[:2], layout.source_x, layout.origin[0]]
    )
    ay = np.unique([*layout.tfsf_box[2:], *layout.contour_box[2:]])
    if anchors is not None:
        if len(anchors) != 2:
            raise ValueError("anchors must contain x and y coordinate vectors")
        merged = []
        for base, extra, length in zip((ax, ay), anchors, geometry.size):
            values = list(base)
            for value in np.asarray(extra, float):
                if not np.isfinite(value):
                    raise ValueError("Anchors must be finite")
                if not any(abs(value - v) <= 64 * np.finfo(float).eps * length for v in values):
                    values.append(value)
            merged.append(np.sort(values))
        ax, ay = merged
    fixed = [
        exterior_lines(length, n, collar, contour, box, layout.exterior_cells)
        for length, n, collar, contour, box in zip(
            geometry.size,
            (nx, ny),
            (pml.x, pml.y),
            (layout.contour_box[:2], layout.contour_box[2:]),
            (layout.tfsf_box[:2], layout.tfsf_box[2:]),
        )
    ]
    if layout.scatterer_margin_cells is not None:
        bounds = geometry.bounds
        if bounds is None:
            raise MeshInfeasibleError("Scatterer margins require PEC geometry")
        margin = layout.scatterer_margin_cells
        outer = sum(layout.exterior_cells)
        for indices, n, collar, box, occupied in zip(
            fixed,
            (nx, ny),
            (pml.x, pml.y),
            (layout.tfsf_box[:2], layout.tfsf_box[2:]),
            (bounds[:2], bounds[2:]),
        ):
            left, right = collar.cells + outer, n - collar.cells - outer
            if right - left <= 2 * margin:
                raise MeshInfeasibleError(
                    "Cell budget leaves no cells inside scatterer bounding box"
                )
            if not box[0] < occupied[0] < occupied[1] < box[1]:
                raise MeshInfeasibleError(
                    "Geometry no longer fits the fitted TFSF box; call fit_domain again"
                )
            indices.update(
                {
                    left + j: float(v)
                    for j, v in enumerate(np.linspace(box[0], occupied[0], margin + 1))
                }
            )
            indices.update(
                {
                    right - margin + j: float(v)
                    for j, v in enumerate(np.linspace(occupied[1], box[1], margin + 1))
                }
            )
    # Preserve exact source/layout anchors that differ from linspace by roundoff.
    for indices, values, length in zip(fixed, (ax, ay), geometry.size):
        for v in values:
            matches = [
                i for i, x in indices.items() if abs(x - v) <= 64 * np.finfo(float).eps * length
            ]
            if matches:
                indices[matches[0]] = float(v)
    meta = dict(
        strategy=strategy,
        strategy_version=2,
        anchor_assignment=anchor_assignment,
        exterior_cells=layout.exterior_cells,
        scatterer_margin_cells=layout.scatterer_margin_cells,
    )
    if strategy == "uniform":
        # Fast exact-uniform path. Never alter a user-supplied coordinate mesh.
        axes = []
        for length, n, collar, anchors, reserved in zip(
            geometry.size, (nx, ny), (pml.x, pml.y), (ax, ay), fixed
        ):
            lines = np.linspace(0, length, n + 1)
            valid = True
            for index, value in collar.fixed_lines(length, n).items():
                valid &= abs(lines[index] - value) < 1e-12 * length
                lines[index] = value
            for value in anchors:
                i = np.argmin(abs(lines - value))
                valid &= abs(lines[i] - value) < 1e-12 * length
                lines[i] = value
            for index, value in reserved.items():
                valid &= abs(lines[index] - value) < 1e-12 * length
                lines[index] = value
            if not valid:
                break
            axes.append(lines)
        if len(axes) == 2:
            mesh = Mesh(*axes, metadata=meta)
            if constraints is not None:
                from .mesh import validate_spacing

                for lines in axes:
                    validate_spacing(lines, constraints)
            return mesh
        if strict:
            raise MeshInfeasibleError(
                "Exact uniform mesh is incompatible with fixed collars/anchors at this budget"
            )
        density = (np.ones(64), np.ones(64))
    elif strategy == "deterministic":
        density = deterministic_density(geometry)
    elif strategy == "cnn":
        raise RuntimeError("CNN prediction must be resolved by Simulation before projection")
    elif strategy != "density":
        raise ValueError(
            "Mesh strategy must be uniform, deterministic, cnn, density, or an explicit Mesh"
        )
    if density is None or len(density) != 2:
        raise ValueError("density requires two positive axis vectors")
    for a in density:
        a = np.asarray(a)
        if a.ndim != 1 or not len(a) or not np.isfinite(a).all() or np.any(a <= 0):
            raise ValueError("Density vectors must be finite and strictly positive")
    for length, n, collar, values, indices in zip(
        geometry.size, (nx, ny), (pml.x, pml.y), (ax, ay), fixed
    ):
        if anchor_assignment == "fixed":
            known = {**collar.fixed_lines(length, n), **indices}
            ordered = sorted(known.items())
            for (lo, left), (hi, right) in zip(ordered, ordered[1:]):
                interior = [v for v in values if left < v < right]
                if len(interior) > hi - lo - 1:
                    raise MeshInfeasibleError("Too many fixed anchors for interval budget")
                previous = lo
                for k, v in enumerate(interior):
                    target = round(np.interp(v, [left, right], [lo, hi]))
                    index = max(previous + 1, min(target, hi - (len(interior) - k)))
                    indices[index] = v
                    previous = index
    meta["anchor_assignment"] = anchor_assignment
    mesh = density_mesh(
        *geometry.size,
        nx,
        ny,
        *density,
        x_anchors=ax,
        y_anchors=ay,
        x_collar=pml.x,
        y_collar=pml.y,
        x_constraints=constraints,
        y_constraints=constraints,
        time_limit=time_limit,
        x_fixed_indices=fixed[0] or None,
        y_fixed_indices=fixed[1] or None,
        anchor_window=2 if anchor_assignment == "local" else None,
    )
    return Mesh(mesh.x, mesh.y, {**mesh.metadata, **meta})
