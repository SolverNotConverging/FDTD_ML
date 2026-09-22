"""Actual uniform-grid baseline with nearest-line PEC geometry discretization."""

from dataclasses import replace

import numpy as np

from fdtdmesh.mesh import cell_count

BASELINE_POLICY = "uniform_snapped_pec_pml_and_quasi_uniform_v2"


def uniform_scene(spec, budget):
    # A true uniform candidate discretizes the absorber as well as PEC. References
    # keep the strict fixed-thickness rule in SceneSpec.build(reference=True).
    thicknesses, pml_mapping = [], {}
    for axis, count, length, thickness in zip("xy", budget, spec.domain, spec.pml["thickness"]):
        count = cell_count(count)
        requested_cells = count * thickness / length
        if abs(requested_cells - round(requested_cells)) <= 1e-9:
            cells, actual = int(round(requested_cells)), thickness
        else:
            # Half-cell ties choose the thinner collar, toward the outer boundary.
            cells = int(np.floor(requested_cells + 0.5 - 1e-12))
            actual = cells * length / count
        if cells < 1 or 2 * cells >= count:
            raise ValueError("Uniform PML rounding must retain absorber and interior cells")
        thicknesses.append(actual)
        pml_mapping[axis] = dict(requested_thickness=thickness, actual_thickness=actual,
                                 cells=cells, interface_shift=actual - thickness)
    discretized = replace(spec, pml=dict(spec.pml, thickness=thicknesses))
    scene = discretized.build(budget, reference=True)
    scene.x_anchors.clear()
    scene.y_anchors.clear()
    mesh = scene.mesh_uniform()
    changes = []
    primitives = []

    def snap(value, nodes):
        # Midpoint ties go toward the lower node (within roundoff tolerance).
        distances = abs(nodes - value)
        indices = np.flatnonzero(distances <= distances.min() + 1e-14 * nodes[-1])
        return float(nodes[indices[0]])

    for index, (kind, material, data) in enumerate(scene.primitives):
        if material.kind != "PEC":
            primitives.append((kind, material, data))
            continue
        if kind == "polygon":
            xs, ys = np.unique(data[:, 0]), np.unique(data[:, 1])
            if len(data) != 4 or len(xs) != 2 or len(ys) != 2:
                raise ValueError("Uniform PEC snapping supports rectangles and axis-aligned wires")
            if any(a[0] != b[0] and a[1] != b[1] for a, b in zip(data, np.roll(data, -1, axis=0))):
                raise ValueError("Uniform PEC snapping requires axis-aligned rectangle edges")
            snapped = np.array([[snap(x, mesh.x), snap(y, mesh.y)] for x, y in data])
            if np.ptp(snapped[:, 0]) == 0 or np.ptp(snapped[:, 1]) == 0:
                raise ValueError(
                    "PEC rectangle collapses under nearest-line snapping at this budget"
                )
            original, mapped = data.tolist(), snapped.tolist()
        elif kind == "line":
            if np.ndim(data.x) == 0:
                snapped = replace(
                    data, x=snap(data.x, mesh.x), y=tuple(snap(v, mesh.y) for v in data.y)
                )
                collapsed = snapped.y[0] == snapped.y[1]
            else:
                snapped = replace(
                    data, y=snap(data.y, mesh.y), x=tuple(snap(v, mesh.x) for v in data.x)
                )
                collapsed = snapped.x[0] == snapped.x[1]
            if collapsed:
                raise ValueError("PEC wire collapses under nearest-line snapping at this budget")
            original, mapped = dict(x=data.x, y=data.y), dict(x=snapped.x, y=snapped.y)
        else:
            raise ValueError("Uniform PEC snapping supports rectangles and axis-aligned wires")
        primitives.append((kind, material, snapped))
        changes.append(dict(primitive=index, kind=kind, original=original, snapped=mapped))
    scene.primitives = primitives
    # Probes retain their physical coordinates. Integrated point currents and
    # receivers use the existing bilinear stencil; they do not create mesh lines.
    scene.pml.validate_scene(scene)
    scene.mesh.metadata.update(
        baseline_policy=BASELINE_POLICY,
        strategy="uniform",
        pec_snapping=changes,
        pml_snapping=pml_mapping,
        probe_mapping="physical_bilinear",
    )
    return scene
