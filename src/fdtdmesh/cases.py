"""Small laptop benchmarks, all lengths scaled by the evaluation wavelength."""

import numpy as np

from .constants import C0
from .mesh import AxisCollar, AxisConstraints, Mesh, density_mesh
from .pml import PML
from .scene import Scene2D
from .simulation import Convergence, ScatteringCase


def canonical_case(
    kind="cylinder",
    *,
    ppw=24,
    nonuniform=False,
    frequency=1e9,
    shift=(0.0, 0.0),
    radius=0.47,
    contour_offset=1.0,
    pml_cells=None,
    convergence=None,
    frequencies=(),
):
    if isinstance(ppw, bool) or ppw < 8 or ppw % 8:
        raise ValueError("ppw must be a positive multiple of eight, at least eight")
    lam = C0 / frequency
    length = 6 * lam
    scene = Scene2D(length, length)
    center = (3 * lam + shift[0] * lam, 3 * lam + shift[1] * lam)
    x, y = center
    if kind == "cylinder":
        scene.add_circle(center, radius * lam)
    elif kind == "rectangle":
        scene.add_rectangle((x - 0.43 * lam, x + 0.43 * lam), (y - 0.37 * lam, y + 0.37 * lam))
    elif kind == "pair":
        scene.add_circle((x - 0.4 * lam, y), 0.27 * lam)
        scene.add_circle((x + 0.4 * lam, y), 0.27 * lam)
    elif kind == "slot":
        scene.add_rectangle((x - 0.5 * lam, x + 0.5 * lam), (y - 0.45 * lam, y + 0.45 * lam))
        scene.add_rectangle(
            (x - 0.13 * lam, x + 0.13 * lam), (y - 0.15 * lam, y + 0.5 * lam), pec=False
        )
    elif kind != "empty":
        raise ValueError("Unknown canonical geometry")
    collar = AxisCollar(ppw // 2 if pml_cells is None else pml_cells, 0.5 * lam)
    pml = PML(collar, collar)
    case = ScatteringCase(
        scene,
        frequency,
        tuple(lam * np.array([1.5, 4.5, 1.5, 4.5])),
        tuple(
            lam * np.array([contour_offset, 6 - contour_offset, contour_offset, 6 - contour_offset])
        ),
        pml,
        0.75 * lam,
        (3 * lam, 3 * lam),
        frequencies=frequencies,
        convergence=convergence or Convergence(),
    )
    n = 6 * ppw
    anchors = np.unique(
        lam * np.array([0.5, 0.75, contour_offset, 1.5, 3, 4.5, 6 - contour_offset, 5.5])
    )
    if nonuniform or pml_cells is not None:
        u = (np.arange(96) + 0.5) / 96
        rho = 1 + 1.0 * np.exp(-(((u - 0.5) / 0.18) ** 2)) if nonuniform else np.ones_like(u)
        mesh = density_mesh(
            length,
            length,
            n,
            n,
            rho,
            rho,
            x_anchors=anchors,
            y_anchors=anchors,
            x_collar=collar,
            y_collar=collar,
            x_constraints=AxisConstraints(min_spacing=lam / (4 * ppw)),
            y_constraints=AxisConstraints(min_spacing=lam / (4 * ppw)),
        )
    else:
        # Snap anchors to exact shared coordinates, including fixed PML lines.
        axis = np.linspace(0, length, n + 1)
        for v in anchors:
            i = np.argmin(abs(axis - v))
            if abs(axis[i] - v) > 1e-10 * length:
                raise ValueError("Requested contour incompatible with uniform grid")
            axis[i] = v
        for i, v in collar.fixed_lines(length, n).items():
            axis[i] = v
        mesh = Mesh(axis, axis)
    return case, mesh
