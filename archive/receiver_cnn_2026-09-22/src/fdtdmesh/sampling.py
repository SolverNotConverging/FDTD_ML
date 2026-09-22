"""Physical-coordinate interpolation shared across meshes; no extrapolation."""

import numpy as np


def point_stencil(mesh, x, y):
    if not 0 <= x <= mesh.x[-1] or not 0 <= y <= mesh.y[-1]:
        raise ValueError("Point outside mesh")
    i = np.clip(np.searchsorted(mesh.x, x, side="right") - 1, 0, mesh.Nx - 1)
    j = np.clip(np.searchsorted(mesh.y, y, side="right") - 1, 0, mesh.Ny - 1)
    u = (x - mesh.x[i]) / (mesh.x[i + 1] - mesh.x[i])
    v = (y - mesh.y[j]) / (mesh.y[j + 1] - mesh.y[j])
    sites = np.array(
        [
            i * (mesh.Ny + 1) + j,
            (i + 1) * (mesh.Ny + 1) + j,
            i * (mesh.Ny + 1) + j + 1,
            (i + 1) * (mesh.Ny + 1) + j + 1,
        ],
        dtype=np.int32,
    )
    return sites, np.array([(1 - u) * (1 - v), u * (1 - v), (1 - u) * v, u * v])


def dual_areas(mesh):
    dx, dy = np.diff(mesh.x), np.diff(mesh.y)
    x = np.r_[dx[0] / 2, (dx[:-1] + dx[1:]) / 2, dx[-1] / 2]
    y = np.r_[dy[0] / 2, (dy[:-1] + dy[1:]) / 2, dy[-1] / 2]
    return x[:, None] * y[None, :]


def receiver_points(probe, mesh):
    if probe.kind == "point":
        return np.array([[probe.x, probe.y]])
    if np.ndim(probe.x) == 0:
        y = (
            np.linspace(*probe.y, probe.samples)
            if probe.samples is not None
            else mesh.y[(mesh.y >= probe.y[0]) & (mesh.y <= probe.y[1])]
        )
        return np.column_stack((np.full(len(y), probe.x), y))
    x = (
        np.linspace(*probe.x, probe.samples)
        if probe.samples is not None
        else mesh.x[(mesh.x >= probe.x[0]) & (mesh.x <= probe.x[1])]
    )
    return np.column_stack((x, np.full(len(x), probe.y)))
