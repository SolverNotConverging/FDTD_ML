"""Closed-contour 2D far fields and independent PEC cylinder series.

Phasors use exp(+i omega t); outgoing waves use Hankel functions of kind 2.
The dimensionless amplitude S has sigma_2D = 4 |S|^2 / k.
"""

from dataclasses import dataclass

import numpy as np
from scipy.special import hankel2, jv

from .constants import C0, EPS0, MU0

ETA0 = np.sqrt(MU0 / EPS0)


def grid_index(lines, value):
    i = int(np.argmin(abs(lines - value)))
    if abs(lines[i] - value) > 1e-10 * lines[-1]:
        raise ValueError(f"Monitor/TFSF coordinate {value} must be anchored on the mesh")
    return i


def box_indices(mesh, box):
    x0, x1, y0, y1 = box
    indices = (
        grid_index(mesh.x, x0),
        grid_index(mesh.x, x1),
        grid_index(mesh.y, y0),
        grid_index(mesh.y, y1),
    )
    a, b, c, d = indices
    if not 1 <= a < b < mesh.Nx or not 1 <= c < d < mesh.Ny:
        raise ValueError("Box needs increasing, interior grid coordinates")
    return indices


@dataclass
class Contour:
    points: np.ndarray
    normals: np.ndarray
    ds: np.ndarray
    indices: np.ndarray
    weights: np.ndarray


def make_contour(mesh, box):
    a, b, c, d = box_indices(mesh, box)
    points, normals, ds, sites, weights = [], [], [], [], []
    nhx = (mesh.Nx + 1) * mesh.Ny
    # Ht = nx Hy - ny Hx. Interpolate H to the E-node contour in physical space.
    for fixed_axis, fixed, lo, hi, normal in (
        (0, a, c, d, (-1, 0)),
        (0, b, c, d, (1, 0)),
        (1, c, a, b, (0, -1)),
        (1, d, a, b, (0, 1)),
    ):
        along = mesh.y if fixed_axis == 0 else mesh.x
        steps = np.diff(along[lo : hi + 1])
        quadrature = np.r_[steps[0] / 2, (steps[:-1] + steps[1:]) / 2, steps[-1] / 2]
        for u, weight in zip(range(lo, hi + 1), quadrature):
            i, j = (fixed, u) if fixed_axis == 0 else (u, fixed)
            points.append((mesh.x[i], mesh.y[j]))
            normals.append(normal)
            ds.append(weight)
            if fixed_axis == 0:
                hm, hp = nhx + (i - 1) * (mesh.Ny + 1) + j, nhx + i * (mesh.Ny + 1) + j
                left, right = mesh.x[i] - mesh.x[i - 1], mesh.x[i + 1] - mesh.x[i]
                sign = normal[0]
            else:
                hm, hp = i * mesh.Ny + j - 1, i * mesh.Ny + j
                left, right = mesh.y[j] - mesh.y[j - 1], mesh.y[j + 1] - mesh.y[j]
                sign = -normal[1]
            sites.append((i * (mesh.Ny + 1) + j, hm, hp))
            weights.append((sign * right / (left + right), sign * left / (left + right)))
    return Contour(
        np.asarray(points),
        np.asarray(normals),
        np.asarray(ds),
        np.asarray(sites, np.int32),
        np.asarray(weights),
    )


def far_field(contour, ez, ht, frequency, angles, incident, origin):
    angles = np.asarray(angles, float)
    if angles.ndim != 1 or not angles.size or not np.isfinite(angles).all():
        raise ValueError("Angles must be a finite nonempty vector in radians")
    if not np.isfinite(incident) or abs(incident) < 1e-20:
        raise ValueError("Incident spectrum is zero or nonfinite")
    k = 2 * np.pi * frequency / C0
    direction = np.c_[np.cos(angles), np.sin(angles)]
    phase = np.exp(1j * k * direction @ (contour.points - np.asarray(origin)).T)
    integrand = (direction @ contour.normals.T) * ez[None, :] - ETA0 * ht[None, :]
    amplitude = k / 4 * np.sum(phase * integrand * contour.ds, axis=1) / incident
    return amplitude, 4 / k * abs(amplitude) ** 2


def cylinder_amplitude(radius, frequency, angles, *, incidence=0.0, terms=None):
    if not np.isfinite([radius, frequency]).all() or min(radius, frequency) <= 0:
        raise ValueError("Radius and frequency must be positive")
    ka = 2 * np.pi * frequency / C0 * radius
    terms = int(np.ceil(ka + 4 * ka ** (1 / 3) + 15)) if terms is None else int(terms)
    if terms < 1:
        raise ValueError("Series requires at least one term")
    n = np.arange(terms + 1)
    ratio = jv(n, ka) / hankel2(n, ka)
    theta = np.asarray(angles) - incidence
    return -ratio[0] - 2 * np.sum(ratio[1:, None] * np.cos(n[1:, None] * theta), axis=0)


def pattern_error(actual, reference):
    actual, reference = np.asarray(actual), np.asarray(reference)
    if (
        actual.shape != reference.shape
        or not np.isfinite(actual).all()
        or not np.isfinite(reference).all()
    ):
        raise ValueError("Comparable finite patterns required")
    denominator = np.linalg.norm(reference)
    if denominator == 0:
        raise ValueError("Use an absolute error for a zero-scattering reference")
    return float(np.linalg.norm(actual - reference) / denominator)
