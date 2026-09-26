"""Streaming stagger-aware surface DFT and 2D TMz near-to-far transformation.

Fourier convention: integral f(t) exp(+i omega t) dt, with exp(-i omega t)
phasors. Far field is Ez_s ~ A(phi) exp(i k r)/sqrt(r). Scattering width is
2*pi*|A/E_inc|^2 [metres]; angular differential cross section is width/(2*pi).
"""

import numpy as np

from .constants import C0, MU0


def field_point_stencil(grid, points):
    """Return bilinear interpolation stencils for fixed physical field points."""
    points = np.asarray(points, dtype=float)
    if (
        points.ndim != 2
        or points.shape[1] != 2
        or not len(points)
        or not np.isfinite(points).all()
        or np.any(points[:, 0] < grid.x[0])
        or np.any(points[:, 0] > grid.x[-1])
        or np.any(points[:, 1] < grid.y[0])
        or np.any(points[:, 1] > grid.y[-1])
    ):
        raise ValueError("Field sample points must be finite [point, x/y] coordinates in the grid")
    i0 = np.clip(np.searchsorted(grid.x, points[:, 0], side="right") - 1, 0, len(grid.x) - 2)
    j0 = np.clip(np.searchsorted(grid.y, points[:, 1], side="right") - 1, 0, len(grid.y) - 2)
    wx = (points[:, 0] - grid.x[i0]) / (grid.x[i0 + 1] - grid.x[i0])
    wy = (points[:, 1] - grid.y[j0]) / (grid.y[j0 + 1] - grid.y[j0])
    return i0, j0, wx, wy


def interpolate_field_points(values, stencil):
    """Bilinearly interpolate an x/y nodal field using a prepared stencil."""
    i0, j0, wx, wy = stencil
    return (
        values[i0, j0] * (1 - wx) * (1 - wy)
        + values[i0 + 1, j0] * wx * (1 - wy)
        + values[i0, j0 + 1] * (1 - wx) * wy
        + values[i0 + 1, j0 + 1] * wx * wy
    )


class Contour:
    """Closed rectangle on existing mesh nodes; does not insert mesh lines."""

    def __init__(self, grid, bounds):
        x0, x1, y0, y1 = bounds
        self.i0, self.i1 = [int(np.argmin(abs(grid.x - v))) for v in (x0, x1)]
        self.j0, self.j1 = [int(np.argmin(abs(grid.y - v))) for v in (y0, y1)]
        i0, i1, j0, j1 = self.i0, self.i1, self.j0, self.j1
        if not (1 <= i0 < i1 < len(grid.x) - 1 and 1 <= j0 < j1 < len(grid.y) - 1):
            raise ValueError("Monitor must be a nonempty closed interior contour")
        self.grid = grid
        self.bounds = grid.x[i0], grid.x[i1], grid.y[j0], grid.y[j1]
        xc, yc = grid.centers
        self.points = np.concatenate(
            [np.column_stack([np.full(j1 - j0, grid.x[i]), yc[j0:j1]]) for i in (i0, i1)]
            + [np.column_stack([xc[i0:i1], np.full(i1 - i0, grid.y[j])]) for j in (j0, j1)]
        )
        self.normals = np.concatenate(
            [
                np.tile(n, (count, 1))
                for n, count in (
                    ((-1, 0), j1 - j0),
                    ((1, 0), j1 - j0),
                    ((0, -1), i1 - i0),
                    ((0, 1), i1 - i0),
                )
            ]
        )
        self.weights = np.r_[
            np.diff(grid.y)[j0:j1],
            np.diff(grid.y)[j0:j1],
            np.diff(grid.x)[i0:i1],
            np.diff(grid.x)[i0:i1],
        ]

    def fields(self, ez, hx, hy):
        i0, i1, j0, j1 = self.i0, self.i1, self.j0, self.j1
        electric, magnetic = [], []
        for i, sign in ((i0, -1), (i1, 1)):
            electric.append((ez[i, j0:j1] + ez[i, j0 + 1 : j1 + 1]) / 2)
            left, right = self.grid.x[i] - self.grid.x[i - 1], self.grid.x[i + 1] - self.grid.x[i]
            h = (right * hy[i - 1, :] + left * hy[i, :]) / (left + right)
            magnetic.append(sign * (h[j0:j1] + h[j0 + 1 : j1 + 1]) / 2)
        for j, sign in ((j0, 1), (j1, -1)):
            electric.append((ez[i0:i1, j] + ez[i0 + 1 : i1 + 1, j]) / 2)
            left, right = self.grid.y[j] - self.grid.y[j - 1], self.grid.y[j + 1] - self.grid.y[j]
            h = (right * hx[:, j - 1] + left * hx[:, j]) / (left + right)
            magnetic.append(sign * (h[i0:i1] + h[i0 + 1 : i1 + 1]) / 2)
        return np.concatenate(electric), np.concatenate(magnetic)


class SurfaceDFT:
    def __init__(self, contour, frequencies):
        frequencies = np.array(frequencies, dtype=float, copy=True)
        if (
            frequencies.ndim != 1
            or not len(frequencies)
            or not np.isfinite(frequencies).all()
            or np.any(frequencies <= 0)
        ):
            raise ValueError("Positive finite observation frequencies required")
        self.contour, self.frequencies = contour, frequencies
        self.electric = np.zeros((len(frequencies), len(contour.weights)), dtype=np.complex128)
        self.tangential_h = np.zeros_like(self.electric)
        self.incident = np.zeros(len(frequencies), dtype=np.complex128)
        self.incident_l1 = 0.0

    def accumulate(self, ez, hx, hy, *, electric_time, magnetic_time, dt, incident):
        e, h = self.contour.fields(ez, hx, hy)
        pe = np.exp(2j * np.pi * self.frequencies * electric_time) * dt
        ph = np.exp(2j * np.pi * self.frequencies * magnetic_time) * dt
        self.electric += pe[:, None] * e
        self.tangential_h += ph[:, None] * h
        self.incident += pe * incident
        self.incident_l1 += dt * abs(incident)

    def far_amplitude(self, angles):
        angles = np.atleast_1d(np.asarray(angles, dtype=float))
        if angles.ndim != 1 or not np.isfinite(angles).all():
            raise ValueError("Finite observation angles required")
        direction = np.column_stack([np.cos(angles), np.sin(angles)])
        normal_dot = direction @ self.contour.normals.T
        position_dot = direction @ self.contour.points.T
        amplitude = []
        for index, frequency in enumerate(self.frequencies):
            omega, k = 2 * np.pi * frequency, 2 * np.pi * frequency / C0
            integrand = 1j * (
                omega * MU0 * self.tangential_h[index][None, :]
                - k * normal_dot * self.electric[index][None, :]
            )
            amplitude.append(
                np.exp(1j * np.pi / 4)
                / np.sqrt(8 * np.pi * k)
                * ((integrand * np.exp(-1j * k * position_dot)) @ self.contour.weights)
            )
        return np.asarray(amplitude)

    def normalized_far_field(self, angles, *, incident_floor=1e-3):
        """Complex A/Einc [sqrt(m)], retaining amplitude and absolute phase.

        A uses the global coordinate origin in exp(i*k*r)/sqrt(r). Einc is
        sampled at the incident wave's declared phase origin. Never align the
        prediction's phase to a reference before comparing physics targets.
        """
        if not np.isfinite(incident_floor) or not 0 < incident_floor < 1:
            raise ValueError("incident_floor must be between 0 and 1")
        amplitude = abs(self.incident)
        # The time-domain L1 norm also detects a frequency list entirely outside
        # the source band; a relative check among requested bins alone cannot.
        scale = max(float(amplitude.max()), self.incident_l1)
        if scale == 0 or np.any(amplitude < incident_floor * scale):
            raise ValueError("Incident spectrum too small for reliable normalization")
        return self.far_amplitude(angles) / self.incident[:, None]

    def scattering_width(self, angles, *, incident_floor=1e-3):
        return (
            2 * np.pi * abs(self.normalized_far_field(angles, incident_floor=incident_floor)) ** 2
        )
