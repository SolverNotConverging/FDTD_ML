"""Explicit NumPy conformal/ECT reference, with the same time staggering as CUDA."""

import numpy as np

from .conformal import apply_operator


def run_reference(c, shape, box, wave, source, contour, frequency, initial=None):
    nx, ny = shape
    dtype = c.ex.dtype
    ez = np.zeros((nx + 1, ny + 1), dtype=dtype)
    hx, hy = np.zeros((nx + 1, ny), dtype=dtype), np.zeros((nx, ny + 1), dtype=dtype)
    if initial is not None:
        ez[:], hx[:], hy[:] = initial
    ez[c.pec.astype(bool)] = 0
    ei, hi = np.zeros(nx + 1, dtype=dtype), np.zeros(nx, dtype=dtype)
    pe, ph = np.zeros_like(ei), np.zeros_like(hi)
    pex, pey, phx, phy = np.zeros_like(ez), np.zeros_like(ez), np.zeros_like(hx), np.zeros_like(hy)
    ex, ey, hpx, hpy = np.split(c.cpml, [nx + 1, nx + ny + 2, 2 * nx + ny + 2])
    a, b, low, high = box
    ii, jj = np.indices(ez.shape)
    oe = ~((ii >= a) & (ii <= b) & (jj >= low) & (jj <= high))
    ii, jj = np.indices(hy.shape)
    oy = ~((ii >= a) & (ii < b) & (jj >= low) & (jj <= high))
    out = np.zeros((len(contour.indices), 2), complex)
    inc = np.zeros(nx + 1, complex)
    samples = []
    stride = max(1, len(wave) // 200)
    for n, pulse in enumerate(wave):
        external = oe * ei[:, None]
        dx = np.diff(ez, axis=0) + np.diff(external, axis=0) - oy * np.diff(ei)[:, None]
        dy = np.diff(ez, axis=1) + np.diff(external, axis=1)
        phx = hpy[None, :, 1] * phx + hpy[None, :, 2] * dy
        phy = hpx[:, None, 1] * phy + hpx[:, None, 2] * dx
        hx -= apply_operator(c.hx, hpy[None, :, 0] * dy + phx)
        hy += apply_operator(c.hy, hpx[:, None, 0] * dx + phy)
        diff = np.diff(ei)
        ph = hpx[:, 1] * ph + hpx[:, 2] * diff
        hi += c.ih * (hpx[:, 0] * diff + ph)
        dx = (
            np.diff(hy, axis=0)[:, 1:-1]
            + np.diff(oy * hi[:, None], axis=0)[:, 1:-1]
            - oe[1:-1, 1:-1] * np.diff(hi)[:, None]
        )
        dy = np.diff(hx, axis=1)[1:-1]
        pex[1:-1, 1:-1] = ex[1:-1, None, 1] * pex[1:-1, 1:-1] + ex[1:-1, None, 2] * dx
        pey[1:-1, 1:-1] = ey[None, 1:-1, 1] * pey[1:-1, 1:-1] + ey[None, 1:-1, 2] * dy
        ez[1:-1, 1:-1] += c.ex[1:-1, None] * (ex[1:-1, None, 0] * dx + pex[1:-1, 1:-1]) - c.ey[
            None, 1:-1
        ] * (ey[None, 1:-1, 0] * dy + pey[1:-1, 1:-1])
        ez[c.pec.astype(bool)] = 0
        diff = np.diff(hi)
        pe[1:-1] = ex[1:-1, 1] * pe[1:-1] + ex[1:-1, 2] * diff
        ei[1:-1] += c.ex[1:-1] * (ex[1:-1, 0] * diff + pe[1:-1])
        ei[source] += pulse
        qe = c.dt * np.exp(-2j * np.pi * frequency * (n + 1) * c.dt)
        qh = c.dt * np.exp(-2j * np.pi * frequency * (n + 0.5) * c.dt)
        h = np.r_[hx.ravel(), hy.ravel()]
        indices, w = contour.indices, contour.weights
        out[:, 0] += qe * ez.ravel()[indices[:, 0]]
        out[:, 1] += qh * (w[:, 0] * h[indices[:, 1]] + w[:, 1] * h[indices[:, 2]])
        inc += qe * ei
        if n % stride == 0 or n == len(wave) - 1:
            samples.append((n + 1, float(np.max(abs(ez))), float(np.linalg.norm(ez))))
    if not all(np.isfinite(v).all() for v in (ez, hx, hy, out, inc)):
        raise FloatingPointError("Nonfinite fields in conformal FDTD")
    return ez, hx, hy, out, inc, np.asarray(samples), {"backend": "numpy"}
