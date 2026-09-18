"""Small NumPy oracle for tests, never an automatic production fallback."""

import numpy as np


def run_reference(c, initial, source_indices, waveforms, receiver_indices):
    ez, hx, hy = [a.copy() for a in initial]
    nt = len(waveforms)
    history = np.empty((nt, len(receiver_indices)), dtype=ez.dtype)
    nx, ny = ez.shape[0] - 1, ez.shape[1] - 1
    psi_hx, psi_hy = np.zeros_like(hx), np.zeros_like(hy)
    psi_ex, psi_ey = np.zeros_like(ez), np.zeros_like(ez)
    if c.cpml.size:
        ex, ey, hpx, hpy = np.split(c.cpml, [nx + 1, nx + ny + 2, 2 * nx + ny + 2])
    for n in range(nt):
        dy, dx = np.diff(ez, axis=1), np.diff(ez, axis=0)
        if c.cpml.size:
            psi_hx = hpy[None, :, 1] * psi_hx + hpy[None, :, 2] * dy
            psi_hy = hpx[:, None, 1] * psi_hy + hpx[:, None, 2] * dx
            dy, dx = hpy[None, :, 0] * dy + psi_hx, hpx[:, None, 0] * dx + psi_hy
        hx -= c.chx * dy
        hy += c.chy * dx
        dex, dey = np.diff(hy, axis=0)[:, 1:-1], np.diff(hx, axis=1)[1:-1, :]
        if c.cpml.size:
            psi_ex[1:-1, 1:-1] = ex[1:-1, None, 1] * psi_ex[1:-1, 1:-1] + ex[1:-1, None, 2] * dex
            psi_ey[1:-1, 1:-1] = ey[None, 1:-1, 1] * psi_ey[1:-1, 1:-1] + ey[None, 1:-1, 2] * dey
            dex = ex[1:-1, None, 0] * dex + psi_ex[1:-1, 1:-1]
            dey = ey[None, 1:-1, 0] * dey + psi_ey[1:-1, 1:-1]
        ez[1:-1, 1:-1] = (
            c.ca[1:-1, 1:-1] * ez[1:-1, 1:-1] + c.cbx[1:-1, 1:-1] * dex - c.cby[1:-1, 1:-1] * dey
        )
        np.add.at(ez.ravel(), source_indices, waveforms[n])
        ez[c.pec.astype(bool)] = 0
        history[n] = ez.ravel()[receiver_indices]
    return ez, hx, hy, history
