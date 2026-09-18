"""Small NumPy oracle for tests, never an automatic production fallback."""

import numpy as np


def run_reference(c, initial, source_indices, waveforms, receiver_indices):
    ez, hx, hy = [a.copy() for a in initial]
    nt = len(waveforms)
    history = np.empty((nt, len(receiver_indices)), dtype=ez.dtype)
    for n in range(nt):
        hx -= c.chx * np.diff(ez, axis=1)
        hy += c.chy * np.diff(ez, axis=0)
        ez[1:-1, 1:-1] = (
            c.ca[1:-1, 1:-1] * ez[1:-1, 1:-1]
            + c.cbx[1:-1, 1:-1] * np.diff(hy, axis=0)[:, 1:-1]
            - c.cby[1:-1, 1:-1] * np.diff(hx, axis=1)[1:-1, :]
        )
        np.add.at(ez.ravel(), source_indices, waveforms[n])
        ez[c.pec.astype(bool)] = 0
        history[n] = ez.ravel()[receiver_indices]
    return ez, hx, hy, history
