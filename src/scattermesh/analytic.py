"""Independent infinite dielectric cylinder TMz scattering benchmark (mu_r=1)."""

import numpy as np
from scipy.special import h1vp, hankel1, jv, jvp

from .constants import C0, EPS0
from .geometry import PEC


def cylinder_width(radius, material, frequency, angles, incidence_angle=0.0, *, order=None):
    """2D scattering width [m] under exp(-i omega t); lossy eps has +imaginary part."""
    if radius <= 0 or frequency <= 0 or not np.isfinite([radius, frequency, incidence_angle]).all():
        raise ValueError("Positive finite radius and frequency required")
    k = 2 * np.pi * frequency / C0
    z = k * radius
    m = (
        1
        if isinstance(material, PEC)
        else np.sqrt(material.epsilon_r + 1j * material.sigma_e / (2 * np.pi * frequency * EPS0))
    )
    order = (
        int(np.ceil(max(z, abs(m * z)) + 4 * max(z, abs(m * z)) ** (1 / 3) + 12))
        if order is None
        else order
    )
    n = np.arange(-order, order + 1)
    if isinstance(material, PEC):
        a = -jv(n, z) / hankel1(n, z)
    else:
        a = (m * jvp(n, m * z) * jv(n, z) - jvp(n, z) * jv(n, m * z)) / (
            h1vp(n, z) * jv(n, m * z) - m * jvp(n, m * z) * hankel1(n, z)
        )
    amplitude = np.exp(1j * np.outer(np.asarray(angles) - incidence_angle, n)) @ a
    return 4 / k * abs(amplitude) ** 2
