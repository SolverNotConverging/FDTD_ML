"""Independent infinite dielectric cylinder TMz scattering benchmark (mu_r=1)."""

import numpy as np
from scipy.special import h1vp, hankel1, jv, jvp

from .constants import C0, EPS0
from .geometry import PEC


def cylinder_far_field(
    radius,
    material,
    frequency,
    angles,
    incidence_angle=0.0,
    *,
    center=(0.0, 0.0),
    incident_origin=(0.0, 0.0),
    order=None,
):
    """Complex normalized 2D amplitude [sqrt(m)], with global far-field origin.

    Uses exp(-i omega t). Translations retain the physical incident/scattered
    phase factors. Lossy permittivity has positive imaginary part.
    """
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
    angles = np.atleast_1d(angles)
    center, incident_origin = np.asarray(center), np.asarray(incident_origin)
    if (
        center.shape != (2,)
        or incident_origin.shape != (2,)
        or not np.isfinite([center, incident_origin]).all()
    ):
        raise ValueError("Finite 2D center and incident phase origin required")
    incident_direction = np.array([np.cos(incidence_angle), np.sin(incidence_angle)])
    observation_dot_center = np.cos(angles) * center[0] + np.sin(angles) * center[1]
    translation = np.exp(
        1j * k * (incident_direction @ (center - incident_origin) - observation_dot_center)
    )
    return np.sqrt(2 / (np.pi * k)) * np.exp(-1j * np.pi / 4) * amplitude * translation


def cylinder_width(radius, material, frequency, angles, incidence_angle=0.0, *, order=None):
    """2D scattering width [m], independent of cylinder translation."""
    return (
        2
        * np.pi
        * abs(cylinder_far_field(radius, material, frequency, angles, incidence_angle, order=order))
        ** 2
    )
