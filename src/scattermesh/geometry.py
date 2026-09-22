"""Continuous dielectric geometry sampled over Ez dual-cell areas; SI units."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PEC:
    """Perfect electric conductor, handled by boundary geometry, never averaged."""


@dataclass(frozen=True)
class Material:
    epsilon_r: float = 1.0
    sigma_e: float = 0.0

    def __post_init__(self):
        if (
            not np.isfinite([self.epsilon_r, self.sigma_e]).all()
            or self.epsilon_r < 1
            or self.sigma_e < 0
        ):
            raise ValueError("Passive nondispersive material requires epsilon_r>=1, sigma_e>=0")


@dataclass(frozen=True)
class Circle:
    center: tuple[float, float]
    radius: float
    material: Material | PEC

    def __post_init__(self):
        if (
            len(self.center) != 2
            or not np.isfinite([*self.center, self.radius]).all()
            or self.radius <= 0
        ):
            raise ValueError("Invalid circle")

    @property
    def bounds(self):
        x, y = self.center
        return x - self.radius, x + self.radius, y - self.radius, y + self.radius

    def contains(self, x, y):
        return (x - self.center[0]) ** 2 + (y - self.center[1]) ** 2 <= self.radius**2


@dataclass(frozen=True)
class Rectangle:
    bounds: tuple[float, float, float, float]
    material: Material | PEC

    def __post_init__(self):
        a, b, c, d = self.bounds
        if not np.isfinite(self.bounds).all() or a >= b or c >= d:
            raise ValueError("Invalid rectangle bounds")

    def contains(self, x, y):
        a, b, c, d = self.bounds
        return (x >= a) & (x <= b) & (y >= c) & (y <= d)


def average_materials(grid, objects, samples=8):
    """Midpoint filling fractions; later objects override earlier ones at each sample."""
    objects = tuple(objects)
    if any(isinstance(obj.material, PEC) for obj in objects):
        raise ValueError("PEC requires a boundary treatment, not material averaging")
    if isinstance(samples, bool) or int(samples) != samples or samples < 1:
        raise ValueError("Quadrature samples must be a positive integer")
    bounds = [np.r_[a[0], (a[:-1] + a[1:]) / 2, a[-1]] for a in (grid.x, grid.y)]
    eps, sigma = np.ones(grid.shape), np.zeros(grid.shape)
    if not objects:
        return eps, sigma
    # Sample only dual cells intersecting a material bounding box.
    selected = np.zeros(grid.shape, dtype=bool)
    bx, by = bounds
    for obj in objects:
        a, b, c, d = obj.bounds
        selected |= ((bx[:-1] < b) & (bx[1:] > a))[:, None] & ((by[:-1] < d) & (by[1:] > c))[
            None, :
        ]
    ix, iy = np.nonzero(selected)
    e, s = np.zeros(len(ix)), np.zeros(len(ix))
    for qx in (np.arange(samples) + 0.5) / samples:
        x = bx[ix] + qx * (bx[ix + 1] - bx[ix])
        for qy in (np.arange(samples) + 0.5) / samples:
            y = by[iy] + qy * (by[iy + 1] - by[iy])
            eq, sq = np.ones(len(ix)), np.zeros(len(ix))
            for obj in objects:
                mask = obj.contains(x, y)
                eq[mask], sq[mask] = obj.material.epsilon_r, obj.material.sigma_e
            e += eq
            s += sq
    eps[ix, iy], sigma[ix, iy] = e / samples**2, s / samples**2
    return eps, sigma
