"""Continuous CSG bounds from boundary extrema and analytic intersections."""

import numpy as np
from numpy.polynomial import Polynomial as Poly


def material_bounds(scene):
    points, segments, curves = [], [], []
    for kind, _, data in scene.primitives:
        if kind == "polygon":
            points.extend(data)
            segments.extend(zip(data, np.roll(data, -1, axis=0)))
        else:
            if kind == "circle":
                x, y, a = data
                b, angle = a, 0.0
            else:
                x, y, a, b, angle = data
            c = np.array([x, y])
            co, si = np.cos(angle), np.sin(angle)
            A = np.array([[co * a, -si * b], [si * a, co * b]])
            Q = np.linalg.inv(A @ A.T)
            curves.append((c, A, Q))
            covariance = A @ A.T
            for axis in (0, 1):
                v = covariance[:, axis] / np.sqrt(covariance[axis, axis])
                points.extend([c - v, c + v])

    def cross(u, v):
        return u[0] * v[1] - u[1] * v[0]

    for i, (a, b) in enumerate(segments):
        v = b - a
        for c, d in segments[i + 1 :]:
            w = d - c
            den = cross(v, w)
            if abs(den) > 1e-14 * np.linalg.norm(v) * np.linalg.norm(w):
                t, u = cross(c - a, w) / den, cross(c - a, v) / den
                if -1e-12 <= t <= 1 + 1e-12 and -1e-12 <= u <= 1 + 1e-12:
                    points.append(a + np.clip(t, 0, 1) * v)
        for c, A, Q in curves:
            u = a - c
            roots = np.roots([v @ Q @ v, 2 * v @ Q @ u, u @ Q @ u - 1])
            for t in roots:
                if abs(t.imag) < 1e-9 and -1e-12 <= t.real <= 1 + 1e-12:
                    points.append(a + np.clip(t.real, 0, 1) * v)
    # tan(theta/2) turns intersection of two ellipses into a quartic.
    D = Poly([1.0, 0.0, 1.0])
    for i, (c, A, _) in enumerate(curves):
        for d, _, Q in curves[i + 1 :]:
            u = [Poly([c[k] - d[k] + A[k, 0], 2 * A[k, 1], c[k] - d[k] - A[k, 0]]) for k in (0, 1)]
            polynomial = (
                Q[0, 0] * u[0] * u[0] + 2 * Q[0, 1] * u[0] * u[1] + Q[1, 1] * u[1] * u[1] - D * D
            )
            scale = max(1.0, np.max(abs(polynomial.coef)))
            polynomial = polynomial.trim(tol=1e-13 * scale)
            if polynomial.degree() > 0:
                for t in polynomial.roots():
                    if abs(t.imag) <= 1e-8 * max(1.0, abs(t.real)):
                        theta = 2 * np.arctan(t.real)
                        points.append(c + A @ np.array([np.cos(theta), np.sin(theta)]))
            # The missing tan-half-angle point is theta=pi.
            p = c - A[:, 0]
            if abs((p - d) @ Q @ (p - d) - 1) < 1e-10:
                points.append(p)
    if not points:
        return None
    points = np.asarray(points)
    points = points[scene.contains(points[:, 0], points[:, 1])]
    if not len(points):
        return None
    lo, hi = points.min(axis=0), points.max(axis=0)
    return tuple(float(v) for v in (lo[0], hi[0], lo[1], hi[1]))
