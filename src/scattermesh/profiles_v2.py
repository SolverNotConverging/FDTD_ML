"""Exact-budget density profile transforms for mesh CNN v2."""

import numpy as np

DOMAIN = 1.2


def axis_probability(axis, bins, *, length=DOMAIN):
    """Convert grid nodes to probability mass on fixed uniform spatial bins.

    Every mesh cell contributes equal mass. Within a cell that mass is distributed
    uniformly, so inverse-CDF projection approximately reconstructs the source axis.
    """
    axis = np.asarray(axis, dtype=np.float64)
    if (
        axis.ndim != 1
        or len(axis) < 5
        or not np.isfinite(axis).all()
        or axis[0] != 0
        or not np.isclose(axis[-1], length)
        or np.any(np.diff(axis) <= 0)
        or isinstance(bins, bool)
        or int(bins) != bins
        or bins < 8
    ):
        raise ValueError("A finite increasing axis and at least eight profile bins are required")
    edges = np.linspace(0.0, length, int(bins) + 1)
    cells = len(axis) - 1
    indices = np.searchsorted(axis, edges, side="right") - 1
    indices = np.clip(indices, 0, cells - 1)
    fractions = (edges - axis[indices]) / np.diff(axis)[indices]
    cumulative = (indices + np.clip(fractions, 0.0, 1.0)) / cells
    cumulative[0], cumulative[-1] = 0.0, 1.0
    probability = np.diff(cumulative)
    probability = np.maximum(probability, 0.0)
    return probability / probability.sum()


def _quantile_axis(probability, cells, length):
    probability = np.asarray(probability, dtype=np.float64)
    probability = np.maximum(probability, np.finfo(np.float64).tiny)
    probability /= probability.sum()
    cumulative = np.r_[0.0, np.cumsum(probability)]
    cumulative[-1] = 1.0
    quantiles = np.linspace(0.0, 1.0, int(cells) + 1)
    indices = np.searchsorted(cumulative, quantiles, side="right") - 1
    indices = np.clip(indices, 0, len(probability) - 1)
    fractions = (quantiles - cumulative[indices]) / probability[indices]
    axis = (indices + np.clip(fractions, 0.0, 1.0)) * length / len(probability)
    axis[0], axis[-1] = 0.0, float(length)
    return axis


def _axis_grading(axis):
    spacing = np.diff(axis)
    return float(max(np.max(spacing[1:] / spacing[:-1]), np.max(spacing[:-1] / spacing[1:])))


def probability_axis(probability, cells, *, length=DOMAIN, max_ratio=3.0):
    """Project a positive profile to exactly ``cells`` intervals with a grading cap.

    If the raw inverse-CDF axis exceeds the cap, the profile is minimally mixed
    toward uniform by bisection. The returned repair fraction is auditable and is
    zero when no repair was needed.
    """
    probability = np.asarray(probability, dtype=np.float64)
    if (
        probability.ndim != 1
        or len(probability) < 8
        or not np.isfinite(probability).all()
        or np.any(probability < 0)
        or probability.sum() <= 0
        or isinstance(cells, bool)
        or int(cells) != cells
        or cells < 4
        or not np.isfinite(length)
        or length <= 0
        or not np.isfinite(max_ratio)
        or max_ratio < 1
    ):
        raise ValueError("Invalid probability profile, cell count, domain, or grading cap")
    probability = probability / probability.sum()
    raw = _quantile_axis(probability, cells, length)
    if _axis_grading(raw) <= max_ratio * (1 + 1e-12):
        return raw, 0.0
    uniform = np.full_like(probability, 1 / len(probability))
    low, high = 0.0, 1.0
    accepted = np.linspace(0.0, length, int(cells) + 1)
    for _ in range(48):
        middle = (low + high) / 2
        candidate = _quantile_axis((1 - middle) * probability + middle * uniform, cells, length)
        if _axis_grading(candidate) <= max_ratio * (1 + 1e-12):
            high, accepted = middle, candidate
        else:
            low = middle
    return accepted, high


def resample_axis_profiles(profiles, bins):
    """Resample probability profiles by interpolating their piecewise-linear CDF."""
    values = np.asarray(profiles, dtype=np.float64)
    if values.ndim < 1 or values.shape[-1] < 8 or bins < 8:
        raise ValueError("Profiles and destination require at least eight bins")
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Profiles must be finite and nonnegative")
    values = values / values.sum(axis=-1, keepdims=True)
    old_edges = np.linspace(0.0, 1.0, values.shape[-1] + 1)
    new_edges = np.linspace(0.0, 1.0, int(bins) + 1)
    flattened = values.reshape(-1, values.shape[-1])
    result = np.empty((len(flattened), int(bins)), dtype=np.float64)
    for index, profile in enumerate(flattened):
        cumulative = np.r_[0.0, np.cumsum(profile)]
        cumulative[-1] = 1.0
        result[index] = np.diff(np.interp(new_edges, old_edges, cumulative))
    result = np.maximum(result, 0.0)
    result /= result.sum(axis=-1, keepdims=True)
    return result.reshape(*values.shape[:-1], int(bins)).astype(np.float32)
