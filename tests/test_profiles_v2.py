"""Physical-grid invariants retained by the active mesh-profile projection."""

import numpy as np
import pytest

from scattermesh import Grid
from scattermesh.monitor_v2 import widest_non_pml_monitor_bounds
from scattermesh.profiles_v2 import axis_probability, probability_axis, resample_axis_profiles


def test_exact_axis_roundtrip_and_grading_repair():
    axis = np.linspace(0, 1.2, 49)
    profile = axis_probability(axis, 128)
    reconstructed, repair = probability_axis(profile, 48)
    assert repair == 0
    assert reconstructed == pytest.approx(axis, abs=1e-12)

    spike = np.full(128, 1e-12)
    spike[63:65] = 1
    repaired, repair = probability_axis(spike, 32, max_ratio=3)
    widths = np.diff(repaired)
    ratio = max(np.max(widths[1:] / widths[:-1]), np.max(widths[:-1] / widths[1:]))
    assert len(repaired) == 33
    assert repaired[0] == 0 and repaired[-1] == 1.2
    assert repair > 0
    assert ratio <= 3 * (1 + 1e-12)


def test_profile_resampling_preserves_mass():
    values = np.full((2, 3, 2, 128), 1 / 128, dtype=np.float32)
    expanded = resample_axis_profiles(values, 512)
    assert expanded.shape == (2, 3, 2, 512)
    assert expanded.sum(axis=-1) == pytest.approx(np.ones((2, 3, 2)))
    assert expanded == pytest.approx(np.full_like(expanded, 1 / 512), abs=1e-7)


def test_monitor_has_pml_stencil_and_vacuum_buffer():
    grid = Grid(np.linspace(0, 1.2, 33), np.linspace(0, 1.2, 33))
    bounds = [(0.82 - 0.135, 0.82 + 0.135, 0.82 - 0.135, 0.82 + 0.135)]
    monitor = widest_non_pml_monitor_bounds(grid, bounds, 0.12)
    assert monitor == pytest.approx((0.1875, 1.0125, 0.1875, 1.0125))
