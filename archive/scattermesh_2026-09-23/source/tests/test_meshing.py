import numpy as np
import pytest

from scattermesh import Grid, circular_interface_axes, density_axis


def test_density_axis_has_exact_count_and_concentrates_near_focus():
    axis = density_axis(1.0, 32, [(0.37, 0.05, 4.0)])
    assert len(axis) == 33
    assert axis[0] == 0
    assert axis[-1] == 1
    assert np.all(np.diff(axis) > 0)
    focused = np.diff(axis)[np.argmin(abs((axis[:-1] + axis[1:]) / 2 - 0.37))]
    assert focused < np.median(np.diff(axis))


def test_circle_importance_projects_to_both_interfaces_without_anchors():
    x, y = circular_interface_axes(1.2, 48, (0.61, 0.57), 0.08, width=0.02, weight=2)
    grid = Grid(x, y, max_ratio=3)
    assert grid.shape == (49, 49)
    assert not np.any(np.isin([0.53, 0.69], x))
    assert not np.any(np.isin([0.49, 0.65], y))


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(length=-1, cells=8),
        dict(length=1, cells=3),
        dict(length=1, cells=8, foci=[(0.5, 0, 1)]),
        dict(length=1, cells=8, foci=[(1.1, 0.1, 1)]),
        dict(length=1, cells=8, foci=[(0.5, 0.1, -1)]),
    ],
)
def test_density_axis_rejects_invalid_controls(kwargs):
    with pytest.raises(ValueError):
        density_axis(**kwargs)
