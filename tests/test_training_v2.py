"""Curriculum phase contracts that can be checked without a training GPU."""

import pytest

from scattermesh.training_v2 import curriculum_stages


def test_cumulative_training_schedule_introduces_separated_scenes_after_warmup():
    assert curriculum_stages(0) == {"C0"}
    assert curriculum_stages(9) == {"C0"}
    assert curriculum_stages(10) == {"C0", "C1", "C2", "C3", "C5"}
    assert curriculum_stages(59) == {"C0", "C1", "C2", "C3", "C5"}


@pytest.mark.parametrize("epoch", (-1, 1.5, True))
def test_curriculum_schedule_rejects_invalid_epochs(epoch):
    with pytest.raises(ValueError, match="nonnegative integer"):
        curriculum_stages(epoch)
