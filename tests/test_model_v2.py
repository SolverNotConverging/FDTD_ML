"""Trainable density and masked-teacher invariants for the v2 CNN."""

import pytest

torch = pytest.importorskip("torch")

from scattermesh.model_v2 import AxisDensityUNetV2, set_valued_profile_loss  # noqa: E402


def test_small_model_emits_positive_normalized_profiles_and_gradients():
    model = AxisDensityUNetV2(base_channels=16)
    raster = torch.randn(1, 9, 64, 64)
    conditioning = torch.randn(1, 15)
    prediction = model(raster, conditioning)
    assert prediction.shape == (1, 2, 64)
    assert torch.isfinite(prediction).all()
    assert (prediction > 0).all()
    torch.testing.assert_close(prediction.sum(-1), torch.ones(1, 2))

    targets = torch.stack((torch.full((2, 64), 1 / 64), prediction.detach()[0]), dim=0)[None]
    scores = torch.tensor([[0.2, 0.1]])
    mask = torch.tensor([[True, True]])
    loss, _ = set_valued_profile_loss(prediction, targets, scores, torch.tensor([1]), mask)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_teacher_loss_rejects_masked_best_candidate():
    prediction = torch.full((1, 2, 32), 1 / 32)
    targets = prediction[:, None].expand(1, 2, 2, 32)
    with pytest.raises(ValueError, match="best candidate"):
        set_valued_profile_loss(
            prediction,
            targets,
            torch.tensor([[0.1, 0.2]]),
            torch.tensor([1]),
            torch.tensor([[True, False]]),
        )
