import numpy as np
import pytest

from scattermesh import PEC, Circle, Grid, PlaneWave, focused_axis, simulate
from scattermesh.analytic import cylinder_far_field
from scattermesh.constants import C0
from scattermesh.metrics import compare_far_fields, scattering_loss


def test_phase_only_error_is_visible_even_when_rcs_is_identical():
    reference = np.array([[1 + 2j, 2 - 1j, 0.1j]])
    prediction = 1j * reference
    np.testing.assert_allclose(abs(prediction) ** 2, abs(reference) ** 2)
    metrics = compare_far_fields(prediction, reference)
    assert metrics["complex_relative_l2_by_frequency"][0] == pytest.approx(np.sqrt(2))
    assert metrics["phase_weighted_rms_degrees"][0] == pytest.approx(90)
    assert metrics["phase_alignment_applied"] is False


def test_complex_translation_law_preserves_physical_phase():
    angles = np.linspace(0, 2 * np.pi, 80, endpoint=False)
    shift = np.array([0.0321, -0.027])
    theta = 0.7
    before = cylinder_far_field(0.08, PEC(), 1e9, angles, theta)
    after = cylinder_far_field(0.08, PEC(), 1e9, angles, theta, center=shift)
    direction = np.array([np.cos(theta), np.sin(theta)])
    factor = np.exp(
        2j
        * np.pi
        * 1e9
        / C0
        * (direction @ shift - shift[0] * np.cos(angles) - shift[1] * np.sin(angles))
    )
    np.testing.assert_allclose(after, before * factor)
    np.testing.assert_allclose(abs(after), abs(before))
    assert (
        compare_far_fields(after[None, :], before[None, :])["complex_relative_l2_by_frequency"][0]
        > 0.5
    )


def test_zero_field_has_no_defined_phase_and_finite_absolute_error():
    metrics = compare_far_fields(np.zeros((2, 8)), np.zeros((2, 8)))
    assert metrics["phase_weighted_rms_degrees"] == [None, None]
    assert metrics["complex_relative_l2_by_frequency"] == [0.0, 0.0]
    assert metrics["phase_valid_angle_counts"] == [0, 0]


def test_joint_loss_includes_rcs_and_preserves_phase_sensitivity():
    reference = np.array([[1 + 2j, 2 - 1j, 0.1j]])
    phase = scattering_loss(1j * reference, reference)
    assert phase["complex_mse_loss"] == pytest.approx(2)
    assert phase["rcs_log_loss"] == pytest.approx(0)
    assert phase["joint_scattering_loss"] == pytest.approx(2)
    amplitude = scattering_loss(1.1 * reference, reference)
    assert amplitude["complex_mse_loss"] == pytest.approx(0.01)
    assert amplitude["rcs_log_loss"] > 0
    assert amplitude["joint_scattering_loss"] == pytest.approx(
        amplitude["complex_mse_loss"] + 0.25 * amplitude["rcs_log_loss"]
    )
    assert amplitude["joint_scattering_loss"] > amplitude["complex_mse_loss"]


def test_joint_loss_is_finite_at_scattering_nulls():
    reference = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    result = scattering_loss(reference, reference)
    assert result["joint_scattering_loss"] == 0
    prediction = reference + 1e-6
    result = scattering_loss(prediction, reference)
    assert np.isfinite(result["joint_scattering_loss"])


def test_pec_complex_far_field_matches_absolute_analytic_phase():
    x = focused_axis(1.2, 96)
    source = PlaneWave(1e9, 1e-9, 9e-9, angle=0.7, origin=(0.6, 0.6))
    center = (0.613, 0.591)
    frequencies = [0.8e9, 1e9, 1.2e9]
    angles = np.linspace(0, 2 * np.pi, 90, endpoint=False)
    result = simulate(
        Grid(x, x),
        [Circle(center, 0.08, PEC())],
        source,
        frequencies=frequencies,
        duration=35e-9,
        pml_thickness=0.15,
        pec_mode="enlarged",
    )
    reference = np.array(
        [
            cylinder_far_field(
                0.08, PEC(), f, angles, source.angle, center=center, incident_origin=source.origin
            )
            for f in frequencies
        ]
    )
    field = result.monitor.normalized_far_field(angles)
    metrics = compare_far_fields(field, reference)
    assert max(metrics["complex_relative_l2_by_frequency"]) < 0.02
    assert max(metrics["phase_weighted_rms_degrees"]) < 1.0
    np.testing.assert_allclose(result.monitor.scattering_width(angles), 2 * np.pi * abs(field) ** 2)
    assert result.diagnostics["incident_phase_origin"] == [0.6, 0.6]
    assert result.diagnostics["far_field_origin"] == [0.0, 0.0]
