"""Joint complex-field/RCS scores and phase diagnostics for candidate meshes."""

import numpy as np


def compare_far_fields(prediction, reference, *, amplitude_floor=1e-12, phase_fraction=0.05):
    """Compare [frequency, observation_angle] fields without phase alignment.

    amplitude_floor has units sqrt(m) for normalized 2D far fields. A campaign
    must choose it using its physical scale/noise floor. Phase RMS is diagnostic
    only, excluding reference nulls and predictions with undefined phase.
    """
    prediction, reference = np.asarray(prediction), np.asarray(reference)
    if (
        prediction.shape != reference.shape
        or reference.ndim != 2
        or 0 in reference.shape
        or not np.isfinite(prediction).all()
        or not np.isfinite(reference).all()
    ):
        raise ValueError("Matching finite [frequency, angle] far fields required")
    if not np.isfinite(amplitude_floor) or amplitude_floor <= 0 or not 0 < phase_fraction < 1:
        raise ValueError("Positive amplitude floor and phase fraction in (0, 1) required")
    absolute_rms = np.sqrt(np.mean(abs(prediction - reference) ** 2, axis=1))
    scale = np.sqrt(np.mean(abs(reference) ** 2, axis=1) + amplitude_floor**2)
    amplitude = abs(reference)
    valid = (
        (amplitude >= phase_fraction * amplitude.max(axis=1, keepdims=True))
        & (amplitude > amplitude_floor)
        & (abs(prediction) > amplitude_floor)
    )
    weight = np.where(valid, amplitude**2, 0.0)
    difference = np.angle(prediction * np.conj(reference))
    phase_rms = []
    for w, d in zip(weight, difference):
        phase_rms.append(
            float(np.rad2deg(np.sqrt(np.sum(w * d * d) / w.sum()))) if w.sum() else None
        )
    return dict(
        complex_relative_l2_by_frequency=(absolute_rms / scale).tolist(),
        complex_absolute_rms_by_frequency=absolute_rms.tolist(),
        phase_weighted_rms_degrees=phase_rms,
        phase_valid_angle_counts=valid.sum(axis=1).tolist(),
        amplitude_floor_sqrt_m=amplitude_floor,
        phase_reference_fraction=phase_fraction,
        phase_alignment_applied=False,
    )


def scattering_loss(
    prediction,
    reference,
    *,
    complex_weight=1.0,
    rcs_weight=0.25,
    amplitude_floor=1e-12,
    rcs_floor_fraction=1e-4,
):
    """Joint complex-field + floored log-scattering-width score for mesh ranking.

    RCS dB error is divided by 20/ln(10), so its small-amplitude-error limit is
    comparable to squared relative amplitude error. Weights 1/.25 and a -40 dB
    reference-power floor are provisional pilot settings, not tuned CNN weights.
    Inputs are [frequency, angle] on uniformly weighted observation samples.
    Computational-budget and mesh-feasibility checks are separate hard gates.
    """
    if (
        not np.isfinite([complex_weight, rcs_weight, rcs_floor_fraction]).all()
        or complex_weight < 0
        or rcs_weight < 0
        or complex_weight + rcs_weight <= 0
        or not 0 < rcs_floor_fraction < 1
    ):
        raise ValueError("Nonnegative loss weights and RCS floor fraction in (0,1) required")
    metrics = compare_far_fields(prediction, reference, amplitude_floor=amplitude_floor)
    prediction, reference = np.asarray(prediction), np.asarray(reference)
    predicted_width, reference_width = (
        2 * np.pi * abs(prediction) ** 2,
        2 * np.pi * abs(reference) ** 2,
    )
    floor = np.maximum(
        rcs_floor_fraction * reference_width.max(axis=1, keepdims=True),
        2 * np.pi * amplitude_floor**2,
    )
    db_error = 10 / np.log(10) * np.log((predicted_width + floor) / (reference_width + floor))
    db_scale = 20 / np.log(10)
    complex_loss = float(np.mean(np.asarray(metrics["complex_relative_l2_by_frequency"]) ** 2))
    rcs_db_mse = float(np.mean(db_error**2))
    rcs_loss = rcs_db_mse / db_scale**2
    return dict(
        joint_scattering_loss=complex_weight * complex_loss + rcs_weight * rcs_loss,
        complex_mse_loss=complex_loss,
        rcs_log_loss=rcs_loss,
        rcs_db_mse=rcs_db_mse,
        scattering_loss_config=dict(
            complex_weight=complex_weight,
            rcs_weight=rcs_weight,
            amplitude_floor_sqrt_m=amplitude_floor,
            rcs_floor_fraction=rcs_floor_fraction,
            rcs_db_scale=db_scale,
            weights_status="provisional_pilot",
        ),
    )
