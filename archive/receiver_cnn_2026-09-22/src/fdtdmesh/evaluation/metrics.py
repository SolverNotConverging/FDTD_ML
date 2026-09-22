"""Matched physical-time waveform and spectrum metrics with explicit quiet-signal floors."""

import numpy as np


def sample_observables(result, times):
    times = np.asarray(times, dtype=float)
    if (
        times.ndim != 1
        or len(times) < 2
        or np.any(np.diff(times) <= 0)
        or not np.isfinite(times).all()
    ):
        raise ValueError("Expected increasing finite observation times")
    if times[0] < 0 or times[-1] > result.times[-1]:
        raise ValueError("Observation times require extrapolation")
    # Stage-3 runs start with zero fields; the known t=0 sample is exact.
    columns = []
    for key in sorted(result.receivers):
        for column in result.receivers[key].T:
            columns.append(np.interp(times, np.r_[0, result.times], np.r_[0, column]))
    if not columns:
        raise ValueError("No receiver observations")
    values = np.column_stack(columns)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite solver observations")
    return values


def spectrum(values, times, frequencies):
    times, frequencies = np.asarray(times), np.asarray(frequencies)
    dt = np.diff(times)
    if not np.allclose(dt, dt[0], rtol=1e-10, atol=0):
        raise ValueError("Spectral sampling requires uniform physical times")
    if np.any(frequencies < 0) or np.any(frequencies > 0.5 / dt[0]):
        raise ValueError("Frequencies outside observation Nyquist band")
    # Common Hann window and trapezoidal time integral; no per-run normalization.
    weights = np.hanning(len(times)) * dt[0]
    weights[[0, -1]] *= 0.5
    # Bound temporary memory as longer ring-down windows add times/frequencies.
    return np.concatenate(
        [
            (np.exp(-2j * np.pi * frequencies[i : i + 64, None] * times) * weights) @ values
            for i in range(0, len(frequencies), 64)
        ]
    )


def compare_observables(
    candidate, reference, times, frequencies, *, amplitude_floor=1e-8, phase_gate=0.01
):
    candidate, reference = np.asarray(candidate), np.asarray(reference)
    if candidate.shape != reference.shape or candidate.ndim != 2 or len(candidate) != len(times):
        raise ValueError("Observable shapes must match (time, receiver)")
    if not np.isfinite(candidate).all() or not np.isfinite(reference).all():
        raise ValueError("Observables must be finite")
    if amplitude_floor <= 0 or not 0 < phase_gate < 1:
        raise ValueError("Invalid normalization floor or phase gate")
    delta = candidate - reference
    denom = np.maximum(np.linalg.norm(reference, axis=0), amplitude_floor * np.sqrt(len(times)))
    waveform = np.linalg.norm(delta, axis=0) / denom
    peaks = np.max(abs(delta), axis=0) / np.maximum(np.max(abs(reference), axis=0), amplitude_floor)
    actual, expected = (
        spectrum(candidate, times, frequencies),
        spectrum(reference, times, frequencies),
    )
    spectral_floor = amplitude_floor * (times[-1] - times[0])
    spectral = np.linalg.norm(actual - expected, axis=0) / np.maximum(
        np.linalg.norm(expected, axis=0), spectral_floor
    )
    threshold = np.maximum(
        phase_gate * np.max(abs(expected), axis=0, keepdims=True), spectral_floor
    )
    valid = (abs(expected) > threshold) & (abs(actual) > spectral_floor)
    phase = np.angle(actual[valid] * np.conj(expected[valid]))
    return {
        "waveform_l2": float(waveform.mean()),
        "waveform_l2_max": float(waveform.max()),
        "waveform_l2_per_receiver": waveform.tolist(),
        "waveform_peak_max": float(peaks.max()),
        "spectrum_l2": float(spectral.mean()),
        "spectrum_l2_max": float(spectral.max()),
        "phase_rms_degrees": float(np.rad2deg(np.sqrt(np.mean(phase**2)))) if phase.size else None,
        "phase_samples": int(phase.size),
    }
