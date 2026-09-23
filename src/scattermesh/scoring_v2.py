"""Accuracy-led teacher ranking and equal-accuracy cell savings."""

import math

import numpy as np


def soft_dt_score(accuracy_loss, candidate_dt, uniform_dt, *, exponent=0.05):
    if (
        not math.isfinite(float(accuracy_loss))
        or accuracy_loss < 0
        or not all(math.isfinite(float(x)) and float(x) > 0 for x in (candidate_dt, uniform_dt))
    ):
        raise ValueError(
            "Accuracy loss must be finite and nonnegative; time steps must be positive"
        )
    if not math.isfinite(exponent) or exponent < 0:
        raise ValueError("The time-step exponent must be finite and nonnegative")
    return float(accuracy_loss * (uniform_dt / candidate_dt) ** exponent)


def rank_candidates(rows, *, exponent=0.05):
    """Return scored valid candidates, preserving failed rows for provenance."""
    uniform = next((r for r in rows if r["name"] == "uniform"), None)
    if uniform is None or not uniform.get("accepted"):
        raise ValueError("An accepted uniform baseline is required")
    reference_dt = float(uniform["dt"])
    ranked = []
    for row in rows:
        result = dict(row)
        if row.get("accepted"):
            result["teacher_score"] = soft_dt_score(
                row["joint_scattering_loss"], row["dt"], reference_dt, exponent=exponent
            )
            ranked.append(result)
    ranked.sort(key=lambda row: (row["teacher_score"], row["name"]))
    return ranked


def penalty_sensitivity(rows, exponents=(0.0, 0.02, 0.05, 0.1)):
    return {
        str(value): {
            "best_candidate": selected[0]["name"],
            "best_teacher_score": selected[0]["teacher_score"],
            "best_raw_accuracy_loss": selected[0]["joint_scattering_loss"],
        }
        for value in exponents
        if (selected := rank_candidates(rows, exponent=value))
    }


def relative_l2(prediction, reference, *, floor=1e-12):
    prediction = np.asarray(prediction)
    reference = np.asarray(reference)
    if (
        prediction.shape != reference.shape
        or not np.isfinite(prediction).all()
        or not np.isfinite(reference).all()
    ):
        raise ValueError("Matching finite arrays are required")
    return float(np.linalg.norm(prediction - reference) / max(np.linalg.norm(reference), floor))


def threshold_cell_savings(uniform, learned, thresholds=(0.02, 0.05, 0.10)):
    """Find minimum *tested* cell counts meeting both physical error limits."""
    output = {}
    for threshold in thresholds:
        selection = {}
        for name, records in (("uniform", uniform), ("learned", learned)):
            eligible = [
                row
                for row in records
                if row.get("accepted")
                and row["complex_relative_l2"] <= threshold
                and row["width_relative_l2"] <= threshold
            ]
            selection[name] = min(
                (int(row["cells_x"]) * int(row["cells_y"]) for row in eligible), default=None
            )
        cells_u, cells_l = selection["uniform"], selection["learned"]
        output[str(threshold)] = {
            "minimum_tested_uniform_cells": cells_u,
            "minimum_tested_learned_cells": cells_l,
            "cell_saving_ratio": cells_u / cells_l
            if cells_u is not None and cells_l is not None
            else None,
            "censored": cells_u is None or cells_l is None,
        }
    return output
