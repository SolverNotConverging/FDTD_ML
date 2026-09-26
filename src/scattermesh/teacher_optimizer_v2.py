"""Bounded, reproducible physics search for development teacher meshes.

The optimizer changes smooth log densities, never the continuous geometry.  Every
trial is projected to legal tensor-product axes before the compiled CUDA solver
scores it.  Saved numerical records remain independent of teacher-score ranking.
"""

import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution

from .candidates_v2 import candidate_axes
from .curriculum_v2 import DOMAIN
from .profiles_v2 import axis_probability, probability_axis
from .scoring_v2 import penalty_sensitivity, rank_candidates
from .simulation_v2 import evaluate_case, numerical_source_hashes

PROFILE_BINS = 512
COMPACT_TEACHER_NAMES = ("uniform", "center", "interface", "wide", "smooth_0", "smooth_1")
KNOTS_PER_AXIS = 6
PARAMETER_BOUND = 1.5
POPULATION = 2 * (2 * KNOTS_PER_AXIS)


class _DeadlineReached(Exception):
    """Stop a search without turning partial trials into training labels."""


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _basis():
    coordinate = (np.arange(PROFILE_BINS) + 0.5) / PROFILE_BINS
    centers = np.linspace(0.08, 0.92, KNOTS_PER_AXIS)
    width = 0.16
    basis = np.exp(-0.5 * ((coordinate[:, None] - centers[None, :]) / width) ** 2)
    return basis - basis.mean(axis=0, keepdims=True)


def density_axes(parameters, cells):
    """Project 12 bounded log-density weights to exact, graded x/y axes."""
    weights = np.asarray(parameters, dtype=np.float64)
    if weights.shape != (2 * KNOTS_PER_AXIS,) or not np.isfinite(weights).all():
        raise ValueError("Two finite six-knot density profiles are required")
    if np.any(np.abs(weights) > PARAMETER_BOUND + 1e-12):
        raise ValueError("Density weights exceed the search bounds")
    basis = _basis()
    axes, repairs = [], []
    for axis_weights in np.split(weights, 2):
        log_density = basis @ axis_weights
        density = np.exp(log_density - np.max(log_density))
        axis, repair = probability_axis(density, cells, length=DOMAIN, max_ratio=3.0)
        axes.append(axis)
        repairs.append(repair)
    return (axes[0], axes[1]), tuple(repairs)


def _initial_population(scene, cells, seed):
    """Include uniform and fitted versions of the already evaluated policies."""
    basis = _basis()
    population = [np.zeros(2 * KNOTS_PER_AXIS)]
    for name in COMPACT_TEACHER_NAMES:
        if name == "uniform":
            continue
        axes = candidate_axes(scene, cells, name)[:2]
        weights = []
        for axis in axes:
            target = np.log(np.maximum(axis_probability(axis, PROFILE_BINS), 1e-12))
            target -= target.mean()
            estimate = np.linalg.lstsq(basis, target, rcond=None)[0]
            weights.extend(np.clip(estimate, -PARAMETER_BOUND, PARAMETER_BOUND))
        population.append(np.asarray(weights))
    rng = np.random.default_rng(seed)
    while len(population) < POPULATION:
        population.append(rng.normal(0, 0.5, 2 * KNOTS_PER_AXIS).clip(-1.5, 1.5))
    return np.asarray(population)


def optimize_teacher(
    scene,
    cells,
    incidence_angle,
    reference,
    output,
    *,
    device="cuda:0",
    evaluations=96,
    seed=20260923,
    uniform_record=None,
    reference_uncertainty=None,
    deadline=None,
):
    """Search a development condition and retain all candidate records and axes.

    ``evaluations`` must be a multiple of 24; SciPy's population evaluates the
    initial 24 trials, then one full generation per additional 24 trials.  Repeating
    the same call reuses matching saved numerical trials.
    """
    if not device.startswith("cuda:"):
        raise ValueError("Teacher FDTD searches require the compiled CUDA backend")
    if cells not in (32, 48, 64, 96):
        raise ValueError("Pilot teacher searches use 32/48/64/96 cells per axis")
    if evaluations < POPULATION or evaluations % POPULATION:
        raise ValueError("Evaluation count must be a positive multiple of 24")
    if scene.get("split") == "test":
        raise ValueError("Frozen test scenes cannot be used for teacher optimization")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    source_hashes = numerical_source_hashes()
    reference = np.asarray(reference)
    if uniform_record is None:
        uniform_record, _ = evaluate_case(
            scene,
            cells,
            incidence_angle,
            "uniform",
            output / "uniform",
            device=device,
            reference=reference,
            source_hashes=source_hashes,
        )
    if not uniform_record.get("accepted"):
        raise ValueError("An accepted uniform baseline is required")
    uniform_dt = float(uniform_record["dt"])
    rows = []

    def objective(weights):
        if deadline is not None and time.time() >= deadline:
            raise _DeadlineReached
        index = len(rows)
        axes, repair = density_axes(weights, cells)
        record, reused = evaluate_case(
            scene,
            cells,
            incidence_angle,
            "learned",
            output / "trials" / f"trial_{index:04d}",
            device=device,
            reference=reference,
            selected_axes=axes,
            source_hashes=source_hashes,
        )
        valid = bool(record.get("accepted") and record.get("joint_scattering_loss") is not None)
        score = (
            float(record["joint_scattering_loss"]) * (uniform_dt / float(record["dt"])) ** 0.05
            if valid
            else 1e6
        )
        rows.append(
            {
                "index": index,
                "name": f"optimized_{index:04d}",
                "status": record["status"],
                "accepted": valid,
                "joint_scattering_loss": record.get("joint_scattering_loss"),
                "complex_relative_l2": record.get("complex_relative_l2"),
                "width_relative_l2": record.get("width_relative_l2"),
                "dt": record.get("dt"),
                "teacher_score": score if valid else None,
                "repair_fraction": repair,
                "weights": np.asarray(weights).tolist(),
                "reused": reused,
                "record": str(output / "trials" / f"trial_{index:04d}" / "record.json"),
            }
        )
        return score

    complete = True
    try:
        differential_evolution(
            objective,
            bounds=[(-PARAMETER_BOUND, PARAMETER_BOUND)] * (2 * KNOTS_PER_AXIS),
            init=_initial_population(scene, cells, seed),
            maxiter=evaluations // POPULATION - 1,
            mutation=(0.5, 1.0),
            recombination=0.7,
            seed=seed,
            polish=False,
            updating="immediate",
        )
    except _DeadlineReached:
        complete = False
    scored = [
        {
            "name": "uniform",
            "accepted": True,
            "joint_scattering_loss": uniform_record["joint_scattering_loss"],
            "dt": uniform_dt,
        },
        *[row for row in rows if row["accepted"]],
    ]
    ranked = rank_candidates(scored)
    raw_best = min(scored, key=lambda row: row["joint_scattering_loss"])
    uniform_loss = float(uniform_record["joint_scattering_loss"])
    report = {
        "schema_version": 1,
        "method": "seeded_differential_evolution_smooth_log_density_v1",
        "scene_id": scene["lineage_id"],
        "stage": scene["stage"],
        "cells": cells,
        "incidence_angle_rad": incidence_angle,
        "seed": seed,
        "requested_evaluations": evaluations,
        "completed_evaluations": len(rows),
        "complete": complete,
        "accepted_evaluations": sum(row["accepted"] for row in rows),
        "uniform_raw_accuracy_loss": uniform_loss,
        "uniform_dt": uniform_dt,
        "best_teacher": ranked[0]["name"],
        "best_raw_accuracy_candidate": raw_best["name"],
        "best_raw_accuracy_loss": raw_best["joint_scattering_loss"],
        "uniform_over_best_raw_accuracy_ratio": uniform_loss
        / max(float(raw_best["joint_scattering_loss"]), 1e-30),
        "reference_uncertainty_relative": reference_uncertainty,
        "penalty_sensitivity": penalty_sensitivity(scored),
        "reference_sha256": hashlib.sha256(reference.tobytes()).hexdigest(),
        "numerical_source_hashes": source_hashes,
        "optimizer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "trials": rows,
    }
    atomic_json(output / "optimization_report.json", report)
    return report


def promote_optimized_candidates(report, search_output, case_output, *, count=3):
    """Copy top distinct valid trials to fixed dataset slots with provenance."""
    search_output, case_output = Path(search_output), Path(case_output)
    valid = sorted(
        (row for row in report["trials"] if row["accepted"]),
        key=lambda row: (row["teacher_score"], row["index"]),
    )
    selected = []
    if report["complete"]:
        for row in valid:
            if all(
                np.linalg.norm(np.asarray(row["weights"]) - np.asarray(prior["weights"])) > 0.05
                for prior in selected
            ):
                selected.append(row)
            if len(selected) == count:
                break
    promotions = []
    for slot in range(count):
        name = f"optimized_{slot}"
        destination = case_output / name
        destination.mkdir(parents=True, exist_ok=True)
        if slot >= len(selected):
            status = "incomplete" if not report["complete"] else "no_valid_optimized_candidate"
            atomic_json(destination / "record.json", {"status": status, "accepted": False})
            promotions.append({"name": name, "status": status})
            continue
        row = selected[slot]
        source = search_output / "trials" / f"trial_{row['index']:04d}"
        shutil.copy2(source / "spectra.npz", destination / "spectra.npz")
        record = json.loads((source / "record.json").read_text())
        record["teacher_optimizer_trial"] = row["index"]
        record["teacher_optimizer_report"] = str(search_output / "optimization_report.json")
        atomic_json(destination / "record.json", record)
        promotions.append({"name": name, "status": "accepted", "trial_index": row["index"]})
    return promotions
