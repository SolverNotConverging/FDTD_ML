"""Budgeted black-box mesh optimization with checkpointed DE and optional Powell."""

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from time import perf_counter

import h5py
import numpy as np
from scipy.optimize import minimize

from ..mesh import AxisConstraints, Mesh, MeshInfeasibleError, MeshOptimizationError
from ..result import Result, json_text
from ..simulation import ConvergenceError
from ..solver.conformal import UnresolvedGeometryError
from ..storage import _write
from ..strategies import deterministic_density, mesh_id
from .common import (
    archive_directory,
    clone,
    errors,
    experiment_directory,
    experiment_key,
    identity,
    run_cached,
    simulation_description,
)
from .features import geometry_anchors


class _BudgetReached(Exception):
    pass


@dataclass
class Optimization:
    best: Result | None
    report: dict
    directory: Path

    @property
    def mesh(self):
        return None if self.best is None else self.best.mesh

    @classmethod
    def load(cls, directory):
        directory = archive_directory(directory, "optimization.h5")
        with h5py.File(directory / "optimization.h5", "r") as h:
            if h.attrs.get("kind") != "mesh-optimization":
                raise ValueError("Not a mesh optimization archive")
            report = json.loads(h["report"].asstr()[()])
        best = Result.load(directory / report["best_file"]) if report.get("best_file") else None
        return cls(best, report, directory)

    def plot_history(self, ax=None):
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(7, 4))
        x, y = [], []
        best = np.inf
        for i, t in enumerate(self.report["trials"]):
            if t.get("error") is not None:
                best = min(best, t["error"])
            if np.isfinite(best):
                x.append(i + 1)
                y.append(best)
        ax.semilogy(x, np.maximum(y, 1e-16), marker=".")
        ax.set(
            xlabel="Candidate evaluation",
            ylabel="Best complex relative L2 error",
            title="Best feasible mesh found",
        )
        return ax.figure


def _save(directory, report):
    def writer(h):
        h.create_dataset("report", data=json_text(report))
        h.create_dataset(
            "trials/json",
            data=np.array([json_text(t) for t in report["trials"]], dtype=h5py.string_dtype()),
        )

    _write(directory / "optimization.h5", "mesh-optimization", writer)


def optimize_mesh(
    sim,
    reference,
    *,
    cells,
    strategy="differential_evolution",
    directory,
    max_evaluations=80,
    max_seconds=600.0,
    controls=6,
    population=12,
    seed=0,
    constraints=None,
    feature_anchors=False,
    initial_mesh=None,
    projection_seconds=5.0,
    resume=True,
    progress=None,
):
    """Find a low-error tensor-product mesh at an exact total cell budget.

    Strategies: differential_evolution (checkpointed rand/1/bin, immediate updates)
    or powell (bounded SciPy search; resume is explicitly a warm restart).
    Budgets count proposals including failed/cached trials and both baselines.
    Time limits are checked between evaluations; an active solve may overrun them.
    The input simulation is not mutated. Inspect best, report and plot_history().
    feature_anchors=True forces geometry corner alignment for all proposals and
    baselines. By default only the physical layout/PML lines are fixed. Invalid
    conformal candidates remain infeasible; reference-only repairs are not used.
    initial_mesh accepts a validated geometry-aware grid. Its witness anchors
    constrain all same-budget projections, and it is scored as a baseline.
    """
    if strategy not in ("differential_evolution", "powell"):
        raise ValueError("strategy must be differential_evolution or powell")
    if not isinstance(feature_anchors, bool):
        raise ValueError("feature_anchors must be a boolean")
    if not reference.qualified or reference.result is None:
        raise ValueError("Mesh optimization requires a qualified numerical reference")
    if reference.report["case"] != experiment_key(sim):
        raise ValueError("Reference does not match this geometry, orientation or configuration")
    if initial_mesh is not None:
        if not isinstance(initial_mesh, Mesh):
            raise TypeError("initial_mesh must be a Mesh")
        construction = initial_mesh.metadata.get("geometry_aware", {})
        if construction.get("status") != "valid" or "witness_anchors" not in construction:
            raise ValueError("initial_mesh must be a validated geometry-aware mesh")
        expected_geometry = hashlib.sha256(
            json_text(sim.computational_geometry.as_dict()).encode()
        ).hexdigest()
        if construction.get("geometry_sha256") != expected_geometry:
            raise ValueError("initial_mesh belongs to a different exact geometry")
    if (
        any(
            isinstance(v, bool) or int(v) != v
            for v in (controls, population, max_evaluations, seed)
        )
        or controls < 2
        or population < 4
        or max_evaluations < 2
        or seed < 0
    ):
        raise ValueError("Invalid control count, population, evaluation budget or seed")
    if (
        not np.isfinite([max_seconds, projection_seconds]).all()
        or min(max_seconds, projection_seconds) <= 0
    ):
        raise ValueError("Time limits must be positive")
    cells = tuple(cells)
    if len(cells) != 2 or any(isinstance(n, bool) or int(n) != n or n < 1 for n in cells):
        raise ValueError("cells must be two positive integers")
    if initial_mesh is not None and (initial_mesh.Nx, initial_mesh.Ny) != cells:
        raise ValueError("cells must match the geometry-aware initial_mesh budget")
    if constraints is None:
        if initial_mesh is not None:
            exterior_h = max(p.thickness / p.cells for p in (sim.pml.x, sim.pml.y))
            constraints = AxisConstraints(
                max_spacing=max(exterior_h, construction["target_spacing"])
            )
        else:
            minimum = sim.wavelength / 160
            if sim.layout.scatterer_margin_cells is not None:
                bounds = sim.geometry.bounds
                for n, collar, span in zip(
                    cells, (sim.pml.x, sim.pml.y), (bounds[1] - bounds[0], bounds[3] - bounds[2])
                ):
                    interior = n - 2 * (
                        collar.cells
                        + sum(sim.layout.exterior_cells)
                        + sim.layout.scatterer_margin_cells
                    )
                    if interior > 0:
                        minimum = min(minimum, span / (2 * interior))
            constraints = AxisConstraints(min_spacing=minimum, max_spacing=sim.wavelength / 12)
    config = dict(
        case=experiment_key(sim),
        reference=reference.report["key"],
        reference_mesh=mesh_id(reference.result.mesh),
        cells=cells,
        strategy=strategy,
        controls=controls,
        population=population,
        seed=seed,
        constraints=asdict(constraints),
        feature_anchors=feature_anchors,
        initial_mesh=None if initial_mesh is None else dict(
            x=initial_mesh.x.tolist(),
            y=initial_mesh.y.tolist(),
            witness_anchors=construction["witness_anchors"],
        ),
        projection_seconds=projection_seconds,
        parameterization="log-density-v1",
        max_evaluations=max_evaluations,
        max_seconds=max_seconds,
    )
    description = dict(
        schema=1,
        kind="optimization",
        simulation=simulation_description(sim),
        optimizer=config,
        reference=dict(
            description=reference.report.get("description"),
            settings=reference.report.get("settings"),
            key=reference.report["key"],
            geometry=reference.result.geometry.as_dict(),
            configuration=reference.result.configuration,
            mesh=dict(x=reference.result.mesh.x.tolist(), y=reference.result.mesh.y.tolist()),
            frequencies=reference.result.frequencies.tolist(),
            angles=reference.result.angles.tolist(),
            far_field_sha256=hashlib.sha256(reference.result.far_field.tobytes()).hexdigest(),
        ),
    )
    directory = experiment_directory(directory, description)
    key = identity(description)
    rng = np.random.default_rng(seed)
    dimensions = 2 * (controls - 1)
    report = dict(
        key=key,
        description=description,
        config=config,
        trials=[],
        best_file=None,
        best_error=None,
        best_parameters=None,
        status="running",
        time_step_mode="native_cfl",
        population=None,
        energies=None,
        index=0,
        generation=-1,
        baseline_index=0,
        rng_state=rng.bit_generator.state,
        resume_mode="exact_population" if strategy == "differential_evolution" else "warm_restart",
        wall_seconds=0.0,
    )
    if (directory / "optimization.h5").exists():
        if not resume:
            raise FileExistsError("Use a fresh directory or resume=True")
        report = Optimization.load(directory).report
        if report["key"] != key:
            raise ValueError(
                f"Optimization archive in {directory} is inconsistent with its experiment.json"
            )
        rng.bit_generator.state = report["rng_state"]
    started = perf_counter()
    previous_seconds = report["wall_seconds"]
    reference_result = reference.result
    anchors = geometry_anchors(sim.computational_geometry) if feature_anchors else None
    if initial_mesh is not None:
        witness = construction["witness_anchors"]
        anchors = tuple(
            np.unique(np.r_[w, extra])
            for w, extra in zip(witness, anchors if anchors is not None else ([], []))
        )

    def save():
        report["rng_state"] = rng.bit_generator.state
        report["wall_seconds"] = previous_seconds + perf_counter() - started
        _save(directory, report)

    def notify():
        if progress and report["trials"]:
            progress(
                dict(
                    report["trials"][-1],
                    number=len(report["trials"]),
                    best_error=report["best_error"],
                )
            )

    def budget():
        if len(report["trials"]) >= max_evaluations or perf_counter() - started >= max_seconds:
            raise _BudgetReached

    def density(z):
        q = np.asarray(z).reshape(2, controls - 1)
        return tuple(
            np.exp(np.interp(np.linspace(0, 1, 128), np.linspace(0, 1, controls), np.r_[a, 0.0]))
            for a in q
        )

    def evaluate(z=None, baseline=None, check_budget=True):
        if check_budget:
            budget()
        trial = dict(
            kind=baseline or strategy, parameters=None if z is None else np.asarray(z).tolist()
        )
        begin = perf_counter()
        candidate = clone(sim)
        try:
            if baseline == "geometry_aware":
                candidate.apply_mesh(initial_mesh)
            elif baseline:
                candidate.apply_mesh(
                    baseline,
                    cells=cells,
                    constraints=constraints,
                    anchors=anchors,
                    anchor_assignment="local",
                    time_limit=projection_seconds,
                )
            else:
                candidate.apply_mesh(
                    "density",
                    cells=cells,
                    density=density(z),
                    anchors=anchors,
                    anchor_assignment="local",
                    constraints=constraints,
                    time_limit=projection_seconds,
                )
            result, path, cached = run_cached(candidate, directory / "runs")
            metric = errors(result, reference_result)
            trial.update(
                metric,
                status="feasible",
                cached=cached,
                mesh_id=mesh_id(result.mesh),
                result_file=str(path.relative_to(directory)),
                dt=result.diagnostics["dt"],
                gpu_ms=result.diagnostics["gpu_ms"],
                steps=result.diagnostics["Nt"],
            )
            if report["best_error"] is None or metric["error"] < report["best_error"]:
                report.update(
                    best_error=metric["error"],
                    best_file=trial["result_file"],
                    best_parameters=trial["parameters"],
                )
            loss = metric["error"]
        except (
            MeshInfeasibleError,
            MeshOptimizationError,
            UnresolvedGeometryError,
            ConvergenceError,
        ) as exc:
            trial.update(status=type(exc).__name__, message=str(exc), error=None)
            loss = 1e6
        trial["wall_seconds"] = perf_counter() - begin
        report["trials"].append(trial)
        return loss

    report["status"] = "running"
    try:
        baselines = (
            ("geometry_aware", "uniform", "deterministic")
            if initial_mesh is not None
            else ("uniform", "deterministic")
        )
        while report["baseline_index"] < len(baselines):
            evaluate(baseline=baselines[report["baseline_index"]])
            report["baseline_index"] += 1
            save()
            notify()
        if strategy == "differential_evolution":
            if report["population"] is None:
                pop = rng.uniform(-0.3, 0.3, (population, dimensions))
                pop[0] = 0
                controls0 = []
                for rho in deterministic_density(sim.computational_geometry):
                    values = np.log(
                        np.interp(np.linspace(0, 1, controls), np.linspace(0, 1, len(rho)), rho)
                    )
                    controls0.extend(np.clip(values[:-1] - values[-1], -1.5, 1.5))
                pop[1] = controls0
                for j, strength in enumerate((0.2, 0.4, 0.7), start=2):
                    if j >= population:
                        break
                    shape = strength * np.exp(-(((np.linspace(0, 1, controls) - 0.5) / 0.2) ** 2))
                    pop[j] = np.tile(shape[:-1] - shape[-1], 2)
                report["population"], report["energies"] = pop.tolist(), [None] * population
                save()
            pop = np.array(report["population"])
            energy = np.array([np.inf if v is None else v for v in report["energies"]])
            while True:
                budget()
                i = report["index"]
                if report["generation"] < 0:
                    trial = pop[i].copy()
                else:
                    others = np.delete(np.arange(population), i)
                    a, b, c = rng.choice(others, 3, replace=False)
                    mutant = np.clip(pop[a] + 0.7 * (pop[b] - pop[c]), -1.5, 1.5)
                    mask = rng.random(dimensions) < 0.8
                    mask[rng.integers(dimensions)] = True
                    trial = np.where(mask, mutant, pop[i])
                value = evaluate(trial, check_budget=False)
                if value <= energy[i]:
                    pop[i], energy[i] = trial, value
                report["index"] = (i + 1) % population
                if report["index"] == 0:
                    report["generation"] += 1
                report["population"] = pop.tolist()
                report["energies"] = [None if not np.isfinite(v) else float(v) for v in energy]
                save()
                notify()
        else:

            def objective(z):
                value = evaluate(z)
                save()
                notify()
                return value

            x0 = report["best_parameters"] or np.zeros(dimensions)
            answer = minimize(
                objective,
                x0,
                method="Powell",
                bounds=[(-1.5, 1.5)] * dimensions,
                options=dict(maxfev=max_evaluations - len(report["trials"]), xtol=0.02, ftol=1e-3),
            )
            report["optimizer_message"] = str(answer.message)
            report["status"] = "optimizer_finished"
    except _BudgetReached:
        report["status"] = (
            "evaluation_limit" if len(report["trials"]) >= max_evaluations else "time_limit"
        )
    except BaseException:
        report["status"] = "interrupted_or_failed"
        save()
        raise
    save()
    best = Result.load(directory / report["best_file"]) if report["best_file"] else None
    # Qualification is separate from search and retained in the archive.
    if best is not None:
        stop = sim.settings.stop
        stricter = clone(
            sim,
            stop=replace(
                stop, rtol=stop.rtol * 0.1, atol=stop.atol * 0.1, field_tol=stop.field_tol * 0.1
            ),
        )
        stricter.apply_mesh(best.mesh)
        try:
            check, path, cached = run_cached(stricter, directory / "validation")
            report["validation"] = dict(
                **errors(check, reference_result),
                change=errors(check, best),
                cached=cached,
                result_file=str(path.relative_to(directory)),
            )
        except ConvergenceError as exc:
            report["validation"] = dict(status="unconverged", message=str(exc))
        if "change" in report.get("validation", {}):
            change = report["validation"]["change"]
            settings = reference.report["settings"]
            report["validation"]["passed"] = (
                change["error"] <= settings["rtol"]
                and change["worst_frequency"] <= settings["worst_rtol"]
            )
        report["observed_reference_difference"] = reference.report.get(
            "observed_reference_difference"
        )
    else:
        report["status"] = "no_feasible_mesh"
    save()
    return Optimization(best, report, directory)
