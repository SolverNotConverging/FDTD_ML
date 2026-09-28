"""Budgeted mesh optimization with feasible-local and differential-evolution search."""

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from time import perf_counter

import h5py
import numpy as np

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
    physical_key,
    run_cached,
    simulation_description,
)
from .feasible import FeasibleSettings, FeasibleSpace, search_statistics
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


def _optimize_mesh(
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
    local_settings=None,
    validate_winner=True,
    max_solves=None,
    projection_seconds=5.0,
    resume=True,
    progress=None,
):
    """Find a low-error tensor-product mesh at an exact total cell budget.

    Strategies: differential_evolution (checkpointed rand/1/bin) or feasible_local
    (seed-relative, exact population/RNG resume).
    Budgets count proposals including failed/cached trials and both baselines.
    Time limits are checked between evaluations; an active solve may overrun them.
    The input simulation is not mutated. Inspect best, report and plot_history().
    feature_anchors=True forces geometry corner alignment for all proposals and
    baselines. By default only the physical layout/PML lines are fixed. Invalid
    conformal candidates remain infeasible; reference-only repairs are not used.
    initial_mesh accepts a validated geometry-aware grid. Its witness anchors
    constrain all same-budget projections, and it is scored as a baseline.
    feasible_local requires that seed, preserves witness indices, and accepts
    FeasibleSettings through local_settings. max_solves optionally limits unique
    candidate FDTD evaluations, including cached hits and failed convergence but
    excluding final stricter validation; max_evaluations still limits proposals.
    """
    if strategy not in ("differential_evolution", "feasible_local"):
        raise ValueError("strategy must be differential_evolution or feasible_local")
    if strategy == "feasible_local":
        if initial_mesh is None:
            raise ValueError("feasible_local requires a geometry-aware initial_mesh")
        local_settings = local_settings or FeasibleSettings()
        if not isinstance(local_settings, FeasibleSettings):
            raise TypeError("local_settings must be FeasibleSettings")
    elif local_settings is not None:
        raise ValueError("local_settings is only used by feasible_local")
    if max_solves is not None and (
        isinstance(max_solves, bool)
        or not np.isfinite(max_solves)
        or int(max_solves) != max_solves
        or max_solves < 1
    ):
        raise ValueError("max_solves must be a positive integer")
    if not isinstance(feature_anchors, bool):
        raise ValueError("feature_anchors must be a boolean")
    if not reference.qualified or reference.result is None:
        raise ValueError("Mesh optimization requires a qualified numerical reference")
    if reference.report["case"] != experiment_key(sim):
        if reference.report.get("physical_case") != physical_key(sim):
            raise ValueError("Reference does not match this geometry, orientation or configuration")
    if initial_mesh is not None:
        if not isinstance(initial_mesh, Mesh):
            raise TypeError("initial_mesh must be a Mesh")
        validation = clone(sim)
        validation._apply_mesh(initial_mesh)
        construction = validation.mesh.metadata["preparation"]
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
                max_spacing=max(
                    exterior_h,
                    float(np.diff(initial_mesh.x).max()),
                    float(np.diff(initial_mesh.y).max()),
                )
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
        initial_mesh=None
        if initial_mesh is None
        else dict(
            x=initial_mesh.x.tolist(),
            y=initial_mesh.y.tolist(),
            witness_anchors=construction["witness_anchors"],
        ),
        projection_seconds=projection_seconds,
        parameterization="log-density-v1",
        max_evaluations=max_evaluations,
        max_seconds=max_seconds,
        validate_winner=validate_winner,
    )
    if strategy == "feasible_local":
        config.update(
            local_settings=asdict(local_settings), parameterization="seed-relative-fixed-indices-v1"
        )
    if max_solves is not None:
        config["max_solves"] = max_solves
    description = dict(
        schema=1,
        kind="optimization",
        simulation=simulation_description(sim),
        optimizer=config,
        optimizer_implementation=identity(
            {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in Path(__file__).parent.glob("*.py")
            }
        ),
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
    space = (
        FeasibleSpace(sim, initial_mesh, constraints, anchors, projection_seconds)
        if strategy == "feasible_local"
        else None
    )
    report.setdefault("solve_meshes", [])
    if space is not None:
        report.setdefault("feasible_state", dict(radius=local_settings.radius, pool=[]))
        report["resume_mode"] = "exact_feasible_state"
        report["adaptivity"] = dict(
            method="fixed-index-local-search",
            free_lines=[len(s.free) for s in space.axes],
            fixed_indices=[s.fixed.tolist() for s in space.axes],
            global_minimum_budget_certified=False,
        )

    def save():
        if space is not None:
            report["search_statistics"] = search_statistics(report["trials"])
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
        if (
            len(report["trials"]) >= max_evaluations
            or perf_counter() - started >= max_seconds
            or (max_solves is not None and len(report["solve_meshes"]) >= max_solves)
        ):
            raise _BudgetReached

    def density(z):
        q = np.asarray(z).reshape(2, controls - 1)
        return tuple(
            np.exp(np.interp(np.linspace(0, 1, 128), np.linspace(0, 1, controls), np.r_[a, 0.0]))
            for a in q
        )

    def evaluate(z=None, baseline=None, check_budget=True, proposed_mesh=None, proposal=None):
        if check_budget:
            budget()
        trial = dict(
            kind=baseline or strategy, parameters=None if z is None else np.asarray(z).tolist()
        )
        if proposal is not None:
            trial["proposal"] = proposal
        begin = perf_counter()
        candidate = clone(sim)
        try:
            if proposed_mesh is not None:
                candidate._apply_mesh(proposed_mesh)
            elif baseline == "geometry_aware":
                candidate._apply_mesh(initial_mesh)
            elif baseline:
                candidate._apply_mesh(
                    baseline,
                    cells=cells,
                    constraints=constraints,
                    anchors=anchors,
                    anchor_assignment="local",
                    time_limit=projection_seconds,
                )
            else:
                candidate._apply_mesh(
                    "density",
                    cells=cells,
                    density=density(z),
                    anchors=anchors,
                    anchor_assignment="local",
                    constraints=constraints,
                    time_limit=projection_seconds,
                )
            identifier = mesh_id(candidate.mesh)
            if proposed_mesh is not None and identifier in report["solve_meshes"]:
                trial.update(status="duplicate", mesh_id=identifier, error=None)
                trial["wall_seconds"] = perf_counter() - begin
                report["trials"].append(trial)
                return 1e6
            if identifier not in report["solve_meshes"]:
                report["solve_meshes"].append(identifier)
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
                boundary=result.diagnostics.get("boundary", {}),
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
        elif strategy == "feasible_local":
            state = report["feasible_state"]
            if not state["pool"]:
                initial = next(t for t in report["trials"] if t["kind"] == "geometry_aware")
                if initial["status"] != "feasible":
                    raise MeshInfeasibleError(
                        "The geometry-aware seed must converge before local search"
                    )
                state["pool"] = [
                    dict(
                        x=initial_mesh.x.tolist(),
                        y=initial_mesh.y.tolist(),
                        error=initial["error"],
                        mesh_id=mesh_id(initial_mesh),
                    )
                ]
            while True:
                budget()
                pool = state["pool"]
                parent_index = (
                    int(rng.integers(len(pool))) if rng.random() < local_settings.exploration else 0
                )
                entry = pool[parent_index]
                parent = Mesh(entry["x"], entry["y"])
                begin = perf_counter()
                mesh, proposal = space.propose(
                    parent, rng, state["radius"], controls, local_settings
                )
                proposal["parent_mesh_id"] = entry["mesh_id"]
                proposal["seconds"] = perf_counter() - begin
                if mesh is None:
                    report["trials"].append(
                        dict(
                            kind=strategy,
                            status="proposal_rejected",
                            parameters=None,
                            error=None,
                            proposal=proposal,
                            wall_seconds=proposal["seconds"],
                        )
                    )
                    value = 1e6
                else:
                    value = evaluate(proposed_mesh=mesh, proposal=proposal, check_budget=False)
                    report["trials"][-1]["wall_seconds"] += proposal["seconds"]
                accepted = report["trials"][-1]["status"] == "feasible"
                if accepted:
                    pool.append(
                        dict(
                            x=mesh.x.tolist(), y=mesh.y.tolist(), error=value, mesh_id=mesh_id(mesh)
                        )
                    )
                    pool.sort(key=lambda item: item["error"])
                    if len(pool) > population:
                        # Keep good solutions plus geometrically diverse parents.
                        good = pool[: max(1, population // 2)]
                        rest = pool[len(good) :]
                        best_mesh = Mesh(good[0]["x"], good[0]["y"])
                        rest.sort(
                            key=lambda item: space.movement(Mesh(item["x"], item["y"]), best_mesh)[
                                "rms_cells"
                            ],
                            reverse=True,
                        )
                        pool[:] = good + rest[: population - len(good)]
                state["radius"] = float(
                    np.clip(
                        state["radius"] * (1.12 if accepted else 0.65),
                        local_settings.min_radius,
                        local_settings.max_radius,
                    )
                )
                save()
                notify()
    except _BudgetReached:
        report["status"] = (
            "evaluation_limit"
            if len(report["trials"]) >= max_evaluations
            else "solve_limit"
            if max_solves is not None and len(report["solve_meshes"]) >= max_solves
            else "time_limit"
        )
    except BaseException:
        report["status"] = "interrupted_or_failed"
        save()
        raise
    save()
    best = Result.load(directory / report["best_file"]) if report["best_file"] else None
    # Qualification is separate from search and retained in the archive.
    if best is not None and validate_winner:
        stop = sim.settings.stop
        stricter = clone(
            sim,
            stop=replace(
                stop, rtol=stop.rtol * 0.1, atol=stop.atol * 0.1, field_tol=stop.field_tol * 0.1
            ),
        )
        stricter._apply_mesh(best.mesh)
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
    elif best is None:
        report["status"] = "no_feasible_mesh"
    save()
    return Optimization(best, report, directory)


@dataclass(frozen=True)
class SearchSettings:
    max_evaluations: int = 80
    max_solves: int | None = None
    max_seconds: float = 600.0
    controls: int = 6
    population: int = 12
    seed: int = 0
    projection_seconds: float = 5.0
    validate_winner: bool = True

    def __post_init__(self):
        for name, low in (("max_evaluations", 2), ("controls", 2), ("population", 4), ("seed", 0)):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not np.isfinite(value)
                or int(value) != value
                or value < low
            ):
                raise ValueError(f"Invalid {name}")
        if self.max_solves is not None and (
            isinstance(self.max_solves, bool)
            or not np.isfinite(self.max_solves)
            or int(self.max_solves) != self.max_solves
            or self.max_solves < 1
        ):
            raise ValueError("max_solves must be a positive integer")
        if (
            not np.isfinite([self.max_seconds, self.projection_seconds]).all()
            or min(self.max_seconds, self.projection_seconds) <= 0
        ):
            raise ValueError("Time limits must be positive")
        if not isinstance(self.validate_winner, bool):
            raise TypeError("validate_winner must be boolean")


def optimize_mesh(
    sim,
    reference,
    *,
    cells,
    directory,
    strategy="feasible_local",
    settings=None,
    local_settings=None,
    initial_mesh=None,
    constraints=None,
    resume=True,
    progress=None,
):
    """Search a fixed total budget; optional winner check is reported separately."""
    if not reference.qualified or reference.result is None:
        raise ValueError("Mesh optimization requires a qualified numerical reference")
    settings = settings or SearchSettings()
    if not isinstance(settings, SearchSettings):
        raise TypeError("settings must be SearchSettings")
    if strategy == "feasible_local" and initial_mesh is None:
        prepared = clone(sim)
        prepared.apply_mesh("geometry_aware", cells=cells)
        initial_mesh = prepared.mesh
    return _optimize_mesh(
        sim,
        reference,
        cells=cells,
        directory=directory,
        strategy=strategy,
        initial_mesh=initial_mesh,
        local_settings=local_settings,
        constraints=constraints,
        resume=resume,
        progress=progress,
        **asdict(settings),
    )
