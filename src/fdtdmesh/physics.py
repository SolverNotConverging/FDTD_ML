"""Stage-5 real-FDTD candidate search, Pareto selection and target distillation."""

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from torch.utils.data import Dataset

from fdtdmesh.data.schema import SceneSpec, provenance, read_manifest
from fdtdmesh.evaluation.metrics import sample_observables, spectrum
from fdtdmesh.evaluation.pipeline import (
    EvaluationConfig,
    converge_reference,
    grids,
    heuristic_density,
    metrics,
    run_scene,
    tail_diagnostic,
)
from fdtdmesh.mesh import MESH_POLICY, projected_density
from fdtdmesh.ml import conditioning, load_model, pool_axes, rasterize
from fdtdmesh.training import (
    TrainingConfig,
    load_teacher_targets,
    square_budgets,
    train_model,
)

PHYSICS_TARGET_VERSION = 1


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


@dataclass(frozen=True)
class SearchConfig:
    beta: float = 0.02
    physics_weight: float = 0.5
    perturbations: int = 2
    perturbation_scale: float = 0.35
    perturbation_smoothing: float = 4.0
    mixture_fraction: float = 0.5
    seed: int = 2026

    def __post_init__(self):
        for name in (
            "beta",
            "physics_weight",
            "perturbation_scale",
            "perturbation_smoothing",
            "mixture_fraction",
        ):
            value = getattr(self, name)
            if not np.isfinite(value):
                raise ValueError(f"Search {name} must be finite")
        if self.beta < 0 or not 0 <= self.physics_weight <= 1 or self.perturbation_scale < 0:
            raise ValueError("Invalid search weights")
        if self.perturbation_smoothing <= 0 or not 0 <= self.mixture_fraction <= 1:
            raise ValueError("Invalid perturbation smoothing or mixture fraction")
        if (
            isinstance(self.perturbations, bool)
            or int(self.perturbations) != self.perturbations
            or self.perturbations < 0
        ):
            raise ValueError("Perturbations must be a nonnegative integer")
        if isinstance(self.seed, bool) or int(self.seed) != self.seed or self.seed < 0:
            raise ValueError("Seed must be a nonnegative integer")


def _normalize(density):
    density = np.asarray(density, dtype=np.float64)
    if (
        density.ndim != 1
        or not len(density)
        or not np.isfinite(density).all()
        or np.any(density <= 0)
    ):
        raise ValueError("Candidate densities must be positive finite vectors")
    return density / density.mean()


def _predict_density(spec, budget, model, metadata, device):
    simulation = spec.build(budget)
    inputs = rasterize(simulation, metadata["raster_shape"], spec.f_max)
    condition = conditioning(simulation, *budget, spec.f_max, spec.f_min)
    with torch.inference_mode():
        logits = model(
            torch.from_numpy(inputs[None]).to(device),
            torch.from_numpy(condition[None]).to(device),
        )
        rho_x, rho_y = pool_axes(logits, metadata["pooling"]["alpha"])
    return _normalize(rho_x[0].cpu().numpy()), _normalize(rho_y[0].cpu().numpy())


def candidate_densities(
    spec,
    budget,
    model,
    metadata,
    config,
    device,
    *,
    baseline_model=None,
    baseline_metadata=None,
):
    simulation = spec.build(budget)
    heuristic = tuple(_normalize(v) for v in heuristic_density(simulation, spec.raster_shape))
    cnn = _predict_density(spec, budget, model, metadata, device)
    uniform = (np.ones_like(cnn[0]), np.ones_like(cnn[1]))
    fraction = config.mixture_fraction

    def mix(a, b):
        return tuple(_normalize((1 - fraction) * x + fraction * y) for x, y in zip(a, b))

    candidates = [
        {"name": "uniform", "density": uniform, "parameters": {}},
        {"name": "heuristic", "density": heuristic, "parameters": {}},
        {"name": "cnn", "density": cnn, "parameters": {}},
        {
            "name": "uniform_heuristic_mix",
            "density": mix(uniform, heuristic),
            "parameters": {"heuristic_fraction": fraction},
        },
        {
            "name": "uniform_cnn_mix",
            "density": mix(uniform, cnn),
            "parameters": {"cnn_fraction": fraction},
        },
        {
            "name": "heuristic_cnn_mix",
            "density": mix(heuristic, cnn),
            "parameters": {"cnn_fraction": fraction},
        },
    ]
    if baseline_model is not None:
        candidates.append(
            {
                "name": "baseline_cnn",
                "density": _predict_density(
                    spec, budget, baseline_model, baseline_metadata, device
                ),
                "parameters": {},
            }
        )
    seed_material = f"{config.seed}:{spec.scene_id}:{budget[0]}:{budget[1]}".encode()
    seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    for index in range(config.perturbations):
        perturbed = []
        for density in cnn:
            noise = gaussian_filter1d(
                rng.standard_normal(len(density)), config.perturbation_smoothing, mode="reflect"
            )
            noise /= max(noise.std(), 1e-12)
            perturbed.append(_normalize(density * np.exp(config.perturbation_scale * noise)))
        candidates.append(
            {
                "name": f"cnn_perturb_{index}",
                "density": tuple(perturbed),
                "parameters": {
                    "scale": config.perturbation_scale,
                    "smoothing": config.perturbation_smoothing,
                    "seed": seed,
                    "index": index,
                },
            }
        )
    return candidates


def pareto_front(rows):
    valid = [row for row in rows if row.get("status") == "ok" and row.get("settled", True)]
    front = []
    for row in valid:
        dominated = any(
            other is not row
            and other["em_error"] <= row["em_error"]
            and other["cost_ratio"] <= row["cost_ratio"]
            and (other["em_error"] < row["em_error"] or other["cost_ratio"] < row["cost_ratio"])
            for other in valid
        )
        if not dominated:
            front.append(row["name"])
    return sorted(front)


def score_candidates(rows, beta):
    successful = [row for row in rows if row.get("status") == "ok"]
    uniform = next((row for row in successful if row["name"] == "uniform"), None)
    if uniform is None or uniform["diagnostics"]["cell_updates"] <= 0:
        raise ValueError("A successful uniform candidate is required for cost normalization")
    baseline_cost = uniform["diagnostics"]["cell_updates"]
    for row in successful:
        row["cost_ratio"] = row["diagnostics"]["cell_updates"] / baseline_cost
        row["em_error"] = max(row["metrics"]["waveform_l2_max"], row["metrics"]["spectrum_l2_max"])
        row["objective"] = row["em_error"] + beta * row["cost_ratio"]
    front = pareto_front(rows)
    eligible = [row for row in successful if row.get("settled", True) and row["name"] in front]
    if not eligible:
        raise ValueError("No settled successful candidate is eligible for selection")
    selected = min(eligible, key=lambda row: (row["objective"], row["em_error"], row["name"]))
    return selected, front


def _teacher_map(manifest_path, target_path):
    _, _, metadata, target_x, target_y = load_teacher_targets(manifest_path, target_path)
    return {
        (sample["scene_id"], tuple(sample["budget"])): (target_x[index], target_y[index])
        for index, sample in enumerate(metadata["samples"])
    }


def _read_reference(spec, directory, *, arrays_name):
    status_path, arrays_path = directory / "reference.json", directory / arrays_name
    scene_path = directory / "evaluated_scene.json"
    if not status_path.exists():
        return (
            {"status": "missing", "levels": [], "accepted_budget": None},
            spec,
            None,
            None,
            None,
        )
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if (
        status.get("scene_id", spec.scene_id) != spec.scene_id
        or status.get("scene_hash", spec.content_hash) != spec.content_hash
    ):
        raise ValueError(f"Reference identity differs for {spec.scene_id}")
    effective = SceneSpec.from_dict(json.loads(scene_path.read_text(encoding="utf-8")))
    if status["status"] != "converged":
        return status, effective, None, None, None
    if not arrays_path.exists():
        raise ValueError(f"Converged reference arrays are missing for {spec.scene_id}")
    with np.load(arrays_path) as arrays:
        times = arrays["times"].copy()
        frequencies = arrays["frequencies"].copy()
        waveforms = arrays["waveforms"].copy()
    if (
        times.ndim != 1
        or len(times) < 2
        or frequencies.ndim != 1
        or waveforms.shape[0] != len(times)
        or not np.isfinite(times).all()
        or not np.isfinite(frequencies).all()
        or not np.isfinite(waveforms).all()
        or not np.isclose(times[-1], effective.t_end)
    ):
        raise ValueError(f"Invalid reference arrays for {spec.scene_id}")
    return status, effective, times, frequencies, waveforms


def _reference(spec, directory, evaluation, reference_root=None):
    if reference_root is not None:
        return _read_reference(
            spec, Path(reference_root) / spec.scene_id, arrays_name="reference_latest.npz"
        )
    status_path, arrays_path = directory / "reference.json", directory / "reference.npz"
    scene_path = directory / "evaluated_scene.json"
    if status_path.exists():
        return _read_reference(spec, directory, arrays_name="reference.npz")
    status, latest = converge_reference(spec, evaluation)
    effective = replace(spec, t_end=status.get("duration", spec.t_end))
    _json(status_path, status)
    _json(scene_path, effective.to_dict())
    if status["status"] != "converged" or latest is None:
        return status, effective, None, None, None
    times, frequencies = grids(effective, evaluation)
    observations = latest[1]
    np.savez_compressed(
        arrays_path,
        times=times,
        frequencies=frequencies,
        waveforms=observations,
        x=latest[0].mesh.x,
        y=latest[0].mesh.y,
    )
    return status, effective, times, frequencies, observations


def _candidate_mesh(spec, budget, density, time_limit):
    simulation = spec.build(budget)
    return simulation.mesh_from_density(*density, time_limit=time_limit)


def search_physics_targets(
    manifest_path,
    teacher_targets,
    checkpoint,
    output,
    *,
    splits=("train", "validation"),
    limit=None,
    search_config=None,
    evaluation_config=None,
    baseline_checkpoint=None,
    references=None,
    device=None,
    scene_ids=None,
    budgets=None,
):
    search_config = search_config or SearchConfig()
    evaluation = evaluation_config or EvaluationConfig()
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    manifest, scenes = read_manifest(manifest_path)
    reference_identity = None
    if references is not None:
        references = Path(references)
        reference_run = references / "run.json"
        if not reference_run.exists():
            raise ValueError("Reference corpus is missing run.json")
        reference_identity = json.loads(reference_run.read_text(encoding="utf-8"))
        if reference_identity.get("dataset_id") != manifest["dataset_id"]:
            raise ValueError("Reference corpus belongs to a different dataset")
    if scene_ids is not None:
        scene_map = {scene.scene_id: scene for scene in scenes}
        if not all(scene_id in scene_map for scene_id in scene_ids):
            raise ValueError("Every requested scene ID must exist in the manifest")
        selected_scenes = [scene_map[scene_id] for scene_id in scene_ids]
    else:
        selected_scenes = []
        for split in splits:
            subset = [scene for scene in scenes if scene.split == split]
            selected_scenes.extend(subset if limit is None else subset[:limit])
    if not selected_scenes:
        raise ValueError("No scenes selected for physics search")
    budget_override = square_budgets(budgets)
    teacher = _teacher_map(manifest_path, teacher_targets)
    model, metadata = load_model(checkpoint, device=device)
    baseline_model = baseline_metadata = None
    if baseline_checkpoint is not None:
        baseline_model, baseline_metadata = load_model(baseline_checkpoint, device=device)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    identity = {
        "physics_target_version": PHYSICS_TARGET_VERSION,
        "dataset_id": manifest["dataset_id"],
        "teacher_targets_sha256": _sha256(teacher_targets),
        "checkpoint_sha256": _sha256(checkpoint),
        "baseline_checkpoint_sha256": (
            None if baseline_checkpoint is None else _sha256(baseline_checkpoint)
        ),
        "reference_run_sha256": None if references is None else _sha256(reference_run),
        "splits": list(splits),
        "limit": limit,
        "scene_ids": scene_ids,
        "budget_override": (
            None if budget_override is None else [list(value) for value in budget_override]
        ),
        "search_config": asdict(search_config),
        "evaluation_config": asdict(evaluation),
        "mesh_policy": MESH_POLICY,
    }
    identity = json.loads(json.dumps(identity, allow_nan=False))
    identity_path = output / "run.json"
    if identity_path.exists():
        if json.loads(identity_path.read_text(encoding="utf-8")) != identity:
            raise ValueError("Physics search output belongs to a different run")
    else:
        _json(identity_path, identity)
    started = perf_counter()
    samples, target_x, target_y, reference_rows = [], [], [], []
    for spec in selected_scenes:
        scene_dir = output / spec.scene_id
        scene_dir.mkdir(exist_ok=True)
        status, effective, times, frequencies, reference = _reference(
            spec, scene_dir, evaluation, reference_root=references
        )
        reference_rows.append({"scene_id": spec.scene_id, **status})
        if status["status"] != "converged":
            print(f"{spec.scene_id}: {status['status']}; no physics targets", flush=True)
            continue
        for budget in budget_override or spec.budgets:
            key = (spec.scene_id, tuple(budget))
            if key not in teacher:
                print(f"{spec.scene_id} {budget}: no feasible teacher target", flush=True)
                continue
            sample_dir = scene_dir / f"{budget[0]}_{budget[1]}"
            sample_dir.mkdir(exist_ok=True)
            result_path = sample_dir / "result.json"
            target_path = sample_dir / "target.npz"
            if result_path.exists() and target_path.exists():
                row = json.loads(result_path.read_text(encoding="utf-8"))
                with np.load(target_path) as arrays:
                    tx, ty = arrays["target_x"].copy(), arrays["target_y"].copy()
                samples.append(row["sample"])
                target_x.append(tx)
                target_y.append(ty)
                continue
            candidates = candidate_densities(
                effective,
                budget,
                model,
                metadata,
                search_config,
                device,
                baseline_model=baseline_model,
                baseline_metadata=baseline_metadata,
            )
            rows, meshes = [], {}
            mesh_hashes = {}
            for candidate in candidates:
                row = {
                    "name": candidate["name"],
                    "parameters": candidate["parameters"],
                    "status": "ok",
                }
                try:
                    mesh = _candidate_mesh(
                        effective,
                        budget,
                        candidate["density"],
                        evaluation.meshing_time_limit,
                    )
                    mesh_hash = hashlib.sha256(mesh.x.tobytes() + mesh.y.tobytes()).hexdigest()
                    if mesh_hash in mesh_hashes:
                        original = next(r for r in rows if r["name"] == mesh_hashes[mesh_hash])
                        row.update(
                            duplicate_of=original["name"],
                            metrics=original["metrics"],
                            diagnostics=original["diagnostics"],
                            tail=original["tail"],
                            settled=original["settled"],
                        )
                    else:
                        result = run_scene(
                            effective,
                            budget,
                            evaluation,
                            strategy="mesh",
                            mesh_lines=(mesh.x, mesh.y),
                        )
                        observations = sample_observables(result, times)
                        tail = tail_diagnostic(effective, observations, evaluation)
                        row.update(
                            metrics=metrics(
                                observations, reference, times, frequencies, evaluation
                            ),
                            diagnostics=result.diagnostics,
                            tail=tail,
                            settled=tail["settled"],
                        )
                        np.savez_compressed(
                            sample_dir / f"{candidate['name']}.npz",
                            density_x=candidate["density"][0],
                            density_y=candidate["density"][1],
                            x=mesh.x,
                            y=mesh.y,
                            times=times,
                            frequencies=frequencies,
                            waveforms=observations,
                            spectra=spectrum(observations, times, frequencies),
                        )
                        mesh_hashes[mesh_hash] = candidate["name"]
                    meshes[candidate["name"]] = mesh
                except (ValueError, RuntimeError) as error:
                    row.update(status="failed", error=str(error), error_type=type(error).__name__)
                rows.append(row)
            try:
                selected, front = score_candidates(rows, search_config.beta)
            except ValueError as error:
                _json(
                    result_path,
                    {
                        "status": "failed",
                        "error": str(error),
                        "error_type": type(error).__name__,
                        "candidates": rows,
                    },
                )
                print(f"{spec.scene_id} {budget}: no eligible target: {error}", flush=True)
                continue
            selected_mesh = meshes[selected["name"]]
            px = projected_density(
                selected_mesh.x, spec.raster_shape[1], collar=effective.build(budget).pml.x
            )
            py = projected_density(
                selected_mesh.y, spec.raster_shape[0], collar=effective.build(budget).pml.y
            )
            prior_x, prior_y = teacher[key]
            weight = search_config.physics_weight
            tx, ty = (1 - weight) * prior_x + weight * px, (1 - weight) * prior_y + weight * py
            sample = {
                "scene_id": spec.scene_id,
                "scene_hash": spec.content_hash,
                "split": spec.split,
                "budget": list(budget),
                "selected": selected["name"],
                "pareto_front": front,
                "physics_weight": weight,
                "selected_em_error": selected["em_error"],
                "selected_cost_ratio": selected["cost_ratio"],
                "selected_objective": selected["objective"],
            }
            np.savez_compressed(target_path, target_x=tx, target_y=ty)
            _json(result_path, {"sample": sample, "candidates": rows})
            samples.append(sample)
            target_x.append(tx)
            target_y.append(ty)
            print(
                f"{spec.scene_id} {budget}: selected {selected['name']} "
                f"E={selected['em_error']:.4g} C={selected['cost_ratio']:.3f}",
                flush=True,
            )
    if samples:
        arrays_path = output / "physics_targets.npz"
        np.savez_compressed(
            arrays_path,
            target_x=np.asarray(target_x, dtype=np.float32),
            target_y=np.asarray(target_y, dtype=np.float32),
        )
        target_metadata = {
            "schema_version": 1,
            "physics_target_version": PHYSICS_TARGET_VERSION,
            "dataset_id": manifest["dataset_id"],
            "targets_sha256": _sha256(arrays_path),
            "mesh_policy": MESH_POLICY,
            "raster_shape": selected_scenes[0].raster_shape,
            "samples": samples,
            "search_identity": identity,
            "provenance": provenance(),
        }
        _json(output / "physics_targets.json", target_metadata)
    report = {
        **identity,
        "references": reference_rows,
        "accepted_references": sum(row["status"] == "converged" for row in reference_rows),
        "targets": len(samples),
        "elapsed_seconds": perf_counter() - started,
        "provenance": provenance(),
    }
    _json(output / "report.json", report)
    return report


def reblend_physics_targets(
    manifest_path,
    teacher_targets,
    source_targets,
    output_path,
    *,
    physics_weight,
):
    """Create a new target blend from cached physics targets without rerunning FDTD."""
    if not np.isfinite(physics_weight) or not 0 <= physics_weight <= 1:
        raise ValueError("physics_weight must be in [0,1]")
    manifest, _, metadata, source_x, source_y = load_physics_targets(manifest_path, source_targets)
    teacher = _teacher_map(manifest_path, teacher_targets)
    output_path = Path(output_path)
    metadata_path = output_path.with_suffix(".json")
    if output_path.exists() or metadata_path.exists():
        raise ValueError("Reblended target output already exists")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    target_x, target_y, samples = [], [], []
    for index, sample in enumerate(metadata["samples"]):
        key = (sample["scene_id"], tuple(sample["budget"]))
        if key not in teacher:
            raise ValueError(f"Teacher target is missing {key}")
        old_weight = sample.get("physics_weight")
        if old_weight is None or not 0 < old_weight <= 1:
            raise ValueError("Source physics targets require a positive recorded blend weight")
        prior_x, prior_y = teacher[key]

        def recover(source, prior):
            physics = (source - (1 - old_weight) * prior) / old_weight
            if not np.isfinite(physics).all() or np.min(physics) < -1e-6:
                raise ValueError("Cannot recover a valid physics target from the source blend")
            physics = np.maximum(physics, 0)
            physics /= physics.sum()
            return (1 - physics_weight) * prior + physics_weight * physics

        target_x.append(recover(source_x[index], prior_x))
        target_y.append(recover(source_y[index], prior_y))
        samples.append({**sample, "physics_weight": physics_weight})
    np.savez_compressed(
        output_path,
        target_x=np.asarray(target_x, dtype=np.float32),
        target_y=np.asarray(target_y, dtype=np.float32),
    )
    output_metadata = {
        "schema_version": 1,
        "physics_target_version": PHYSICS_TARGET_VERSION,
        "dataset_id": manifest["dataset_id"],
        "targets_sha256": _sha256(output_path),
        "mesh_policy": MESH_POLICY,
        "raster_shape": metadata["raster_shape"],
        "samples": samples,
        "physics_weight": physics_weight,
        "source_targets_sha256": _sha256(source_targets),
        "teacher_targets_sha256": _sha256(teacher_targets),
        "search_identity": metadata.get("search_identity"),
        "provenance": provenance(),
    }
    _json(metadata_path, output_metadata)
    return output_metadata


def select_validation_checkpoint(evaluations, checkpoints, output, *, beta=0.02):
    """Select a checkpoint from measured validation EM error and cell-update cost."""
    if len(evaluations) != len(checkpoints) or not evaluations:
        raise ValueError("Provide one checkpoint for every nonempty evaluation list")
    if not np.isfinite(beta) or beta < 0:
        raise ValueError("beta must be finite and nonnegative")
    records = []
    contract = None
    for evaluation_path, checkpoint_path in zip(evaluations, checkpoints):
        report = json.loads(Path(evaluation_path).read_text(encoding="utf-8"))
        current = (
            report.get("dataset_id"),
            report.get("reference_run_sha256"),
            report.get("split"),
        )
        if current[2] != "validation" or None in current:
            raise ValueError("Checkpoint selection requires validation evaluation reports")
        if contract is None:
            contract = current
        elif current != contract:
            raise ValueError("Validation evaluation reports use different contracts")
        checkpoint_sha = _sha256(checkpoint_path)
        if report.get("checkpoint_sha256") != checkpoint_sha:
            raise ValueError("Evaluation report does not match its checkpoint")
        rows = [row for row in report["rows"] if row["status"] == "ok"]
        keyed = {(row["scene_id"], tuple(row["budget"]), row["strategy"]): row for row in rows}
        pairs = []
        for scene_id, budget, strategy in keyed:
            if strategy != "distilled":
                continue
            uniform = keyed.get((scene_id, budget, "uniform"))
            if uniform is None or uniform["diagnostics"]["cell_updates"] <= 0:
                continue
            candidate = keyed[(scene_id, budget, "distilled")]
            error = max(
                candidate["metrics"]["waveform_l2_max"],
                candidate["metrics"]["spectrum_l2_max"],
            )
            cost_ratio = (
                candidate["diagnostics"]["cell_updates"] / uniform["diagnostics"]["cell_updates"]
            )
            pairs.append((error, cost_ratio))
        if not pairs:
            raise ValueError("Evaluation report has no measured distilled/uniform pairs")
        median_error = float(np.median([pair[0] for pair in pairs]))
        median_cost_ratio = float(np.median([pair[1] for pair in pairs]))
        records.append(
            {
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": checkpoint_sha,
                "evaluation": str(evaluation_path),
                "pairs": len(pairs),
                "median_em_error": median_error,
                "median_cost_ratio": median_cost_ratio,
                "objective": median_error + beta * median_cost_ratio,
            }
        )
    for row in records:
        row["pareto"] = not any(
            other is not row
            and other["median_em_error"] <= row["median_em_error"]
            and other["median_cost_ratio"] <= row["median_cost_ratio"]
            and (
                other["median_em_error"] < row["median_em_error"]
                or other["median_cost_ratio"] < row["median_cost_ratio"]
            )
            for other in records
        )
    selected = min(
        (row for row in records if row["pareto"]),
        key=lambda row: (row["objective"], row["median_em_error"], row["checkpoint_sha256"]),
    )
    result = {
        "schema_version": 1,
        "dataset_id": contract[0],
        "reference_run_sha256": contract[1],
        "split": contract[2],
        "beta": beta,
        "candidates": records,
        "selected_checkpoint": selected["checkpoint"],
        "selected_checkpoint_sha256": selected["checkpoint_sha256"],
        "provenance": provenance(),
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    _json(output, result)
    return result


def evaluate_physics_checkpoint(
    manifest_path,
    checkpoint,
    references,
    output,
    *,
    split="test_iid",
    baseline_checkpoint=None,
    limit=None,
    max_cell_updates=256_000_000_000,
    device=None,
    budgets=None,
):
    """Compare a distilled checkpoint with fixed baselines on an external corpus."""
    manifest, scenes = read_manifest(manifest_path)
    selected = [scene for scene in scenes if scene.split == split]
    if limit is not None:
        if isinstance(limit, bool) or int(limit) != limit or limit < 1:
            raise ValueError("limit must be a positive integer")
        selected = selected[: int(limit)]
    if not selected:
        raise ValueError("No scenes selected for physics evaluation")
    budget_override = square_budgets(budgets)
    references = Path(references)
    reference_run_path = references / "run.json"
    if not reference_run_path.exists():
        raise ValueError("Reference corpus is missing run.json")
    reference_run = json.loads(reference_run_path.read_text(encoding="utf-8"))
    if reference_run.get("dataset_id") != manifest["dataset_id"]:
        raise ValueError("Reference corpus belongs to a different dataset")
    evaluation = EvaluationConfig(**reference_run["config"])
    evaluation = replace(evaluation, max_cell_updates=max_cell_updates)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, metadata = load_model(checkpoint, device=device)
    baseline_model = baseline_metadata = None
    if baseline_checkpoint is not None:
        baseline_model, baseline_metadata = load_model(baseline_checkpoint, device=device)

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    identity = {
        "physics_target_version": PHYSICS_TARGET_VERSION,
        "dataset_id": manifest["dataset_id"],
        "checkpoint_sha256": _sha256(checkpoint),
        "baseline_checkpoint_sha256": (
            None if baseline_checkpoint is None else _sha256(baseline_checkpoint)
        ),
        "reference_run_sha256": _sha256(reference_run_path),
        "split": split,
        "limit": limit,
        "budget_override": (
            None if budget_override is None else [list(value) for value in budget_override]
        ),
        "max_cell_updates": max_cell_updates,
        "mesh_policy": MESH_POLICY,
    }
    identity_path = output / "run.json"
    if identity_path.exists():
        if json.loads(identity_path.read_text(encoding="utf-8")) != identity:
            raise ValueError("Physics evaluation output belongs to a different run")
    else:
        _json(identity_path, identity)

    rows, reference_status = [], []
    for spec in selected:
        status, effective, times, frequencies, reference = _reference(
            spec, output / spec.scene_id, evaluation, reference_root=references
        )
        reference_status.append({"scene_id": spec.scene_id, **status})
        if status["status"] != "converged":
            continue
        for budget in budget_override or spec.budgets:
            simulation = effective.build(budget)
            densities = {
                "uniform": (np.ones(spec.raster_shape[1]), np.ones(spec.raster_shape[0])),
                "heuristic": tuple(
                    _normalize(value) for value in heuristic_density(simulation, spec.raster_shape)
                ),
                "distilled": _predict_density(effective, budget, model, metadata, device),
            }
            if baseline_model is not None:
                densities["imitation"] = _predict_density(
                    effective, budget, baseline_model, baseline_metadata, device
                )
            for name, density in densities.items():
                row = {
                    "scene_id": spec.scene_id,
                    "scene_hash": spec.content_hash,
                    "split": spec.split,
                    "budget": list(budget),
                    "strategy": name,
                    "status": "ok",
                }
                try:
                    result = run_scene(
                        effective,
                        budget,
                        evaluation,
                        strategy="density",
                        density=density,
                    )
                    observations = sample_observables(result, times)
                    row.update(
                        metrics=metrics(observations, reference, times, frequencies, evaluation),
                        diagnostics=result.diagnostics,
                        tail=tail_diagnostic(effective, observations, evaluation),
                    )
                except (ValueError, RuntimeError) as error:
                    row.update(status="failed", error=str(error), error_type=type(error).__name__)
                rows.append(row)
        report = _physics_evaluation_report(identity, reference_status, rows)
        _json(output / "report.json", report)
    return _physics_evaluation_report(identity, reference_status, rows)


def _physics_evaluation_report(identity, reference_status, rows):
    summaries = {}
    for strategy in sorted({row["strategy"] for row in rows}):
        valid = [row for row in rows if row["strategy"] == strategy and row["status"] == "ok"]
        errors = [
            max(row["metrics"]["waveform_l2_max"], row["metrics"]["spectrum_l2_max"])
            for row in valid
        ]
        costs = [row["diagnostics"]["cell_updates"] for row in valid]
        summaries[strategy] = {
            "successful": len(valid),
            "median_em_error": None if not errors else float(np.median(errors)),
            "mean_em_error": None if not errors else float(np.mean(errors)),
            "median_cell_updates": None if not costs else float(np.median(costs)),
        }
    paired = None
    if "distilled" in summaries and "imitation" in summaries:
        keyed = {
            (row["scene_id"], tuple(row["budget"]), row["strategy"]): row
            for row in rows
            if row["status"] == "ok"
        }
        pairs = []
        for scene_id, budget, strategy in list(keyed):
            if strategy != "distilled" or (scene_id, budget, "imitation") not in keyed:
                continue
            distilled = keyed[(scene_id, budget, "distilled")]
            imitation = keyed[(scene_id, budget, "imitation")]
            distilled_error = max(
                distilled["metrics"]["waveform_l2_max"],
                distilled["metrics"]["spectrum_l2_max"],
            )
            imitation_error = max(
                imitation["metrics"]["waveform_l2_max"],
                imitation["metrics"]["spectrum_l2_max"],
            )
            pairs.append(distilled_error < imitation_error)
        paired = {"count": len(pairs), "distilled_lower_error": sum(pairs)}
    return {
        **identity,
        "references": reference_status,
        "accepted_references": sum(row["status"] == "converged" for row in reference_status),
        "rows": rows,
        "summary": summaries,
        "paired_vs_imitation": paired,
        "provenance": provenance(),
    }


def load_physics_targets(manifest_path, target_path):
    manifest, scenes = read_manifest(manifest_path)
    target_path = Path(target_path)
    metadata = json.loads(target_path.with_suffix(".json").read_text(encoding="utf-8"))
    if (
        metadata.get("schema_version") != 1
        or metadata.get("physics_target_version") != PHYSICS_TARGET_VERSION
        or metadata.get("dataset_id") != manifest["dataset_id"]
        or metadata.get("mesh_policy") != MESH_POLICY
        or metadata.get("targets_sha256") != _sha256(target_path)
    ):
        raise ValueError("Physics targets do not match the dataset or mesh policy")
    with np.load(target_path) as arrays:
        target_x, target_y = arrays["target_x"].copy(), arrays["target_y"].copy()
    if len(target_x) != len(metadata["samples"]) or len(target_y) != len(metadata["samples"]):
        raise ValueError("Physics target count mismatch")
    scene_map = {scene.scene_id: scene for scene in scenes}
    return manifest, scene_map, metadata, target_x, target_y


class PhysicsDataset(Dataset):
    def __init__(self, manifest_path, target_path, split):
        manifest, scene_map, metadata, target_x, target_y = load_physics_targets(
            manifest_path, target_path
        )
        self.dataset_id = manifest["dataset_id"]
        self.target_id = metadata["targets_sha256"]
        self.raster_shape = tuple(metadata["raster_shape"])
        self.records = []
        for index, sample in enumerate(metadata["samples"]):
            if sample["split"] != split:
                continue
            scene, budget = scene_map[sample["scene_id"]], sample["budget"]
            if scene.content_hash != sample["scene_hash"]:
                raise ValueError("Physics sample scene mismatch")
            simulation = scene.build(budget)
            self.records.append(
                {
                    "scene": scene,
                    "budget": budget,
                    "raster": torch.from_numpy(
                        rasterize(simulation, self.raster_shape, scene.f_max)
                    ),
                    "condition": torch.from_numpy(
                        conditioning(simulation, *budget, scene.f_max, scene.f_min)
                    ),
                    "target_x": torch.from_numpy(target_x[index]),
                    "target_y": torch.from_numpy(target_y[index]),
                    "collar": torch.tensor(
                        [
                            simulation.pml.x.thickness / simulation.Lx,
                            simulation.pml.y.thickness / simulation.Ly,
                        ],
                        dtype=torch.float32,
                    ),
                }
            )
        if not self.records:
            raise ValueError(f"No {split} physics samples")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        return {key: value for key, value in record.items() if key not in ("scene", "budget")} | {
            "index": index
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    search = sub.add_parser("search")
    search.add_argument("--manifest", required=True)
    search.add_argument("--teacher-targets", required=True)
    search.add_argument("--checkpoint", required=True)
    search.add_argument("--baseline-checkpoint")
    search.add_argument("--references", required=True)
    search.add_argument("--output", required=True)
    search.add_argument("--splits", nargs="+", default=["train", "validation"])
    search.add_argument("--limit", type=int)
    search.add_argument("--scenes", nargs="+")
    search.add_argument("--budgets", nargs="+", type=int)
    search.add_argument("--beta", type=float, default=0.02)
    search.add_argument("--physics-weight", type=float, default=0.5)
    search.add_argument("--perturbations", type=int, default=2)
    search.add_argument("--levels", type=int, nargs="+", default=[64, 128, 256, 512, 1024])
    search.add_argument("--max-cell-updates", type=int, default=256_000_000_000)
    search.add_argument("--device")
    distill = sub.add_parser("distill")
    distill.add_argument("--manifest", required=True)
    distill.add_argument("--targets", required=True)
    distill.add_argument("--initial-checkpoint", required=True)
    distill.add_argument("--output", required=True)
    distill.add_argument("--epochs", type=int, default=20)
    distill.add_argument("--batch-size", type=int, default=4)
    distill.add_argument("--learning-rate", type=float, default=3e-4)
    distill.add_argument("--repair-weight", type=float, default=0.05)
    distill.add_argument("--repair-every", type=int, default=4)
    distill.add_argument("--projection-samples", type=int, default=8)
    distill.add_argument("--seed", type=int, default=2026)
    distill.add_argument("--patience", type=int)
    distill.add_argument("--min-delta", type=float, default=0.0)
    distill.add_argument("--device")
    distill.add_argument("--resume")
    distill.add_argument("--allow-dirty", action="store_true")
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--manifest", required=True)
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--baseline-checkpoint")
    evaluate.add_argument("--references", required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--split", default="test_iid")
    evaluate.add_argument("--limit", type=int)
    evaluate.add_argument("--budgets", nargs="+", type=int)
    evaluate.add_argument("--max-cell-updates", type=int, default=256_000_000_000)
    evaluate.add_argument("--device")
    reblend = sub.add_parser("reblend")
    reblend.add_argument("--manifest", required=True)
    reblend.add_argument("--teacher-targets", required=True)
    reblend.add_argument("--source-targets", required=True)
    reblend.add_argument("--output", required=True)
    reblend.add_argument("--physics-weight", type=float, required=True)
    select = sub.add_parser("select")
    select.add_argument("--evaluations", nargs="+", required=True)
    select.add_argument("--checkpoints", nargs="+", required=True)
    select.add_argument("--output", required=True)
    select.add_argument("--beta", type=float, default=0.02)
    args = parser.parse_args()
    if args.command == "search":
        search_physics_targets(
            args.manifest,
            args.teacher_targets,
            args.checkpoint,
            args.output,
            splits=args.splits,
            limit=args.limit,
            search_config=SearchConfig(
                beta=args.beta,
                physics_weight=args.physics_weight,
                perturbations=args.perturbations,
            ),
            evaluation_config=EvaluationConfig(
                reference_levels=tuple(args.levels), max_cell_updates=args.max_cell_updates
            ),
            baseline_checkpoint=args.baseline_checkpoint,
            references=args.references,
            device=args.device,
            scene_ids=args.scenes,
            budgets=args.budgets,
        )
    elif args.command == "distill":
        _, initial = load_model(args.initial_checkpoint)
        config = TrainingConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            width=initial["architecture"]["width"],
            alpha=initial["pooling"]["alpha"],
            repair_weight=args.repair_weight,
            repair_every=args.repair_every,
            projection_samples=args.projection_samples,
            seed=args.seed,
            early_stopping_patience=args.patience,
            early_stopping_min_delta=args.min_delta,
        )
        train_model(
            args.manifest,
            args.targets,
            args.output,
            config=config,
            device=args.device,
            resume=args.resume,
            allow_dirty=args.allow_dirty,
            dataset_factory=PhysicsDataset,
            initial_checkpoint=args.initial_checkpoint,
            training_kind="physics_distillation",
            target_version=PHYSICS_TARGET_VERSION,
        )
    elif args.command == "evaluate":
        evaluate_physics_checkpoint(
            args.manifest,
            args.checkpoint,
            args.references,
            args.output,
            split=args.split,
            baseline_checkpoint=args.baseline_checkpoint,
            limit=args.limit,
            max_cell_updates=args.max_cell_updates,
            device=args.device,
            budgets=args.budgets,
        )
    elif args.command == "reblend":
        reblend_physics_targets(
            args.manifest,
            args.teacher_targets,
            args.source_targets,
            args.output,
            physics_weight=args.physics_weight,
        )
    else:
        select_validation_checkpoint(
            args.evaluations,
            args.checkpoints,
            args.output,
            beta=args.beta,
        )


if __name__ == "__main__":
    main()
