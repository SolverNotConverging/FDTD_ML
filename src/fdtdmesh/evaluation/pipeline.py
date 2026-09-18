"""CUDA reference refinement and identical-observable baseline comparisons."""

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from fdtdmesh.data.schema import provenance, read_manifest
from fdtdmesh.mesh import MESH_POLICY
from fdtdmesh.ml import ResUNet, load_model, rasterize, save_model

from .metrics import compare_observables, sample_observables, spectrum


@dataclass(frozen=True)
class EvaluationConfig:
    reference_levels: tuple = (64, 128, 256, 512, 1024)
    relative_tolerance: float = 0.02
    consecutive_passes: int = 2
    samples: int = 1025
    frequency_count: int = 12
    amplitude_floor: float = 1e-8
    phase_gate: float = 0.01
    max_cell_updates: int = 128_000_000_000
    max_history_bytes: int = 256_000_000
    max_field_bytes: int = 1_000_000_000
    meshing_time_limit: float = 30.0
    duration_multiplier: float = 1.0
    max_duration_extensions: int = 2
    tail_relative_tolerance: float | None = 0.01
    tail_fraction: float = 0.2
    samples_per_period: int = 16
    max_observation_samples: int = 65537
    max_frequency_samples: int = 8193

    def __post_init__(self):
        from fdtdmesh.mesh import cell_count

        levels = [cell_count(n) for n in self.reference_levels]
        object.__setattr__(self, "reference_levels", tuple(levels))
        if len(levels) < 2 or any(b != 2 * a for a, b in zip(levels, levels[1:])):
            raise ValueError("Reference levels must double successively")
        for name in (
            "consecutive_passes",
            "samples",
            "frequency_count",
            "max_cell_updates",
            "max_history_bytes",
            "max_field_bytes",
            "samples_per_period",
            "max_observation_samples",
            "max_frequency_samples",
        ):
            object.__setattr__(self, name, cell_count(getattr(self, name)))
        if self.samples < 3 or self.frequency_count < 2 or self.consecutive_passes >= len(levels):
            raise ValueError("Insufficient samples/frequencies/refinement levels")
        if not np.isfinite(
            [
                self.relative_tolerance,
                self.amplitude_floor,
                self.phase_gate,
                self.meshing_time_limit,
                self.duration_multiplier,
                self.tail_fraction,
            ]
        ).all():
            raise ValueError("Configuration must be finite")
        if (
            not 0 < self.relative_tolerance < 1
            or self.amplitude_floor <= 0
            or not 0 < self.phase_gate < 1
            or self.meshing_time_limit <= 0
            or self.duration_multiplier < 1
            or not 0 < self.tail_fraction < 0.5
            or self.samples_per_period < 4
        ):
            raise ValueError("Invalid evaluation tolerance or limits")
        if (
            isinstance(self.max_duration_extensions, bool)
            or int(self.max_duration_extensions) != self.max_duration_extensions
            or self.max_duration_extensions < 0
        ):
            raise ValueError("Duration extensions must be a nonnegative integer")
        object.__setattr__(self, "max_duration_extensions", int(self.max_duration_extensions))
        if self.tail_relative_tolerance is not None and (
            not np.isfinite(self.tail_relative_tolerance)
            or not 0 < self.tail_relative_tolerance < 1
        ):
            raise ValueError("Tail tolerance must be in (0,1), or None for fixed-window evaluation")


def heuristic_density(scene, raster_shape):
    # An explicit baseline, never an extra handcrafted channel supplied to the CNN.
    raster = rasterize(scene, raster_shape, scene.f_max)
    weight = np.sqrt(raster[0] * raster[7]) + raster[1] + 3 * raster[2]
    edge = np.zeros_like(weight)
    edge[:, 1:] += abs(np.diff(weight, axis=1))
    edge[1:, :] += abs(np.diff(weight, axis=0))
    weight = 1 + weight + 2 * edge
    return weight.mean(axis=0), weight.mean(axis=1)


def run_scene(
    spec,
    budget,
    config,
    *,
    strategy="uniform",
    checkpoint=None,
    reference=False,
    density=None,
    mesh_lines=None,
):
    s = spec.build(budget, reference=reference)
    field_bytes = (s.Nx + 1) * (s.Ny + 1) * np.dtype(s.dtype).itemsize * 30
    if field_bytes > config.max_field_bytes or s.Nx * s.Ny > config.max_cell_updates:
        raise RuntimeError("Reference/candidate resource limit exceeded before meshing")
    started = perf_counter()
    if reference:
        s.mesh_uniform()  # Never silently turn an unaligned reference into a nonuniform grid.
    elif strategy == "uniform":
        # Uniform *interior preference*, projected for exact anchors and fixed collars.
        s.mesh_from_density([1], [1], time_limit=config.meshing_time_limit)
    elif strategy == "heuristic":
        s.mesh_from_density(
            *heuristic_density(s, spec.raster_shape), time_limit=config.meshing_time_limit
        )
    elif strategy == "cnn":
        s.mesh_with_model(checkpoint, time_limit=config.meshing_time_limit)
    elif strategy == "density":
        if density is None or len(density) != 2:
            raise ValueError("Density strategy requires x/y densities")
        s.mesh_from_density(*density, time_limit=config.meshing_time_limit)
    elif strategy == "mesh":
        if mesh_lines is None or len(mesh_lines) != 2:
            raise ValueError("Mesh strategy requires x/y line arrays")
        s.set_mesh(*mesh_lines)
    else:
        raise ValueError("Unknown mesh strategy")
    meshing_seconds = perf_counter() - started
    from fdtdmesh.solver.coefficients import build_coefficients

    c = build_coefficients(s, s.mesh, dtype=s.dtype)
    nt = int(np.ceil(s.t_end / c.dt))
    nr = sum(receiver.get("samples", 1) for receiver in spec.receivers)
    history_bytes = nt * nr * 4 * np.dtype(s.dtype).itemsize
    if s.Nx * s.Ny * nt > config.max_cell_updates or history_bytes > config.max_history_bytes:
        raise RuntimeError(
            f"Reference/candidate resource limit exceeded before CUDA allocation: "
            f"{s.Nx * s.Ny * nt} cell updates (limit {config.max_cell_updates}), "
            f"{history_bytes} history bytes (limit {config.max_history_bytes})"
        )
    if spec.f_max > 0.5 / c.dt:
        raise ValueError("Solver timestep cannot sample requested frequency band")
    result = s.run()
    if not all(np.isfinite(field).all() for field in result.fields.values()):
        raise ValueError("Nonfinite final fields")
    result.diagnostics["meshing_wall_seconds"] = meshing_seconds
    result.diagnostics["estimated_receiver_history_bytes"] = history_bytes
    result.diagnostics["estimated_field_bytes"] = field_bytes
    return result


def grids(spec, config):
    samples = max(
        config.samples, int(np.ceil(spec.t_end * spec.f_max * config.samples_per_period)) + 1
    )
    frequency_count = max(
        config.frequency_count, int(np.ceil(2 * spec.t_end * (spec.f_max - spec.f_min))) + 1
    )
    if samples > config.max_observation_samples or frequency_count > config.max_frequency_samples:
        raise ValueError("Observation resource limit exceeded; increase explicit sampling limits")
    times = np.linspace(0, spec.t_end, samples)
    frequencies = np.linspace(spec.f_min, spec.f_max, frequency_count)
    if spec.f_max > 0.5 / (times[1] - times[0]):
        raise ValueError("Observation grid undersamples requested spectrum")
    return times, frequencies


def metrics(a, b, times, frequencies, config):
    return compare_observables(
        a,
        b,
        times,
        frequencies,
        amplitude_floor=config.amplitude_floor,
        phase_gate=config.phase_gate,
    )


def tail_diagnostic(spec, observations, config):
    n = max(1, int(np.ceil(len(observations) * config.tail_fraction)))
    ratios = np.sqrt(np.mean(observations[-n:] ** 2, axis=0)) / np.maximum(
        np.max(abs(observations), axis=0), config.amplitude_floor
    )

    # Tail tests are meaningful only after pulsed excitation has ended.
    def ended(source):
        width = source.get("width")
        width = 1 / spec.f_max if width is None else width
        delay = source.get("delay")
        delay = 4 * width if delay is None else delay
        return (
            source.get("waveform", "gaussian") in ("gaussian", "gaussian_sine")
            and delay + 4 * width <= (1 - config.tail_fraction) * spec.t_end
        )

    pulse_ended = all(ended(source) for source in spec.sources)
    return {
        "tail_rms_over_peak": float(ratios.max()),
        "pulse_ended_before_tail": pulse_ended,
        "settled": bool(
            pulse_ended
            and (
                config.tail_relative_tolerance is None
                or ratios.max() <= config.tail_relative_tolerance
            )
        ),
    }


def _converge_at_duration(spec, config, runner):
    times, frequencies = grids(spec, config)
    history = []
    previous = None
    consecutive = 0
    latest = None
    for level in config.reference_levels:
        try:
            result = runner(spec, [level, level], config, reference=True)
            observations = sample_observables(result, times)
            tail = tail_diagnostic(spec, observations, config)
            entry = {"budget": [level, level], "diagnostics": result.diagnostics, "tail": tail}
            latest = (result, observations)
            if config.tail_relative_tolerance is not None and not tail["settled"]:
                history.append(entry)
                return {
                    "status": "time_unsettled",
                    "levels": history,
                    "accepted_budget": None,
                }, latest
            if previous is not None:
                comparison = metrics(previous, observations, times, frequencies, config)
                passed = (
                    comparison["waveform_l2_max"] <= config.relative_tolerance
                    and comparison["spectrum_l2_max"] <= config.relative_tolerance
                )
                entry.update(comparison=comparison, passed=passed)
                consecutive = consecutive + 1 if passed else 0
            history.append(entry)
            latest = (result, observations)
            previous = observations
            if consecutive >= config.consecutive_passes:
                return {
                    "status": "converged",
                    "levels": history,
                    "accepted_budget": [level, level],
                }, latest
        except (ValueError, RuntimeError) as error:
            history.append(
                {"budget": [level, level], "error": str(error), "error_type": type(error).__name__}
            )
            return {"status": "failed", "levels": history, "accepted_budget": None}, latest
    return {"status": "nonconverged", "levels": history, "accepted_budget": None}, latest


def converge_reference(spec, config, *, runner=run_scene):
    attempts = []
    effective = replace(spec, t_end=spec.t_end * config.duration_multiplier)
    for extension in range(config.max_duration_extensions + 1):
        try:
            status, latest = _converge_at_duration(effective, config, runner)
        except ValueError as error:
            status, latest = (
                {"status": "failed", "levels": [], "accepted_budget": None, "error": str(error)},
                None,
            )
        attempts.append({"duration": effective.t_end, **status})
        if status["status"] != "time_unsettled" or extension == config.max_duration_extensions:
            return {**status, "duration": effective.t_end, "duration_history": attempts}, latest
        # Restart the spatial sequence so every refinement uses the same time window.
        effective = replace(effective, t_end=effective.t_end * 2)


def _json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def evaluate_dataset(
    manifest_path,
    output,
    *,
    config=None,
    split="test_iid",
    limit=None,
    checkpoint=None,
    demo_cnn=False,
):
    config = config or EvaluationConfig()
    manifest, scenes = read_manifest(manifest_path)
    if split != "all":
        scenes = [s for s in scenes if s.split == split]
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        scenes = scenes[:limit]
    if not scenes:
        raise ValueError("No scenes selected")
    if checkpoint is not None and demo_cnn:
        raise ValueError("Choose a supplied checkpoint or the explicitly untrained demo")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("Evaluation output must be empty to preserve prior results")
    if demo_cnn:
        torch.manual_seed(2026)
        checkpoint = output / "untrained_demo.pt"
        save_model(
            checkpoint,
            ResUNet(4),
            raster_shape=tuple(max(s.raster_shape[i] for s in scenes) for i in range(2)),
        )
    model_record = None
    if checkpoint is not None:
        _, metadata = load_model(checkpoint)
        if any(
            any(a < b for a, b in zip(metadata["raster_shape"], s.raster_shape)) for s in scenes
        ):
            raise ValueError(
                "Checkpoint raster is smaller than the dataset feature-resolution contract"
            )
        model_record = {
            "sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
            "metadata": metadata,
            "untrained": metadata["training_commit"] == "untrained",
        }
    report = {
        "schema_version": 1,
        "dataset_id": manifest["dataset_id"],
        "config": asdict(config),
        "mesh_policy": MESH_POLICY,
        "provenance": provenance(),
        "checkpoint": model_record,
        "selection": {"split": split, "limit": limit},
        "scenes": [],
        "candidates": [],
    }
    _json(output / "manifest.json", manifest)
    for spec in scenes:
        print(f"Reference {spec.scene_id}", flush=True)
        status, latest = converge_reference(spec, config)
        evaluated_spec = replace(spec, t_end=status.get("duration", spec.t_end))
        status.update(
            scene_id=spec.scene_id,
            scene_hash=spec.content_hash,
            split=spec.split,
            family=spec.family,
        )
        report["scenes"].append(status)
        directory = output / spec.scene_id
        directory.mkdir()
        _json(directory / "scene.json", spec.to_dict())
        _json(directory / "evaluated_scene.json", evaluated_spec.to_dict())
        _json(directory / "reference.json", status)
        if latest is not None:
            times, frequencies = grids(evaluated_spec, config)
            result, observations = latest
            # The filename says latest; only reference.json status grants acceptance.
            np.savez_compressed(
                directory / "reference_latest.npz",
                times=times,
                frequencies=frequencies,
                waveforms=observations,
                spectra=spectrum(observations, times, frequencies),
                x=result.mesh.x,
                y=result.mesh.y,
                receiver_coordinates=np.concatenate(list(result.receiver_coordinates.values())),
            )
        if status["status"] != "converged":
            print(f"  {status['status']}; excluded from candidate accuracy scoring", flush=True)
            _json(output / "report.json", report)
            continue
        reference = latest[1]
        strategies = ["uniform", "heuristic"] + (["cnn"] if checkpoint is not None else [])
        for budget in spec.budgets:
            for strategy in strategies:
                row = {
                    "scene_id": spec.scene_id,
                    "split": spec.split,
                    "family": spec.family,
                    "budget": budget,
                    "strategy": strategy,
                    "status": "ok",
                }
                try:
                    result = run_scene(
                        evaluated_spec, budget, config, strategy=strategy, checkpoint=checkpoint
                    )
                    observations = sample_observables(result, times)
                    row.update(
                        metrics=metrics(observations, reference, times, frequencies, config),
                        diagnostics=result.diagnostics,
                        tail=tail_diagnostic(evaluated_spec, observations, config),
                    )
                    np.savez_compressed(
                        directory / f"{strategy}_{budget[0]}_{budget[1]}.npz",
                        times=times,
                        waveforms=observations,
                        x=result.mesh.x,
                        y=result.mesh.y,
                        spectra=spectrum(observations, times, frequencies),
                    )
                except (ValueError, RuntimeError) as error:
                    row.update(status="failed", error=str(error), error_type=type(error).__name__)
                report["candidates"].append(row)
                print(f"  {strategy} {budget}: {row['status']}", flush=True)
                _json(output / "report.json", report)
    write_summary(output, report)
    return report


def write_summary(output, report):
    accepted = sum(s["status"] == "converged" for s in report["scenes"])
    lines = [
        "# Mesh evaluation",
        f"Accepted references: {accepted}/{len(report['scenes'])}.",
        "CNN: "
        + (
            "not evaluated"
            if report["checkpoint"] is None
            else "untrained demonstration"
            if report["checkpoint"]["untrained"]
            else "supplied checkpoint"
        ),
        "Uniform means a uniform interior density preference projected to the hard mesh constraints.",
        "",
        "| Scene | Budget | Mesh | Waveform L2 | Spectrum L2 | Cell updates | GPU ms |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in report["candidates"]:
        if row["status"] == "ok":
            m, d = row["metrics"], row["diagnostics"]
            lines.append(
                f"| {row['scene_id']} | {row['budget']} | {row['strategy']} | "
                f"{m['waveform_l2']:.5g} | {m['spectrum_l2']:.5g} | {d['cell_updates']} | {d['gpu_ms']:.3f} |"
            )
        else:
            lines.append(
                f"| {row['scene_id']} | {row['budget']} | {row['strategy']} failed | — | — | — | — |"
            )
    lines += [
        "",
        "Nonconverged/failed references are excluded; see report.json for all failures and refinement history.",
        "Timing is a single-run measurement, not a controlled performance benchmark.",
    ]
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
