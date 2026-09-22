#!/usr/bin/env python3
"""Restartable analytic qualification matrix for off-grid PEC cylinders."""

import argparse
import hashlib
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scattermesh import PEC, Circle, Grid, PlaneWave, focused_axis, simulate
from scattermesh.analytic import cylinder_far_field
from scattermesh.metrics import compare_far_fields, scattering_loss

FREQUENCIES = np.array([0.8e9, 1.0e9, 1.2e9])
ANGLES = np.linspace(0, 2 * np.pi, 180, endpoint=False)
DOMAIN = 1.2


def _sha256_json(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def source_sha256():
    root = Path(__file__).resolve().parents[1]
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((root / "src/scattermesh").glob("*.py"))
    }


def case_fingerprint(case, sources):
    return _sha256_json(dict(case=case, source_sha256=sources))


def _atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _atomic_npz(path, **arrays):
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def scene_definitions():
    """Small deterministic set spanning size, subcell location, and incidence."""
    return [
        dict(scene_id="small_axis", radius=0.045, center=[0.5573, 0.6241], angle=0.0),
        dict(scene_id="small_oblique", radius=0.045, center=[0.6387, 0.5669], angle=1.17),
        dict(scene_id="medium_oblique", radius=0.080, center=[0.6130, 0.5910], angle=0.70),
        dict(scene_id="medium_reverse", radius=0.080, center=[0.5431, 0.6537], angle=3.91),
        dict(scene_id="large_axis", radius=0.120, center=[0.6257, 0.5483], angle=2.40),
    ]


def case_definitions():
    cases = []
    for scene in scene_definitions():
        for cells in (48, 64, 96, 128):
            cases.append(dict(**scene, cells=cells, mesh="uniform", mode="enlarged"))
        cases.append(dict(**scene, cells=64, mesh="uniform", mode="conformal"))
    baseline = next(scene for scene in scene_definitions() if scene["scene_id"] == "medium_oblique")
    cases.append(dict(**baseline, cells=160, mesh="uniform", mode="enlarged"))
    for cells in (96, 128, 160):
        cases.extend(
            [
                dict(
                    **baseline,
                    cells=cells,
                    mesh="uniform",
                    mode="enlarged",
                    variant="longer_time",
                ),
                dict(
                    **baseline,
                    cells=cells,
                    mesh="uniform",
                    mode="enlarged",
                    variant="thicker_pml",
                ),
                dict(
                    **baseline,
                    cells=cells,
                    mesh="uniform",
                    mode="enlarged",
                    variant="larger_contour",
                ),
            ]
        )
    for case in cases:
        variant = case.get("variant", "base")
        case["case_id"] = (
            f"{case['scene_id']}_{case['mesh']}{case['cells']}_{case['mode']}_{variant}"
        )
    return cases


def run_case(case, sources):
    axis = focused_axis(DOMAIN, case["cells"])
    grid = Grid(axis, axis)
    circle = Circle(tuple(case["center"]), case["radius"], PEC())
    source = PlaneWave(1e9, 1e-9, 9e-9, angle=case["angle"], origin=(0.6, 0.6))
    variant = case.get("variant", "base")
    duration = 45e-9 if variant == "longer_time" else 35e-9
    pml = 0.20 if variant == "thicker_pml" else 0.15
    monitor = (0.23, 0.97, 0.23, 0.97) if variant == "larger_contour" else None
    result = simulate(
        grid,
        [circle],
        source,
        frequencies=FREQUENCIES,
        duration=duration,
        pml_thickness=pml,
        monitor_bounds=monitor,
        pec_mode=case["mode"],
    )
    field = result.monitor.normalized_far_field(ANGLES)
    reference = np.array(
        [
            cylinder_far_field(
                case["radius"],
                PEC(),
                frequency,
                ANGLES,
                case["angle"],
                center=case["center"],
                incident_origin=source.origin,
            )
            for frequency in FREQUENCIES
        ]
    )
    maximum_size = 2 * np.pi * FREQUENCIES.max() / 299_792_458 * case["radius"]
    default_order = max(int(np.ceil(maximum_size + 4 * maximum_size ** (1 / 3) + 8)), 12)
    reference_high_order = np.array(
        [
            cylinder_far_field(
                case["radius"],
                PEC(),
                frequency,
                ANGLES,
                case["angle"],
                center=case["center"],
                incident_origin=source.origin,
                order=default_order + 8,
            )
            for frequency in FREQUENCIES
        ]
    )
    analytic_difference = np.linalg.norm(reference_high_order - reference, axis=1) / np.maximum(
        np.linalg.norm(reference_high_order, axis=1), 1e-30
    )
    width = 2 * np.pi * abs(field) ** 2
    reference_width = 2 * np.pi * abs(reference) ** 2
    width_error = np.linalg.norm(width - reference_width, axis=1) / np.linalg.norm(
        reference_width, axis=1
    )
    record = dict(
        case_id=case["case_id"],
        config_sha256=case_fingerprint(case, sources),
        scene={key: case[key] for key in ("scene_id", "radius", "center", "angle")},
        mesh=case["mesh"],
        mode=case["mode"],
        variant=variant,
        width_relative_l2_by_frequency=width_error.tolist(),
        analytic_series_relative_l2_by_frequency=analytic_difference.tolist(),
        analytic_default_order=default_order,
        finite_fields=all(np.isfinite(value).all() for value in result.fields.values()),
        **compare_far_fields(field, reference),
        **scattering_loss(field, reference),
        **result.diagnostics,
    )
    arrays = dict(
        angles=ANGLES,
        frequencies=FREQUENCIES,
        complex_far_field=field,
        analytic_complex_far_field=reference,
        incident_spectrum=result.monitor.incident,
        x=grid.x,
        y=grid.y,
    )
    return record, arrays


def cached_case(record_path, array_path, expected_hash):
    """Return a verified cached record, or None for a stale/incomplete case."""
    if not record_path.exists() or not array_path.exists():
        return None
    try:
        record = json.loads(record_path.read_text())
        with np.load(array_path) as arrays:
            field = arrays["complex_far_field"]
            reference = arrays["analytic_complex_far_field"]
            valid = (
                record.get("config_sha256") == expected_hash
                and field.shape == (len(FREQUENCIES), len(ANGLES))
                and reference.shape == field.shape
                and np.iscomplexobj(field)
                and np.iscomplexobj(reference)
                and np.isfinite(field).all()
                and np.isfinite(reference).all()
                and np.array_equal(arrays["frequencies"], FREQUENCIES)
                and np.array_equal(arrays["angles"], ANGLES)
            )
        return record if valid else None
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None


def field_sensitivities(case_directory):
    sensitivities = {}
    for cells in (96, 128, 160):
        base_name = f"medium_oblique_uniform{cells}_enlarged_base.npz"
        with np.load(case_directory / base_name) as baseline:
            baseline_field = baseline["complex_far_field"]
            reference = baseline["analytic_complex_far_field"]
        scale = np.linalg.norm(reference, axis=1)
        sensitivities[str(cells)] = {}
        for variant in ("longer_time", "thicker_pml", "larger_contour"):
            name = f"medium_oblique_uniform{cells}_enlarged_{variant}.npz"
            with np.load(case_directory / name) as candidate:
                difference = np.linalg.norm(candidate["complex_far_field"] - baseline_field, axis=1)
            sensitivities[str(cells)][variant] = (difference / scale).tolist()
    return sensitivities


def summarize(records, case_directory):
    base = [record for record in records if record["variant"] == "base"]
    resolution_records = {
        cells: [record for record in base if record["mode"] == "enlarged" and record["Nx"] == cells]
        for cells in (96, 128)
    }
    paired = []
    for scene in scene_definitions():
        enlarged_record = next(
            record
            for record in base
            if record["scene"]["scene_id"] == scene["scene_id"]
            and record["Nx"] == 64
            and record["mode"] == "enlarged"
        )
        conformal_record = next(
            record
            for record in base
            if record["scene"]["scene_id"] == scene["scene_id"]
            and record["Nx"] == 64
            and record["mode"] == "conformal"
        )
        paired.append(
            dict(
                scene_id=scene["scene_id"],
                enlarged_joint_loss=enlarged_record["joint_scattering_loss"],
                conformal_joint_loss=conformal_record["joint_scattering_loss"],
                enlarged_cell_updates=enlarged_record["cell_updates"],
                conformal_cell_updates=conformal_record["cell_updates"],
            )
        )
    sensitivities = {}
    for cells in (96, 128, 160):
        baseline = next(
            record
            for record in base
            if record["case_id"] == f"medium_oblique_uniform{cells}_enlarged_base"
        )
        sensitivities[str(cells)] = {}
        for variant in ("longer_time", "thicker_pml", "larger_contour"):
            record = next(
                record
                for record in records
                if record["variant"] == variant and record["Nx"] == cells
            )
            sensitivities[str(cells)][variant] = dict(
                joint_loss_ratio=record["joint_scattering_loss"]
                / baseline["joint_scattering_loss"],
                complex_loss_ratio=record["complex_mse_loss"] / baseline["complex_mse_loss"],
            )
    field_changes = field_sensitivities(case_directory)
    resolution_summary = {
        str(cells): dict(
            scene_count=len(subset),
            max_complex_relative_l2=max(
                max(record["complex_relative_l2_by_frequency"]) for record in subset
            ),
            max_phase_rms_degrees=max(
                max(record["phase_weighted_rms_degrees"]) for record in subset
            ),
            max_tail_ratio=max(record["tail_peak_over_global_peak"] for record in subset),
            min_dt_fraction=min(record["dt_fraction_of_grid_cfl"] for record in subset),
        )
        for cells, subset in resolution_records.items()
    }
    gates = dict(
        analytic_series=dict(
            threshold=1e-10,
            measured=max(
                max(record["analytic_series_relative_l2_by_frequency"]) for record in records
            ),
        ),
        complex_error_128=dict(
            threshold=0.01,
            measured=resolution_summary["128"]["max_complex_relative_l2"],
        ),
        phase_rms_degrees_128=dict(
            threshold=1.0,
            measured=resolution_summary["128"]["max_phase_rms_degrees"],
        ),
        tail_ratio_128=dict(
            threshold=1e-5,
            measured=resolution_summary["128"]["max_tail_ratio"],
        ),
        contour_change_128=dict(
            threshold=0.005,
            measured=max(field_changes["128"]["larger_contour"]),
        ),
        contour_change_160=dict(
            threshold=0.005,
            measured=max(field_changes["160"]["larger_contour"]),
        ),
    )
    for gate in gates.values():
        gate["passed"] = gate["measured"] < gate["threshold"]
    return dict(
        resolution_summary=resolution_summary,
        analytic_series_max_relative_l2=gates["analytic_series"]["measured"],
        paired_64=paired,
        sensitivity_loss_ratios=sensitivities,
        complex_field_sensitivity_by_frequency=field_changes,
        qualification_gates=gates,
        qualification_passed=all(gate["passed"] for gate in gates.values()),
    )


def plot_report(records, output):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    base = [record for record in records if record["variant"] == "base"]
    for scene in scene_definitions():
        subset = sorted(
            (
                record
                for record in base
                if record["scene"]["scene_id"] == scene["scene_id"] and record["mode"] == "enlarged"
            ),
            key=lambda record: record["Nx"],
        )
        axes[0].plot(
            [record["cell_updates"] for record in subset],
            [record["joint_scattering_loss"] for record in subset],
            "o-",
            label=scene["scene_id"],
        )
    axes[0].set(xscale="log", yscale="log", xlabel="Cell updates", ylabel="Joint loss")
    axes[0].grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    scenes = [scene["scene_id"] for scene in scene_definitions()]
    enlarged = [
        next(
            record["joint_scattering_loss"]
            for record in base
            if record["scene"]["scene_id"] == scene
            and record["Nx"] == 64
            and record["mode"] == "enlarged"
        )
        for scene in scenes
    ]
    conformal = [
        next(
            record["joint_scattering_loss"]
            for record in base
            if record["scene"]["scene_id"] == scene
            and record["Nx"] == 64
            and record["mode"] == "conformal"
        )
        for scene in scenes
    ]
    x = np.arange(len(scenes))
    axes[1].bar(x - 0.18, enlarged, 0.36, label="Enlarged")
    axes[1].bar(x + 0.18, conformal, 0.36, label="Plain conformal")
    axes[1].set(yscale="log", xticks=x, xticklabels=scenes, ylabel="Joint loss")
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].grid(axis="y", alpha=0.2)
    axes[1].legend()
    fig.suptitle("PEC-cylinder analytic qualification")
    fig.savefig(output / "qualification.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/pec_cylinder_qualification"))
    parser.add_argument("--limit", type=int, default=None, help="Run only this many ordered cases")
    parser.add_argument("--force", action="store_true", help="Recompute completed cases")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    case_directory = args.output / "cases"
    case_directory.mkdir(exist_ok=True)
    cases = case_definitions()[: args.limit]
    sources = source_sha256()
    records = []
    for index, case in enumerate(cases, 1):
        record_path = case_directory / f"{case['case_id']}.json"
        array_path = case_directory / f"{case['case_id']}.npz"
        expected_hash = case_fingerprint(case, sources)
        record = None if args.force else cached_case(record_path, array_path, expected_hash)
        if record is not None:
            records.append(record)
            print(f"[{index}/{len(cases)}] cached {case['case_id']}", flush=True)
            continue
        print(f"[{index}/{len(cases)}] running {case['case_id']}", flush=True)
        record, arrays = run_case(case, sources)
        _atomic_npz(array_path, **arrays)
        _atomic_json(record_path, record)
        records.append(record)
        print(
            f"  loss={record['joint_scattering_loss']:.5g} Nt={record['Nt']} "
            f"updates={record['cell_updates']} time={record['wall_seconds']:.2f}s",
            flush=True,
        )
    complete = len(records) == len(case_definitions()) and args.limit is None
    report = dict(
        schema=1,
        purpose="bounded M1 PEC-cylinder analytic qualification",
        created_utc=datetime.now(timezone.utc).isoformat(),
        complete=complete,
        completed_cases=len(records),
        planned_cases=len(case_definitions()),
        python_version=platform.python_version(),
        numpy_version=np.__version__,
        source_sha256=sources,
        frequencies_hz=FREQUENCIES.tolist(),
        observation_angles=ANGLES.tolist(),
        records=records,
        summary=summarize(records, case_directory) if complete else None,
        limitations=[
            "Finite 2D TMz single-cylinder matrix, not general PEC qualification",
            "No close gaps, cavities, thin screens, mixed materials, or external-solver comparison",
            "Joint-loss weights remain provisional",
        ],
    )
    _atomic_json(args.output / "report.json", report)
    if complete:
        plot_report(records, args.output)
    print(f"Report: {args.output / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
