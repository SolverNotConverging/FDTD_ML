#!/usr/bin/env python3
"""Restartable analytic qualification matrix for off-grid dielectric cylinders."""

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

from scattermesh import Circle, Grid, Material, PlaneWave, focused_axis, simulate
from scattermesh.analytic import cylinder_far_field
from scattermesh.constants import C0
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
    return [
        dict(
            scene_id="eps2_small",
            radius=0.045,
            center=[0.5573, 0.6241],
            angle=0.0,
            epsilon_r=2,
            sigma_e=0,
            stage="initial",
        ),
        dict(
            scene_id="eps4_medium",
            radius=0.08,
            center=[0.613, 0.591],
            angle=0.70,
            epsilon_r=4,
            sigma_e=0,
            stage="initial",
        ),
        dict(
            scene_id="eps12_small",
            radius=0.05,
            center=[0.6387, 0.5669],
            angle=1.17,
            epsilon_r=12,
            sigma_e=0,
            stage="intermediate_contrast",
        ),
        dict(
            scene_id="eps30_small",
            radius=0.045,
            center=[0.5431, 0.6537],
            angle=3.91,
            epsilon_r=30,
            sigma_e=0,
            stage="stress_high_contrast",
        ),
        dict(
            scene_id="eps4_lossy",
            radius=0.07,
            center=[0.6257, 0.5483],
            angle=2.40,
            epsilon_r=4,
            sigma_e=0.05,
            stage="initial",
        ),
        dict(
            scene_id="eps30_lossy",
            radius=0.045,
            center=[0.5817, 0.6329],
            angle=5.20,
            epsilon_r=30,
            sigma_e=0.2,
            stage="stress_high_contrast",
        ),
    ]


def case_definitions():
    cases = []
    for scene in scene_definitions():
        for cells in (64, 96, 128):
            cases.append(dict(**scene, cells=cells, mesh="uniform", variant="base"))
        if scene["stage"] == "initial":
            cases.append(dict(**scene, cells=192, mesh="uniform", variant="base"))
    base = next(s for s in scene_definitions() if s["scene_id"] == "eps4_medium")
    for variant, extra in (
        ("longer_time", {"duration": 70e-9}),
        ("thicker_pml", {"pml": 0.20}),
        ("larger_contour", {"monitor": (0.23, 0.97, 0.23, 0.97)}),
        ("finer_quadrature", {"samples": 24}),
    ):
        cases.append(dict(**base, cells=192, mesh="uniform", variant=variant, **extra))
    for case in cases:
        case["case_id"] = f"{case['scene_id']}_uniform{case['cells']}_{case['variant']}"
    return cases


def run_case(case, sources):
    axis = focused_axis(DOMAIN, case["cells"])
    grid = Grid(axis, axis)
    material = Material(case["epsilon_r"], case["sigma_e"])
    circle = Circle(tuple(case["center"]), case["radius"], material)
    source = PlaneWave(1e9, 1e-9, 9e-9, angle=case["angle"], origin=(0.6, 0.6))
    result = simulate(
        grid,
        [circle],
        source,
        frequencies=FREQUENCIES,
        duration=case.get("duration", 50e-9),
        pml_thickness=case.get("pml", 0.15),
        monitor_bounds=case.get("monitor"),
        samples=case.get("samples", 12),
    )
    field = result.monitor.normalized_far_field(ANGLES)
    reference = np.array(
        [
            cylinder_far_field(
                case["radius"],
                material,
                f,
                ANGLES,
                case["angle"],
                center=case["center"],
                incident_origin=source.origin,
            )
            for f in FREQUENCIES
        ]
    )
    maximum_size = 2 * np.pi * FREQUENCIES.max() / C0 * case["radius"]
    default_order = max(int(np.ceil(maximum_size + 4 * maximum_size ** (1 / 3) + 8)), 12)
    high = np.array(
        [
            cylinder_far_field(
                case["radius"],
                material,
                f,
                ANGLES,
                case["angle"],
                center=case["center"],
                incident_origin=source.origin,
                order=default_order + 8,
            )
            for f in FREQUENCIES
        ]
    )
    series = np.linalg.norm(high - reference, axis=1) / np.maximum(
        np.linalg.norm(high, axis=1), 1e-30
    )
    record = dict(
        case_id=case["case_id"],
        config_sha256=case_fingerprint(case, sources),
        scene={
            k: case[k]
            for k in (
                "scene_id",
                "radius",
                "center",
                "angle",
                "epsilon_r",
                "sigma_e",
                "stage",
            )
        },
        mesh="uniform",
        variant=case["variant"],
        analytic_series_relative_l2_by_frequency=series.tolist(),
        analytic_default_order=default_order,
        minimum_internal_cells_per_wavelength=C0
        / (FREQUENCIES.max() * np.sqrt(case["epsilon_r"]) * np.diff(axis).max()),
        finite_fields=all(np.isfinite(v).all() for v in result.fields.values()),
        **compare_far_fields(field, reference),
        **scattering_loss(field, reference),
        **result.diagnostics,
    )
    return record, dict(
        angles=ANGLES,
        frequencies=FREQUENCIES,
        complex_far_field=field,
        analytic_complex_far_field=reference,
        incident_spectrum=result.monitor.incident,
        x=grid.x,
        y=grid.y,
    )


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


def summarize(records, case_directory):
    base = [r for r in records if r["variant"] == "base"]
    by_res = {}
    for cells in (64, 96, 128):
        subset = [r for r in base if r["Nx"] == cells]
        by_res[str(cells)] = dict(
            scene_count=len(subset),
            max_complex_relative_l2=max(max(r["complex_relative_l2_by_frequency"]) for r in subset),
            max_phase_rms_degrees=max(
                max(x or 0 for x in r["phase_weighted_rms_degrees"]) for r in subset
            ),
            max_tail_ratio=max(r["tail_peak_over_global_peak"] for r in subset),
        )
    initial_192 = [r for r in base if r["Nx"] == 192 and r["scene"]["stage"] == "initial"]
    sensitivities = {}
    for variant in ("longer_time", "thicker_pml", "larger_contour", "finer_quadrature"):
        candidate = next(r for r in records if r["variant"] == variant)
        baseline = next(
            r for r in base if r["scene"]["scene_id"] == "eps4_medium" and r["Nx"] == 192
        )
        with (
            np.load(case_directory / f"{baseline['case_id']}.npz") as b,
            np.load(case_directory / f"{candidate['case_id']}.npz") as c,
        ):
            sensitivities[variant] = (
                np.linalg.norm(c["complex_far_field"] - b["complex_far_field"], axis=1)
                / np.maximum(np.linalg.norm(b["analytic_complex_far_field"], axis=1), 1e-30)
            ).tolist()
    gates = {
        "analytic_series": max(max(r["analytic_series_relative_l2_by_frequency"]) for r in records),
        "initial_complex_error_192": max(
            max(r["complex_relative_l2_by_frequency"]) for r in initial_192
        ),
        "initial_phase_rms_192": max(
            max(x or 0 for x in r["phase_weighted_rms_degrees"]) for r in initial_192
        ),
        "initial_tail_192": max(r["tail_peak_over_global_peak"] for r in initial_192),
        "variant_field_change": max(max(v) for v in sensitivities.values()),
    }
    for k, v in list(gates.items()):
        threshold = {
            "analytic_series": 1e-10,
            "initial_complex_error_192": 0.02,
            "initial_phase_rms_192": 1.5,
            "initial_tail_192": 1e-5,
            "variant_field_change": 0.005,
        }[k]
        gates[k] = {"measured": v, "threshold": threshold, "passed": v < threshold}
    return dict(
        resolution_summary=by_res,
        initial_192=initial_192,
        deferred_128=[r for r in base if r["Nx"] == 128 and r["scene"]["stage"] != "initial"],
        sensitivities_by_variant=sensitivities,
        qualification_gates=gates,
        qualification_passed=all(x["passed"] for x in gates.values()),
    )


def plot_report(records, output):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    base = [r for r in records if r["variant"] == "base"]
    for scene in scene_definitions():
        sub = sorted(
            [r for r in base if r["scene"]["scene_id"] == scene["scene_id"]], key=lambda r: r["Nx"]
        )
        axes[0].plot(
            [r["cell_updates"] for r in sub],
            [r["joint_scattering_loss"] for r in sub],
            "o-",
            label=scene["scene_id"],
        )
    axes[0].set(xscale="log", yscale="log", xlabel="Cell updates", ylabel="Joint scattering loss")
    axes[0].grid(alpha=0.2)
    axes[0].legend(fontsize=7)
    finest = [r for r in base if r["Nx"] == 128]
    axes[1].bar(
        [r["scene"]["scene_id"] for r in finest],
        [100 * max(r["complex_relative_l2_by_frequency"]) for r in finest],
        color=["tab:blue" if r["scene"]["stage"] == "initial" else "tab:red" for r in finest],
    )
    axes[1].axhline(2, color="k", linestyle="--", linewidth=1, label="initial-stage gate")
    axes[1].set(ylabel="Worst-frequency complex error (%)", yscale="log")
    axes[1].tick_params(axis="x", rotation=35)
    axes[1].grid(axis="y", alpha=0.2)
    axes[1].legend()
    fig.suptitle("Single dielectric-cylinder analytic qualification")
    fig.savefig(output / "qualification.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("runs/dielectric_cylinder_qualification")
    )
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
        purpose="bounded TMz single-cylinder dielectric analytic qualification",
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
            "Finite 2D TMz single-cylinder CPU matrix; results do not claim general dielectric qualification",
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
