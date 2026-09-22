#!/usr/bin/env python3
"""Qualify float32 CUDA far fields against matched float64 references."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from scattermesh import PEC, Circle, Grid, Material, PlaneWave, simulate_cuda
from scattermesh.metrics import compare_far_fields

ANGLES = np.linspace(0, 2 * np.pi, 180, endpoint=False)
FREQUENCIES = np.array([0.8e9, 1.0e9, 1.2e9])
CASES = ("eps12_small", "eps30_lossy", "pec_enlarged")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASES, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=Path("runs/cuda_precision"))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def source_sha256():
    root = Path(__file__).resolve().parents[1]
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((root / "src/scattermesh").glob("*.py"))
    }


def fingerprint(config):
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def json_default(item):
    if isinstance(item, np.generic):
        return item.item()
    raise TypeError(f"Object of type {type(item).__name__} is not JSON serializable")


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            value,
            indent=2,
            allow_nan=False,
            default=json_default,
        )
        + "\n"
    )
    os.replace(temporary, path)


def atomic_npz(path, **arrays):
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def find_dielectric_baseline(root, scene):
    matches = []
    for record_path in root.glob(f"{scene}/*/record.json"):
        record = json.loads(record_path.read_text())
        config = record["config"]
        required_duration = 400.0 if scene == "eps12_small" else 50.0
        if (
            record["status"] == "pass"
            and config["cells"] == 512
            and config["duration_ns"] == required_duration
            and config["samples"] == 24
            and config["contour"] == [0.3, 0.9, 0.3, 0.9]
        ):
            matches.append((record_path, record))
    if len(matches) != 1:
        raise ValueError(
            f"Expected one accepted float64 baseline for {scene}, found {len(matches)}"
        )
    record_path, record = matches[0]
    with np.load(record_path.with_name("spectra.npz")) as arrays:
        baseline = arrays["complex_numerical"].copy()
        analytic = arrays["complex_analytic"].copy()
        x, y = arrays["x"].copy(), arrays["y"].copy()
        if not np.array_equal(arrays["frequencies"], FREQUENCIES) or not np.array_equal(
            arrays["angles"], ANGLES
        ):
            raise ValueError("Baseline frequency/angle coordinates do not match")
    if baseline.shape != analytic.shape or baseline.shape != (len(FREQUENCIES), len(ANGLES)):
        raise ValueError("Baseline complex spectra have inconsistent shapes")
    return record_path, record, baseline, analytic, x, y


def dielectric_case(scene, device):
    root = Path("runs/dielectric_reference_escalation")
    path, baseline_record, baseline, analytic, x, y = find_dielectric_baseline(root, scene)
    config = baseline_record["config"]
    material = Material(config["epsilon_r"], config["sigma_e"])
    source_config = config["source"]
    source = PlaneWave(
        source_config["frequency_hz"],
        source_config["width_s"],
        source_config["delay_s"],
        angle=config["angle"],
        origin=tuple(source_config["origin_m"]),
    )
    result = simulate_cuda(
        Grid(x, y),
        [Circle(tuple(config["center"]), config["radius"], material)],
        source,
        frequencies=FREQUENCIES,
        duration=config["duration_ns"] * 1e-9,
        pml_thickness=config["pml"],
        monitor_bounds=tuple(config["contour"]),
        samples=config["samples"],
        device=device,
        dtype="float32",
    )
    return (
        dict(
            scene=scene,
            geometry=dict(
                type="circle",
                center=config["center"],
                radius=config["radius"],
                epsilon_r=config["epsilon_r"],
                sigma_e=config["sigma_e"],
            ),
            grid=dict(type="uniform", cells=config["cells"], x=x.tolist(), y=y.tolist()),
            duration_ns=config["duration_ns"],
            pml=config["pml"],
            contour=config["contour"],
            samples=config["samples"],
            source=source_config | {"angle": config["angle"]},
            float64_baseline_path=str(path.parent),
            float64_baseline_fingerprint=baseline_record["fingerprint"],
            float64_diagnostics=baseline_record,
        ),
        result,
        baseline,
        analytic,
    )


def pec_case(device):
    baseline_stem = Path(
        "runs/pec_cylinder_qualification/cases/medium_oblique_uniform160_enlarged_base"
    )
    baseline_record = json.loads(baseline_stem.with_suffix(".json").read_text())
    with np.load(baseline_stem.with_suffix(".npz")) as arrays:
        baseline = arrays["complex_far_field"].copy()
        analytic = arrays["analytic_complex_far_field"].copy()
        x, y = arrays["x"].copy(), arrays["y"].copy()
        if not np.array_equal(arrays["frequencies"], FREQUENCIES) or not np.array_equal(
            arrays["angles"], ANGLES
        ):
            raise ValueError("PEC baseline frequency/angle coordinates do not match")
    grid = Grid(x, y)
    geometry = dict(type="circle", center=[0.613, 0.591], radius=0.08)
    source_config = dict(
        frequency_hz=1e9,
        width_s=1e-9,
        delay_s=9e-9,
        angle=0.7,
        origin_m=[0.6, 0.6],
    )
    source = PlaneWave(
        source_config["frequency_hz"],
        source_config["width_s"],
        source_config["delay_s"],
        angle=source_config["angle"],
        origin=tuple(source_config["origin_m"]),
    )
    circle = Circle(tuple(geometry["center"]), geometry["radius"], PEC())
    settings = dict(
        frequencies=FREQUENCIES,
        duration=35e-9,
        pml_thickness=0.15,
        pec_mode="enlarged",
        device=device,
    )
    result = simulate_cuda(grid, [circle], source, dtype="float32", **settings)
    return (
        dict(
            scene="pec_enlarged",
            geometry=geometry | {"material": "PEC"},
            grid=dict(type="focused", cells=160, x=x.tolist(), y=y.tolist()),
            duration_ns=35.0,
            pml=0.15,
            contour=[0.3, 0.9, 0.3, 0.9],
            source=source_config,
            pec_mode="enlarged",
            float64_baseline_path=str(baseline_stem),
            float64_baseline_fingerprint=baseline_record["config_sha256"],
            float64_diagnostics=baseline_record,
        ),
        result,
        baseline,
        analytic,
    )


def main():
    args = parse_args()
    case_directory = args.output / args.case
    record_path, spectra_path = case_directory / "record.json", case_directory / "spectra.npz"
    sources = source_sha256()
    provisional = dict(case=args.case, device=args.device, source_sha256=sources)
    expected = fingerprint(provisional)
    if not args.force and record_path.exists() and spectra_path.exists():
        record = json.loads(record_path.read_text())
        with np.load(spectra_path) as arrays:
            valid = (
                record.get("fingerprint") == expected
                and arrays["float32"].shape == (len(FREQUENCIES), len(ANGLES))
                and arrays["float64"].shape == arrays["float32"].shape
                and np.isfinite(arrays["float32"]).all()
                and np.isfinite(arrays["float64"]).all()
            )
        if valid:
            print(f"{args.case}: resumed {record['decision']} {record_path}")
            return

    case_directory.mkdir(parents=True, exist_ok=True)
    config, result, baseline, analytic = (
        pec_case(args.device)
        if args.case == "pec_enlarged"
        else dielectric_case(args.case, args.device)
    )
    candidate = result.monitor.normalized_far_field(ANGLES)
    precision = compare_far_fields(candidate, baseline)
    float32_analytic = compare_far_fields(candidate, analytic)
    float64_analytic = compare_far_fields(baseline, analytic)
    max_precision = max(precision["complex_relative_l2_by_frequency"])
    max_phase = max(value or 0.0 for value in precision["phase_weighted_rms_degrees"])
    analytic_degradation = max(
        np.asarray(float32_analytic["complex_relative_l2_by_frequency"])
        - np.asarray(float64_analytic["complex_relative_l2_by_frequency"])
    )
    tail = result.diagnostics["tail_peak_over_global_peak"]
    gates = dict(
        maximum_float32_vs_float64_complex_relative_l2=dict(
            measured=max_precision, threshold=1e-3, passed=max_precision < 1e-3
        ),
        maximum_float32_vs_float64_phase_rms_degrees=dict(
            measured=max_phase, threshold=0.1, passed=max_phase < 0.1
        ),
        maximum_analytic_error_degradation=dict(
            measured=float(analytic_degradation),
            threshold=1e-3,
            passed=bool(analytic_degradation < 1e-3),
        ),
        tail_peak_over_global_peak=dict(measured=tail, threshold=1e-5, passed=tail < 1e-5),
    )
    finite = all(np.isfinite(field).all() for field in result.fields.values())
    record = dict(
        fingerprint=expected,
        fingerprint_input=provisional,
        config=config,
        decision="accepted"
        if finite and all(gate["passed"] for gate in gates.values())
        else "rejected",
        finite_fields=finite,
        gates=gates,
        float32_vs_float64=precision,
        float32_vs_analytic=float32_analytic,
        float64_vs_analytic=float64_analytic,
        float32_diagnostics=result.diagnostics,
    )
    atomic_npz(
        spectra_path,
        frequencies=FREQUENCIES,
        angles=ANGLES,
        float32=candidate,
        float64=baseline,
        analytic=analytic,
        incident_float32=result.monitor.incident,
    )
    atomic_json(record_path, record)
    print(
        f"{args.case}: {record['decision']} precision={max_precision:.3g} "
        f"phase={max_phase:.3g}deg tail={tail:.3g} wall={result.diagnostics['wall_seconds']:.2f}s"
    )


if __name__ == "__main__":
    main()
