#!/usr/bin/env python3
"""Qualify separated mixed PEC/dielectric scattering with CUDA self-convergence."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from scattermesh import (
    PEC,
    Circle,
    Grid,
    Material,
    PlaneWave,
    Rectangle,
    focused_axis,
    simulate_cuda,
)

FREQUENCIES = np.array([0.8e9, 1.0e9, 1.2e9])
ANGLES = np.linspace(0, 2 * np.pi, 180, endpoint=False)
SCENES = {
    "separated_circles": dict(
        angle=0.7,
        pec=dict(type="circle", center=[0.47, 0.59], radius=0.08),
        dielectric=dict(
            type="circle", center=[0.73, 0.62], radius=0.07, epsilon_r=4.0, sigma_e=0.0
        ),
    ),
    "close_gap_circles": dict(
        angle=2.1,
        pec=dict(type="circle", center=[0.50, 0.60], radius=0.08),
        dielectric=dict(
            type="circle", center=[0.665, 0.60], radius=0.07, epsilon_r=8.0, sigma_e=0.0
        ),
    ),
    "rectangle_lossy": dict(
        angle=5.0,
        pec=dict(type="rectangle", bounds=[0.42, 0.55, 0.50, 0.72]),
        dielectric=dict(
            type="circle", center=[0.72, 0.58], radius=0.07, epsilon_r=12.0, sigma_e=0.05
        ),
    ),
}
INITIAL_VARIANTS = {
    "coarse": dict(cells=192, duration_ns=40.0, samples=16, larger_contour=False),
    "base": dict(cells=256, duration_ns=40.0, samples=16, larger_contour=False),
    "spatial": dict(cells=320, duration_ns=40.0, samples=16, larger_contour=False),
    "duration": dict(cells=256, duration_ns=55.0, samples=16, larger_contour=False),
    "quadrature": dict(cells=256, duration_ns=40.0, samples=32, larger_contour=False),
    "contour": dict(cells=256, duration_ns=40.0, samples=16, larger_contour=True),
}
PROFILES = {
    "initial": INITIAL_VARIANTS,
    "rectangle_settled": {
        "base": dict(cells=256, duration_ns=55.0, samples=16, larger_contour=False),
        "spatial": dict(cells=320, duration_ns=55.0, samples=16, larger_contour=False),
        "duration": dict(cells=256, duration_ns=70.0, samples=16, larger_contour=False),
        "quadrature": dict(cells=256, duration_ns=55.0, samples=32, larger_contour=False),
        "contour": dict(cells=256, duration_ns=55.0, samples=16, larger_contour=True),
    },
    "close_settled": {
        "base": dict(cells=320, duration_ns=160.0, samples=16, larger_contour=False),
        "spatial": dict(cells=384, duration_ns=160.0, samples=16, larger_contour=False),
        "duration": dict(cells=320, duration_ns=200.0, samples=16, larger_contour=False),
        "quadrature": dict(cells=320, duration_ns=160.0, samples=32, larger_contour=False),
        "contour": dict(cells=320, duration_ns=160.0, samples=16, larger_contour=True),
    },
    "close_final": {
        "base": dict(cells=320, duration_ns=240.0, samples=16, larger_contour=False),
        "spatial": dict(cells=384, duration_ns=240.0, samples=16, larger_contour=False),
        "duration": dict(cells=320, duration_ns=300.0, samples=16, larger_contour=False),
        "quadrature": dict(cells=320, duration_ns=240.0, samples=32, larger_contour=False),
        "contour": dict(cells=320, duration_ns=240.0, samples=16, larger_contour=True),
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=SCENES, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--profile", choices=PROFILES, default="initial")
    parser.add_argument(
        "--run-variant", choices={name for profile in PROFILES.values() for name in profile}
    )
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("runs/mixed_scattering_qualification"))
    return parser.parse_args()


def source_sha256():
    root = Path(__file__).resolve().parents[1]
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((root / "src/scattermesh").glob("*.py"))
    }


def fingerprint(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def atomic_npz(path, **arrays):
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def make_object(spec, material):
    if spec["type"] == "circle":
        return Circle(tuple(spec["center"]), spec["radius"], material)
    return Rectangle(tuple(spec["bounds"]), material)


def cached(record_path, spectra_path, expected):
    if not record_path.exists() or not spectra_path.exists():
        return None
    try:
        record = json.loads(record_path.read_text())
        with np.load(spectra_path) as arrays:
            field = arrays["complex_far_field"]
            valid = (
                record.get("fingerprint") == expected
                and field.shape == (len(FREQUENCIES), len(ANGLES))
                and np.iscomplexobj(field)
                and np.isfinite(field).all()
                and np.array_equal(arrays["frequencies"], FREQUENCIES)
                and np.array_equal(arrays["angles"], ANGLES)
            )
        return record if valid else None
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None


def run_variant(scene_name, profile, variant_name, variant, device, output, sources):
    scene = SCENES[scene_name]
    config = dict(
        scene=scene_name,
        scene_definition=scene,
        variant=variant_name,
        **variant,
        device=device,
        pml=0.15,
        frequencies_hz=FREQUENCIES.tolist(),
        observation_angles=ANGLES.tolist(),
        source=dict(frequency_hz=1e9, width_s=1e-9, delay_s=9e-9, origin_m=[0.6, 0.6]),
        source_sha256=sources,
    )
    if profile != "initial":
        config["profile"] = profile
    digest = fingerprint(config)
    directory = (
        output / scene_name / variant_name
        if profile == "initial"
        else output / scene_name / profile / variant_name
    )
    directory.mkdir(parents=True, exist_ok=True)
    record_path, spectra_path = directory / "record.json", directory / "spectra.npz"
    record = cached(record_path, spectra_path, digest)
    if record is not None:
        return record, spectra_path

    axis = focused_axis(1.2, variant["cells"], width=0.18, strength=1.5)
    grid = Grid(axis, axis)
    dielectric_spec = scene["dielectric"]
    dielectric = Material(dielectric_spec["epsilon_r"], dielectric_spec["sigma_e"])
    pec_object = make_object(scene["pec"], PEC())
    objects = [pec_object, make_object(dielectric_spec, dielectric)]
    source = PlaneWave(1e9, 1e-9, 9e-9, angle=scene["angle"], origin=(0.6, 0.6))
    contour = (0.23, 0.97, 0.23, 0.97) if variant["larger_contour"] else None
    result = simulate_cuda(
        grid,
        objects,
        source,
        frequencies=FREQUENCIES,
        duration=variant["duration_ns"] * 1e-9,
        pml_thickness=0.15,
        monitor_bounds=contour,
        samples=variant["samples"],
        pec_mode="enlarged",
        device=device,
        dtype="float64",
    )
    field = result.monitor.normalized_far_field(ANGLES)
    pec_mask = pec_object.contains(grid.x[:, None], grid.y[None, :])
    total = result.fields["Ez"] + source.electric(
        grid.x[:, None], grid.y[None, :], result.diagnostics["simulated_time"]
    )
    record = dict(
        fingerprint=digest,
        config=config,
        finite_fields=all(np.isfinite(value).all() for value in result.fields.values()),
        maximum_pec_total_field=float(np.max(np.abs(total[pec_mask]))),
        **result.diagnostics,
    )
    atomic_npz(
        spectra_path,
        frequencies=FREQUENCIES,
        angles=ANGLES,
        complex_far_field=field,
        incident_spectrum=result.monitor.incident,
        x=grid.x,
        y=grid.y,
    )
    atomic_json(record_path, record)
    return record, spectra_path


def relative_change(left_path, right_path, scale_path):
    with np.load(left_path) as left, np.load(right_path) as right, np.load(scale_path) as scale:
        denominator = np.maximum(np.linalg.norm(scale["complex_far_field"], axis=1), 1e-30)
        values = (
            np.linalg.norm(left["complex_far_field"] - right["complex_far_field"], axis=1)
            / denominator
        )
    return dict(by_frequency=values.tolist(), maximum=float(values.max()))


def load_saved_variant(output, scene, profile, variant_name, variant, sources):
    directory = (
        output / scene / variant_name
        if profile == "initial"
        else output / scene / profile / variant_name
    )
    record_path, spectra_path = directory / "record.json", directory / "spectra.npz"
    record = json.loads(record_path.read_text())
    config = record["config"]
    if (
        config.get("scene") != scene
        or config.get("variant") != variant_name
        or config.get("cells") != variant["cells"]
        or config.get("duration_ns") != variant["duration_ns"]
        or config.get("samples") != variant["samples"]
        or config.get("larger_contour") != variant["larger_contour"]
        or config.get("source_sha256") != sources
        or (profile != "initial" and config.get("profile") != profile)
        or record.get("fingerprint") != fingerprint(config)
        or cached(record_path, spectra_path, record["fingerprint"]) is None
    ):
        raise ValueError(f"Missing, stale, or inconsistent saved variant at {directory}")
    return record, spectra_path


def main():
    args = parse_args()
    if args.profile == "rectangle_settled" and args.scene != "rectangle_lossy":
        raise ValueError("rectangle_settled profile requires rectangle_lossy scene")
    if args.profile in {"close_settled", "close_final"} and args.scene != "close_gap_circles":
        raise ValueError("Close-gap profiles require close_gap_circles scene")
    sources = source_sha256()
    variants = PROFILES[args.profile]
    if args.run_variant is not None:
        if args.run_variant not in variants:
            raise ValueError(f"Variant {args.run_variant} is not part of profile {args.profile}")
        record, _ = run_variant(
            args.scene,
            args.profile,
            args.run_variant,
            variants[args.run_variant],
            args.device,
            args.output,
            sources,
        )
        print(
            f"{args.scene}/{args.profile}/{args.run_variant}: "
            f"tail={record['tail_peak_over_global_peak']:.3g} wall={record['wall_seconds']:.2f}s",
            flush=True,
        )
        return
    records, spectra = {}, {}
    for variant_name, variant in variants.items():
        if args.summarize:
            records[variant_name], spectra[variant_name] = load_saved_variant(
                args.output, args.scene, args.profile, variant_name, variant, sources
            )
        else:
            records[variant_name], spectra[variant_name] = run_variant(
                args.scene, args.profile, variant_name, variant, args.device, args.output, sources
            )
            print(
                f"{args.scene}/{args.profile}/{variant_name}: "
                f"tail={records[variant_name]['tail_peak_over_global_peak']:.3g} "
                f"wall={records[variant_name]['wall_seconds']:.2f}s",
                flush=True,
            )
    variations = {
        "spatial": relative_change(spectra["base"], spectra["spatial"], spectra["spatial"]),
        "duration": relative_change(spectra["base"], spectra["duration"], spectra["spatial"]),
        "quadrature": relative_change(spectra["base"], spectra["quadrature"], spectra["spatial"]),
        "contour": relative_change(spectra["base"], spectra["contour"], spectra["spatial"]),
    }
    if "coarse" in spectra:
        variations["coarse_to_base"] = relative_change(
            spectra["coarse"], spectra["base"], spectra["spatial"]
        )
    gates = dict(
        all_fields_finite=all(record["finite_fields"] for record in records.values()),
        all_tails_settled=max(record["tail_peak_over_global_peak"] for record in records.values())
        < 1e-5,
        pec_total_field=max(record["maximum_pec_total_field"] for record in records.values())
        < 1e-12,
        spatial_variation=variations["spatial"]["maximum"] < 0.01,
        duration_variation=variations["duration"]["maximum"] < 0.005,
        quadrature_variation=variations["quadrature"]["maximum"] < 0.005,
        contour_variation=variations["contour"]["maximum"] < 0.005,
    )
    report = dict(
        schema_version=1,
        scene=args.scene,
        profile=args.profile,
        decision="accepted" if all(gates.values()) else "not_accepted",
        gates=gates,
        variations=variations,
        records={
            variant: str(
                (
                    args.output / args.scene / variant
                    if args.profile == "initial"
                    else args.output / args.scene / args.profile / variant
                )
                / "record.json"
            )
            for variant in variants
        },
    )
    report_name = "report.json" if args.profile == "initial" else f"report_{args.profile}.json"
    atomic_json(args.output / args.scene / report_name, report)
    print(f"{args.scene}: {report['decision']} {json.dumps(variations)}", flush=True)


if __name__ == "__main__":
    main()
