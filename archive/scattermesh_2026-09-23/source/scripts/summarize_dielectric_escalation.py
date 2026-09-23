#!/usr/bin/env python3
"""Validate and summarize the CUDA high-contrast dielectric escalation runs."""

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DEFAULT_CONTOUR = [0.3, 0.9, 0.3, 0.9]
LARGE_CONTOUR = [0.23, 0.97, 0.23, 0.97]
VARIATION_LIMIT = 0.005


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("runs/dielectric_reference_escalation"))
    return parser.parse_args()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def load_attempts(root):
    attempts = []
    for record_path in sorted(root.glob("*/*/record.json")):
        spectra_path = record_path.with_name("spectra.npz")
        record = json.loads(record_path.read_text())
        with np.load(spectra_path) as arrays:
            numerical = arrays["complex_numerical"].copy()
            analytic = arrays["complex_analytic"].copy()
            frequencies = arrays["frequencies"].copy()
            angles = arrays["angles"].copy()
        if numerical.shape != analytic.shape or numerical.shape != (
            len(frequencies),
            len(angles),
        ):
            raise ValueError(f"Inconsistent spectra shapes in {spectra_path}")
        if not np.isfinite(numerical).all() or not np.isfinite(analytic).all():
            raise ValueError(f"Nonfinite spectra in {spectra_path}")
        attempts.append(
            dict(
                path=str(record_path.parent),
                record=record,
                numerical=numerical,
                analytic=analytic,
            )
        )
    return attempts


def select(attempts, scene, cells, duration, samples=24, contour=DEFAULT_CONTOUR):
    matches = []
    for attempt in attempts:
        config = attempt["record"]["config"]
        if (
            config["scene"] == scene
            and config["cells"] == cells
            and config["duration_ns"] == duration
            and config["samples"] == samples
            and config["contour"] == contour
        ):
            matches.append(attempt)
    if len(matches) != 1:
        raise ValueError(
            f"Expected one {scene}/{cells}/{duration}/{samples}/{contour} attempt, "
            f"found {len(matches)}"
        )
    return matches[0]


def relative_change(left, right):
    denominator = np.maximum(np.linalg.norm(right["analytic"], axis=1), 1e-30)
    values = np.linalg.norm(left["numerical"] - right["numerical"], axis=1) / denominator
    return dict(by_frequency=values.tolist(), maximum=float(values.max()))


def concise(attempt):
    record = attempt["record"]
    config = record["config"]
    return dict(
        path=attempt["path"],
        scene=config["scene"],
        cells=config["cells"],
        duration_ns=config["duration_ns"],
        samples=config["samples"],
        contour=config["contour"],
        status=record["status"],
        maximum_complex_relative_l2=max(record["complex_relative_l2_by_frequency"]),
        maximum_phase_rms_degrees=max(
            value or 0.0 for value in record["phase_weighted_rms_degrees"]
        ),
        tail_peak_over_global_peak=record["tail_peak_over_global_peak"],
        cell_updates=record["cell_updates"],
        wall_seconds=record["wall_seconds"],
    )


def qualify(attempts, scene, base_duration, extended_duration):
    base = select(attempts, scene, 512, base_duration)
    extended = select(attempts, scene, 512, extended_duration)
    sampled = select(attempts, scene, 512, base_duration, samples=48)
    contoured = select(attempts, scene, 512, base_duration, contour=LARGE_CONTOUR)
    variations = {
        "duration": relative_change(base, extended),
        "material_quadrature": relative_change(base, sampled),
        "nf2ff_contour": relative_change(base, contoured),
    }
    passed = base["record"]["status"] == "pass" and all(
        item["maximum"] < VARIATION_LIMIT for item in variations.values()
    )
    return dict(
        decision="accepted" if passed else "not_accepted",
        base=concise(base),
        comparisons={
            "extended_duration": concise(extended),
            "material_quadrature": concise(sampled),
            "nf2ff_contour": concise(contoured),
        },
        complex_field_variations=variations,
        variation_threshold=VARIATION_LIMIT,
    )


def plot_attempts(attempts, path):
    labels = []
    errors = []
    tails = []
    colors = []
    for attempt in attempts:
        item = concise(attempt)
        contour = "L" if item["contour"] == LARGE_CONTOUR else "D"
        labels.append(
            f"{item['scene']}\n{item['cells']}²/{item['duration_ns']:g}ns/"
            f"s{item['samples']}/{contour}"
        )
        errors.append(100 * item["maximum_complex_relative_l2"])
        tails.append(item["tail_peak_over_global_peak"])
        colors.append("#238636" if item["status"] == "pass" else "#cf222e")
    figure, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    positions = np.arange(len(labels))
    axes[0].bar(positions, errors, color=colors)
    axes[0].axhline(2.0, color="black", linestyle="--", linewidth=1)
    axes[0].set_ylabel("Max complex error [%]")
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(positions, tails, color=colors)
    axes[1].axhline(1e-5, color="black", linestyle="--", linewidth=1)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Tail / peak")
    axes[1].set_xticks(positions, labels, rotation=35, ha="right")
    axes[1].grid(axis="y", alpha=0.25)
    figure.suptitle("High-contrast dielectric reference escalation")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main():
    args = parse_args()
    attempts = load_attempts(args.root)
    eps12 = qualify(attempts, "eps12_small", 400, 100)
    eps30_lossy = qualify(attempts, "eps30_lossy", 50, 100)

    lossless_short = select(attempts, "eps30_small", 512, 100)
    lossless_long = select(attempts, "eps30_small", 512, 400)
    lossless_fine = select(attempts, "eps30_small", 768, 100)
    lossless = dict(
        decision="nonconverged_skip",
        reason=(
            "The duration and spatial probes fail individual accuracy/settling gates "
            "and exceed the 0.5% complex-field variation gate."
        ),
        hard_limit={
            "duration_probe": "512x512 cells for 400 ns",
            "spatial_probe": "768x768 cells for 100 ns",
        },
        attempts={
            "base": concise(lossless_short),
            "duration_probe": concise(lossless_long),
            "spatial_probe": concise(lossless_fine),
        },
        complex_field_variations={
            "duration": relative_change(lossless_short, lossless_long),
            "spatial": relative_change(lossless_short, lossless_fine),
        },
        variation_threshold=VARIATION_LIMIT,
    )
    report = dict(
        schema_version=1,
        attempt_count=len(attempts),
        qualification_policy={
            "individual_gates": (
                "complex relative L2 <2%, phase RMS <1.5 degrees, tail/peak <1e-5, "
                "analytic series change <1e-10"
            ),
            "independent_complex_field_variation_limit": VARIATION_LIMIT,
        },
        decisions={
            "eps12_small": eps12,
            "eps30_lossy": eps30_lossy,
            "eps30_small": lossless,
        },
        attempts=[concise(attempt) for attempt in attempts],
    )
    args.root.mkdir(parents=True, exist_ok=True)
    atomic_json(args.root / "report.json", report)
    plot_attempts(attempts, args.root / "qualification.png")
    print(
        f"Summarized {len(attempts)} attempts: eps12={eps12['decision']}, "
        f"eps30_lossy={eps30_lossy['decision']}, eps30_lossless={lossless['decision']}"
    )


if __name__ == "__main__":
    main()
