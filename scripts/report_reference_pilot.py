"""Create timing and convergence figures for a completed reference pilot."""

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()
    root = args.output.resolve()
    while not (root / "summary.json").exists():
        if not args.wait:
            parser.error("Pilot has not completed")
        # No solver launches or mutation of simulation artifacts in this reporter.
        time.sleep(20)
    summary = json.loads((root / "summary.json").read_text())
    rows = summary["scenes"]
    precomputed = json.loads((root / "pilot.json").read_text()).get("supplied_dataset_id")
    timing = "Includes setup, refinement and duration retries; geometry precomputed" if precomputed else "Includes geometry, setup, refinement and duration retries"
    fig, axes = plt.subplots(2, 1, figsize=(11, 9), layout="constrained")
    labels = [r["scene_id"].rsplit("-", 1)[-1] for r in rows]
    minutes = [r["total_wall_seconds"] / 60 for r in rows]
    colors = ["#198754" if r["status"] == "converged" else "#bb4b3e" for r in rows]
    axes[0].bar(labels, minutes, color=colors)
    axes[0].axhline(
        np.mean(minutes), color="black", ls="--", label=f"Mean {np.mean(minutes):.2f} min"
    )
    axes[0].set(
        ylabel="Total time per scene [minutes]",
        title=timing,
    )
    axes[0].legend()
    wave, spectral = [], []
    for row in rows:
        status = json.loads(Path(row["reference"]).read_text()) if row["reference"] else {}
        levels = status.get("levels", [])
        comparison = next((v["comparison"] for v in reversed(levels) if "comparison" in v), {})
        wave.append(100 * comparison.get("waveform_l2_max", np.nan))
        spectral.append(100 * comparison.get("spectrum_l2_max", np.nan))
        row["last_waveform_error_percent"] = wave[-1]
        row["last_spectrum_error_percent"] = spectral[-1]
        row["last_level"] = levels[-1]["budget"][0] if levels else None
    x = np.arange(len(rows))
    axes[1].bar(x - 0.18, wave, width=0.36, label="Waveform")
    axes[1].bar(x + 0.18, spectral, width=0.36, label="Spectrum")
    axes[1].axhline(
        2, color="black", ls="--", label="2% threshold (two consecutive passes required)"
    )
    axes[1].set(
        xticks=x,
        xticklabels=labels,
        xlabel="Training scene index",
        ylabel="Last available refinement error [%]",
    )
    axes[1].legend()
    fig.suptitle(
        f"{len(rows)}-scene reference pilot: {summary['accepted']}/{len(rows)} converged\nGreen: accepted · red: not accepted"
    )
    fig.savefig(root / "pilot_results.png", dpi=160)
    plt.close(fig)
    lines = [
        f"# {len(rows)}-scene reference pilot",
        "",
        f"Converged: **{summary['accepted']}/{len(rows)}**.",
        "",
        f"Mean total time per scene: **{np.mean(minutes):.2f} minutes**; median: **{np.median(minutes):.2f} minutes**.",
        f"Pilot elapsed time on four GPUs: **{summary['elapsed_seconds'] / 60:.2f} minutes**.",
        f"Overall throughput: **{summary['throughput_seconds_per_scene'] / 60:.2f} minutes per attempted scene**.",
        "",
        timing + ". Failed cases are included in the mean.",
        "The final error alone does not establish convergence; the saved record also checks consecutive passes and decay.",
        "",
        "| Scene | Family | Status | Final level | Time (min) | Duration extensions | Waveform error (%) | Spectrum error (%) |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    contention = root / "diagnostic_contention.json"
    if contention.exists():
        lines[2:2] = [
            "**Timing qualification:** GPU 3 was shared with averaging diagnostics. "
            "Overlapping scene times include contention and are not uncontended benchmarks. "
            "See diagnostic_contention.json for the interval.",
            "",
        ]
    for r in rows:
        lines.append(
            f"| {r['scene_id']} | {r['family']} | {r['status']} | {r['last_level']} | {r['total_wall_seconds'] / 60:.2f} | {r['duration_extensions']} | {r['last_waveform_error_percent']:.3f} | {r['last_spectrum_error_percent']:.3f} |"
        )
    lines += ["", "![Timing and convergence](pilot_results.png)", ""]
    (root / "results.md").write_text("\n".join(lines))
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
