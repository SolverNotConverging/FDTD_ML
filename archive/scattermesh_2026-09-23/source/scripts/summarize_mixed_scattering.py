#!/usr/bin/env python3
"""Aggregate the accepted mixed PEC/dielectric qualification profiles."""

import json
import os
from pathlib import Path

PROFILES = {
    "separated_circles": "report.json",
    "close_gap_circles": "report_close_final.json",
    "rectangle_lossy": "report_rectangle_settled.json",
}


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def main():
    root = Path("runs/mixed_scattering_qualification")
    scenes = {}
    for scene, report_name in PROFILES.items():
        report_path = root / scene / report_name
        report = json.loads(report_path.read_text())
        if report["decision"] != "accepted" or not all(report["gates"].values()):
            raise ValueError(f"Mixed-scattering scene is not accepted: {scene}")
        records = [json.loads(Path(path).read_text()) for path in report["records"].values()]
        scenes[scene] = dict(
            report_path=str(report_path),
            profile=report.get("profile", "initial"),
            maximum_tail_peak_over_global_peak=max(
                record["tail_peak_over_global_peak"] for record in records
            ),
            maximum_pec_total_field=max(record["maximum_pec_total_field"] for record in records),
            variations=report["variations"],
            total_wall_seconds=sum(record["wall_seconds"] for record in records),
        )
    summary = dict(
        schema_version=1,
        decision="accepted",
        scope="Separated PEC/dielectric circles and rectangles; enlarged transfer stencils in vacuum",
        scene_count=len(scenes),
        scenes=scenes,
    )
    atomic_json(root / "report.json", summary)
    print(
        "accepted mixed scenes: "
        + ", ".join(
            f"{name} spatial={100 * row['variations']['spatial']['maximum']:.3f}%"
            for name, row in scenes.items()
        )
    )


if __name__ == "__main__":
    main()
