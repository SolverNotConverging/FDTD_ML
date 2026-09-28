"""Render an existing study summary without running a solver."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    study = json.loads((args.directory / "summary.json").read_text())
    lines = [
        "# Mesh study",
        "",
        "Exact 2D PEC geometry; +x excitation. Angles below rotate the object.",
        "A prepared mesh is not a qualified scattering result. Error is relative complex far-field error.",
        "",
        "| Shape | Orientation (deg) | Cells | Status | Error |",
        "|---|---:|---|---|---:|",
    ]
    for row in study["cases"]:
        error = row.get("error")
        lines.append(
            f"| {row['shape']} | {row['orientation_deg']} | {row['cells']} | {row['status']} | {error if error is not None else '—'} |"
        )
    for row in study["cases"]:
        if row.get("figure"):
            lines.extend(["", f"![{row['shape']} best mesh]({row['figure']})"])
    lines.extend(
        [
            "",
            "Full configuration, failure messages and acceptance statistics: [summary.json](summary.json).",
        ]
    )
    (args.directory / "report.md").write_text("\n".join(lines), encoding="utf8")


if __name__ == "__main__":
    main()
