#!/usr/bin/env python3
"""Check relocated archive links, inventories and representative content."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(*, full=False):
    receiver = ROOT / "archive" / "receiver_cnn_2026-09-22"
    scattering = ROOT / "archive" / "scattermesh_2026-09-23"
    old_inventory = json.loads((receiver / "INVENTORY.json").read_text())
    relocations = json.loads((receiver / "RELOCATION.json").read_text())
    mapped = {item["path"]: item for item in relocations["links"]}
    errors = []
    if len(mapped) != len(old_inventory["symlinks"]) or len(mapped) != 404:
        errors.append("Receiver link count differs from saved inventory")
    for item in old_inventory["symlinks"]:
        link = receiver / item["path"]
        change = mapped.get(item["path"])
        if change is None or change["old_target"] != item["target"]:
            errors.append(f"Receiver relocation provenance differs: {item['path']}")
            continue
        if (
            not link.is_symlink()
            or link.readlink().as_posix() != change["new_relative_target"]
            or not link.exists()
            or not link.resolve().is_relative_to(receiver / "artifacts")
        ):
            errors.append(f"Receiver relocated link is broken: {item['path']}")
    files = old_inventory["regular_files"]
    selected = files if full else files[:: max(1, len(files) // 64)]
    for item in selected:
        path = receiver / item["path"]
        if (
            not path.is_file()
            or path.stat().st_size != item["bytes"]
            or sha256(path) != item["sha256"]
        ):
            errors.append(f"Receiver file differs: {item['path']}")
    with (scattering / "metadata" / "runs_inventory.csv").open(newline="") as stream:
        runs = list(csv.DictReader(stream))
    selected_runs = runs if full else runs[:: max(1, len(runs) // 64)]
    for item in selected_runs:
        path = scattering / "runs" / item["path"]
        if item["kind"] == "file":
            if (
                not path.is_file()
                or path.stat().st_size != int(item["size_bytes"])
                or sha256(path) != item["sha256_or_link_target"]
            ):
                errors.append(f"Scattering run file differs: {item['path']}")
        elif not path.is_symlink() or path.readlink().as_posix() != item["sha256_or_link_target"]:
            errors.append(f"Scattering run link differs: {item['path']}")
    legacy_link = ROOT / "runs"
    if not legacy_link.is_symlink() or legacy_link.resolve() != (scattering / "runs").resolve():
        errors.append("Legacy runs compatibility link does not resolve to local archive")
    artifact_link = scattering / "source" / "artifacts"
    if not artifact_link.is_symlink() or not artifact_link.exists():
        errors.append("Archived source artifacts compatibility link is broken")
    if (ROOT / "artifacts").exists() or (ROOT / "artifacts").is_symlink():
        errors.append("Historical artifacts compatibility link remains in active root")
    return {
        "mode": "full" if full else "representative",
        "receiver_regular_inventory_count": len(files),
        "receiver_files_hashed": len(selected),
        "receiver_relocated_links_checked": len(mapped),
        "scattering_run_inventory_count": len(runs),
        "scattering_run_files_hashed": len(selected_runs),
        "errors": errors,
        "passed": not errors,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="hash every inventoried file")
    args = parser.parse_args()
    report = verify(full=args.full)
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
