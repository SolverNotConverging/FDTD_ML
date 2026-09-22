"""Audit and combine immutable accepted reference campaigns for later training."""

import argparse
import fcntl
import json
from collections import Counter
from pathlib import Path

import numpy as np

from fdtdmesh.evaluation.references import _write_json

from .schema import geometry_signature, near_geometry, read_manifest, write_manifest


def reserve_geometry(output, scene):
    """Serialize cross-lane split checks before paying for an FDTD reference.

    Rejected candidates keep their reservations, making interruption/resume safe.
    The final manifest independently validates all published split assignments.
    """
    output = Path(output)
    with (output / "geometry.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = output / "geometry_reservations.json"
        rows = json.loads(path.read_text()) if path.exists() else {}
        mask = geometry_signature(scene)
        record = dict(
            split=scene.split, scene_hash=scene.content_hash, mask=np.packbits(mask).tobytes().hex()
        )
        if scene.scene_id in rows:
            if rows[scene.scene_id] != record:
                raise ValueError("Reserved scene identity changed")
            return True
        for other in rows.values():
            if scene.split != other["split"]:
                other_mask = np.unpackbits(
                    np.frombuffer(bytes.fromhex(other["mask"]), dtype=np.uint8)
                )
                if near_geometry(mask, other_mask.astype(bool)):
                    return False
        rows[scene.scene_id] = record
        _write_json(path, rows)
        return True


def audit_campaign(campaign):
    campaign = Path(campaign).resolve()
    manifest, scenes = read_manifest(campaign / "accepted_manifest.json")
    complete = json.loads((campaign / "complete.json").read_text())
    config = json.loads((campaign / "campaign.json").read_text())
    counts = dict(Counter(s.split for s in scenes))
    if (
        complete["dataset_id"] != manifest["dataset_id"]
        or complete["accepted"] != len(scenes)
        or counts != config["targets"]
    ):
        raise ValueError("Reference campaign completion/count identity differs")
    decisions = {}
    for path in campaign.glob("lane*/decisions/*.json"):
        row = json.loads(path.read_text())
        if row["status"] == "converged":
            if row["scene_id"] in decisions:
                raise ValueError("Duplicate accepted scene decision")
            decisions[row["scene_id"]] = row
    references = {}
    for scene in scenes:
        row = decisions[scene.scene_id]
        path = campaign / row["reference"]
        reference = json.loads(path.read_text())
        if (
            row["scene_hash"] != scene.content_hash
            or reference["scene_hash"] != scene.content_hash
            or reference["status"] != "converged"
        ):
            raise ValueError(f"Invalid accepted reference: {scene.scene_id}")
        if max(reference["accepted_budget"]) > 2048:
            raise ValueError("Accepted reference exceeds the 2048 ceiling")
        with np.load(path.parent / "reference_latest.npz") as arrays:
            if any(not np.isfinite(arrays[key]).all() for key in ("times", "waveforms", "spectra")):
                raise ValueError(f"Nonfinite reference: {scene.scene_id}")
        references[scene.scene_id] = str(path.parent.resolve())
    return (
        manifest,
        scenes,
        dict(dataset_id=manifest["dataset_id"], counts=counts, references=references),
    )


def merge_campaigns(campaigns, output):
    """Publish a new dataset identity; retain original scene hashes and references."""
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / "merge.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        scenes, bindings, parents = [], {}, []
        for campaign in campaigns:
            manifest, items, audit = audit_campaign(campaign)
            parents.append(
                dict(path=str(Path(campaign).resolve()), dataset_id=manifest["dataset_id"])
            )
            for scene in items:
                if scene.scene_id in bindings:
                    raise ValueError("Duplicate scene across campaigns")
                bindings[scene.scene_id] = audit["references"][scene.scene_id]
                scenes.append(scene)
        scenes.sort(key=lambda s: s.scene_id)
        identity = dict(
            kind="combined_reference_campaign",
            parents=parents,
            targets=dict(Counter(s.split for s in scenes)),
        )
        config_path = output / "campaign.json"
        if config_path.exists() and json.loads(config_path.read_text()) != identity:
            raise ValueError("Combined dataset identity changed; use a new output")
        # Full cross-corpus split validation happens before publishing completion.
        manifest = write_manifest(
            output / "accepted_manifest.json",
            scenes,
            generation=dict(selection="union_of_converged_campaigns", parents=parents),
        )
        (output / "lane0" / "decisions").mkdir(parents=True, exist_ok=True)
        for scene in scenes:
            _write_json(
                output / "lane0" / "decisions" / f"{scene.scene_id}.json",
                dict(
                    scene_id=scene.scene_id,
                    scene_hash=scene.content_hash,
                    split=scene.split,
                    status="converged",
                    reference=str(Path(bindings[scene.scene_id]) / "reference.json"),
                ),
            )
        _write_json(config_path, identity)
        _write_json(
            output / "complete.json", dict(dataset_id=manifest["dataset_id"], accepted=len(scenes))
        )
        return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaigns", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = merge_campaigns(args.campaigns, args.output)
    print(f"Combined {len(manifest['scenes'])} converged scenes: {manifest['dataset_id']}")


if __name__ == "__main__":
    main()
