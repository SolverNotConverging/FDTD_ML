"""Finish a frozen C0 acquisition, then start resumable fresh CNN training."""

import argparse
import json
import time
from collections import Counter
from pathlib import Path

from scattermesh.campaign_v2 import atomic_json
from scattermesh.dataset_v2 import build_new_dataset
from scattermesh.training_v2 import train_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    run = args.run.resolve()
    launch = json.loads((run / "launch.json").read_text())
    deadline = float(launch["deadline_epoch_s"])
    summaries = [run / "data" / "workers" / f"shard_{index}.json" for index in range(4)]
    status_path = run / "finalization.json"
    project_status = run.parents[1] / "project_execution_status.json"
    while time.time() < deadline:
        if all(path.exists() and "finished_at" in json.loads(path.read_text()) for path in summaries):
            break
        time.sleep(30)
    else:
        atomic_json(status_path, {"status": "acquisition_deadline", "cnn_training_started": False})
        return

    dataset_path = build_new_dataset(run / "manifest.json", run / "data", run / "dataset")
    metadata = json.loads(Path(dataset_path).read_text())
    examples = metadata["examples"]
    splits = Counter(example["split"] for example in examples)
    families = {split: sorted({ex["scene"]["family"] for ex in examples if ex["split"] == split}) for split in ("train", "validation")}
    lineages = {split: len({ex["scene"]["lineage_id"] for ex in examples if ex["split"] == split}) for split in ("train", "validation")}
    materials = {split: sorted({ex["scene"]["material"]["kind"] for ex in examples if ex["split"] == split}) for split in ("train", "validation")}
    gate = {"status": "labels_built", "dataset": str(dataset_path), "split_examples": dict(splits), "split_lineages": lineages, "families": families, "materials": materials}
    if lineages["train"] < 48 or lineages["validation"] < 8 or any(len(families[s]) < 3 or len(materials[s]) < 2 for s in families):
        gate.update(status="insufficient_qualified_labels", cnn_training_started=False)
        atomic_json(status_path, gate)
        return
    if time.time() >= deadline:
        gate.update(status="acquisition_deadline", cnn_training_started=False)
        atomic_json(status_path, gate)
        return
    gate.update(status="training", cnn_training_started=True, started_at_epoch_s=time.time())
    atomic_json(status_path, gate)
    atomic_json(project_status, gate | {"authoritative_plan": "IMPLEMENTATION_PLAN.md"})
    result = train_model(dataset_path, None, run / "large_cnn", base_channels=64, device=args.device, deadline=deadline, max_epochs=80, curriculum_mode="c0_restart", legacy_fraction=0.0)
    gate.update(status="training_finished", result=result, finished_at_epoch_s=time.time())
    atomic_json(status_path, gate)
    atomic_json(project_status, gate | {"authoritative_plan": "IMPLEMENTATION_PLAN.md"})


if __name__ == "__main__":
    main()
