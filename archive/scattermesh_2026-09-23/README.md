# ScatterMesh archive

This directory preserves the source tree from commit `c514697d7134e41389098f3c28ef847d9f3bc203` on branch `codex/archive-scattermesh-20260923`, together with the legacy run outputs formerly stored at the project root in `runs/`.

The project-root `runs` path is a relative compatibility symlink to `archive/scattermesh_2026-09-23/runs`. Existing scripts that read or write through `runs/` therefore continue to resolve to these archived outputs.

The `source/` directory is a Git snapshot of the recorded commit. It excludes the preexisting `archive/receiver_cnn_2026-09-22` tree to avoid duplicating that archive. Environment package versions and the original virtual-environment configuration are recorded under `metadata/`. The active `.venv` was relocated for this project and reinstalled as an editable package; the preserved package versions and relocation details are recorded under `metadata/ACTIVE_ENV_RELOCATION.json`.

`metadata/runs_inventory.csv` records each archived run file's relative path, byte size, and SHA-256 digest; symlink entries, if any, record their link target. `metadata/archive.json` summarizes the source revision and run inventory. Historical run manifests and checkpoints were moved as-is and were not rewritten.

`metadata/RESTORATION_VERIFICATION.json` records successful load checks and
SHA-256 digests for representative receiver/scattering datasets, completed
checkpoints, and paused training states. The file inventories and relocated
receiver links were also fully checked after the move.

To restore the project-root compatibility path if needed, from the project root run:

```sh
ln -s archive/scattermesh_2026-09-23/runs runs
```

From the project root, `python scripts/verify_local_archive.py` checks all
relocated links and representative SHA-256 entries; add `--full` to hash every
receiver file and scattering run. The historical source snapshot can be
restored into a separate directory with
`cp -a archive/scattermesh_2026-09-23/source/. DESTINATION/`; keep the receiver
archive adjacent or update the recorded relative `artifacts` link there. The
original run manifests are preserved verbatim. Access saved artifacts via the
local `runs/` link, and use the v2 importer to create portable new manifests.

To use the archived source directly, work from `source/`; it is a snapshot and does not replace the active project tree.
