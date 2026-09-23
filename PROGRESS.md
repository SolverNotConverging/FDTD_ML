# Mesh-CNN v2 progress

Updated 23 September 2026. This report covers the active project in `/home/s2307298/projects/FDTD_ML` on `codex/mesh-cnn-v2`. The previous scattering project's original progress report is preserved at `archive/scattermesh_2026-09-23/source/PROGRESS.md`.

## Current status

**Implementation and local verification are complete for the C0–C2 pilot framework. The approved bulk campaign has not started.** The required profiling gate found no complete balanced 128- or 256-lineage campaign within the 14-hour data allocation under the present geometry/grid rules. The automated runner records `sizing_gate_stopped`; no v2 teacher corpus, trained CNN checkpoint, frozen new-lineage test result, or manuscript-level accuracy claim exists yet.

## Completed work

- Preserved the original scattering source at commit `c514697d7134e41389098f3c28ef847d9f3bc203` on `codex/archive-scattermesh-20260923`. Moved 59,178 legacy run files (1,420,527,328 bytes) to `archive/scattermesh_2026-09-23/runs` with a SHA-256 inventory and a local `runs/` compatibility link. New execution outputs use `runs_v2/`.
- Preserved the adjacent receiver-CNN archive. Replaced all 404 former-worktree absolute links with resolving relative links and recorded the mapping separately while leaving original inventories and historical manifests unchanged. Full inventory checks found no file or link mismatch. Representative receiver and scattering datasets, completed checkpoints, and paused training states opened successfully; their checksums and keys are recorded in `archive/scattermesh_2026-09-23/metadata/RESTORATION_VERIFICATION.json`.
- Relocated the active virtual-environment activation scripts and executable paths, preserved the recorded numerical package versions, and reinstalled `scattermesh` editably from this project. Activation, `python`, `pytest`, NumPy, SciPy, PyTorch, and the package import resolve locally. The former Codex worktree is absent. The project `.codex` entry is an empty read-only tmpfs mount using 0 bytes.
- Added continuous ellipses, polygons, rotated rectangles, and smooth-lobed shapes to conformal intersections and material sampling, with explicit rejection of unsupported PEC split edges. Implemented deterministic C0–C2 scene generation, grouped lineage splits, nine-map rasterization, conditioning, exact-budget candidate projection, and weak time-step teacher scoring.
- Added reference qualification with 192/256/384/512 escalation and independent duration, quadrature, contour, and PML probes. Added restartable FDTD case records with separate numerical fingerprints, saved axes and spectra, physical errors, execution status, time step, runtime, and CUDA memory.
- Imported 1,104 accepted historical training examples into a portable local dataset. Its provenance points into the local archive and its teacher scores were recalculated at exponent 0.05. No historical validation or test examples entered the import.
- Implemented the large 26,850,497-parameter residual U-Net and the 1,683,761-parameter width-16 comparison, set-valued distillation, FP16 memory probing, staged training with saved optimizer/RNG state, validation FDTD checkpoint selection, frozen test evaluation, and report/figure generation. These later stages remain unrun because the data gate failed.

## Measured numerical and resource evidence

- Four TITAN RTX GPUs were available. The 16-scene shape/material profile took 109 seconds. Only nine rows had complete compatible projections: eight dielectric and one PEC. Seven PEC examples had an incompatible tested grid at one or more budgets/policies. At 192 cells, four dielectric examples were unsettled after 70 ns, and five PEC examples exceeded the 50,000-step profiling cap.
- Under the recorded 140 ns candidate assumption, the compatible rows alone project to **18.16 elapsed hours for 128 lineages** or **36.31 hours for 256 lineages** on four GPUs. These are partial projections, not complete estimates or rigorous lower bounds. The data allocation is 14 hours, so the runner stopped before bulk work. The machine record is `runs_v2/profile_c0_c2/profile.json`.
- A dielectric-circle reference smoke required 512 cells per axis and passed separate duration, quadrature, contour, and PML checks. Its largest measured reference variation was 0.1853%. At 32 cells, new star dielectric and smooth-lobed PEC CPU/CUDA complex far fields agreed within `5e-16` relative L2.
- The main CNN's FP16 memory probe admitted microbatches 1, 2, 4, and 8 below 20 GiB. Microbatch 8 peaked at 16.16 GiB allocated GPU memory. This verifies memory fit for a training step, not trained-model quality or six-hour training completion.
- Final local checks: **134 tests passed, 6 skipped**. Focused Ruff lint and formatting checks passed. The archive verifier checked all 404 relocated links plus representative file hashes; full receiver and scattering run inventories were also audited.

## Remaining work and decision

1. Diagnose PEC incompatibilities by geometry family, grid budget, and policy. Keep exact boundaries and explicit rejection; qualify any new numerical treatment before using it for labels.
2. Measure settled reference and teacher costs under a revised protocol. Re-profile all 16 shape/material rows and freeze a balanced 128- or 256-lineage size only when the complete data phase fits the allocation. A longer allocation is possible but does not by itself fix incompatible PEC grids.
3. Once the sizing gate passes, execute the 24-hour campaign: references and candidates, portable targets, both CNN trainings, validation FDTD selection, frozen test evaluation, and the prescribed accuracy/cell-saving report. Keep incomplete and unconverged outputs out of labels and report PEC separately.
4. If the positive dielectric gate fails, write a remediation report without changing the frozen test set. Later C3–C9 stages and manuscript claims remain pending independent solver comparisons, repeated seeds, larger tests, and confidence intervals.

The active contract is [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md); detailed assumptions and gate results are in [`docs/v2_pilot.md`](docs/v2_pilot.md). The current automated decision is in [`runs_v2/c0_c2_pilot/campaign_status.json`](runs_v2/c0_c2_pilot/campaign_status.json).
