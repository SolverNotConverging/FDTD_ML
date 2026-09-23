# Mesh-CNN v2: C0–C2 pilot and measured sizing gate

## Project and reproducibility

The active Git branch is `codex/mesh-cnn-v2` in
`/home/s2307298/projects/FDTD_ML`. The original source revision is
`c514697d7134e41389098f3c28ef847d9f3bc203` on
`codex/archive-scattermesh-20260923`; its source, run inventory, dependency
records, checkpoints, logs, and paused states are in
`archive/scattermesh_2026-09-23/`. The receiver archive is adjacent. All 404
receiver links now use relative paths, with original manifests unchanged and
relocation recorded in `RELOCATION.json`. The scattering source archive also
records its relocated `artifacts` link. The historical `runs/` name is a local
relative symlink; new results use `runs_v2/`. The old Codex worktree is absent.

The active `.venv` activation scripts and editable package metadata point to
this directory. Dependency versions and the environment relocation record are
under the scattering archive's `metadata/` directory. Historical manifests
remain verbatim and may mention the former pathname as provenance; the v2
importer maps archived artifacts to local relative paths.

## Numerical and learning contract

The physical model remains 2D TMz scattering on a 1.2 m square, with a
tensor-product nonuniform Yee grid. Frequencies are 0.8, 1.0, and 1.2 GHz.
New scenes cover circle, ellipse, rectangle, triangle, convex and concave
polygon, star, and smooth-lobed families. C0 varies primitive size and
position; C1 adds rotation and aspect ratio; C2 adds corners, concavity, and
curvature changes. New materials are 75% dielectric (`epsilon_r` 2, 4, 8, 12;
lossless or lossy) and 25% PEC. Occupied area is 0.5–8%, with at most 35%
support on either axis. The numerical solver explicitly rejects unrepresented
PEC split edges and never moves or closes geometry to force a mesh to pass.

The intended corpus has 256 independent geometry lineages, or 128 if profiling
shows 256 exceeds the data allocation. Complete lineages are split near
75%/12.5%/12.5% into train/validation/test, stratified by family and material.
Each has two incidence angles and exact square budgets 32, 48, 64, and 96.
Reference checks escalate uniform grids through 192, 256, and 384 cells per
axis, then 512 if needed. Accepted references require a field tail ratio below
`1e-5`, adjacent reference change at most 0.5%, and independent duration,
material quadrature, contour, and PML checks at most 0.5%. Circle analytic
fields provide an additional check. Reference uncertainty is retained.

For each qualified condition, the teacher search has up to 12 meshes: uniform,
four geometry-based densities, four seeded smooth densities, and three local
refinements of the best valid first-round mesh. Candidate axes, complex spectra,
raw errors, stable time step, status, and numerical fingerprint are saved.
Multiple valid profiles become targets. Historical training data are imported
with path mapping and rescored; 25% of training draws are historical. The
previously inspected historical test sets are development evidence, while the
new test lineages remain frozen until selection is complete.

The current portable import contains 1,104 accepted historical training
examples. Its source paths resolve within the scattering archive; no historical
validation or test entries were imported.

The accuracy loss retains normalized complex-field error plus 0.25 times
floored log-width error. Teacher selection uses

`J = L_accuracy * (dt_uniform / dt_candidate)**0.05`.

Candidate records support sensitivity analysis at exponents 0, 0.02, 0.05,
and 0.1. Model selection and the primary results use **raw physical accuracy**.
The large five-level residual U-Net has widths 64/128/256/512/512,
GroupNorm, SiLU, skip connections, bottleneck conditioning, and eight-head
attention at the lowest resolution. It takes nine 512×512 maps and emits
positive x/y density profiles; deterministic inverse-CDF projection enforces
exact cell counts, positive widths, and adjacent ratio at most 3. The small
comparison has base width 16 and identical inputs, targets, and splits. Both
initialize afresh. The training contract uses AdamW (3e-4, weight decay 1e-4),
FP16, effective batch 32, and a measured microbatch below 20 GiB. Epochs are
20 C0, 20 cumulative C0–C1, then up to 80 cumulative C0–C2 with patience 20.
Three best profile checkpoints require validation FDTD before a checkpoint is
frozen for testing.

## Measured sizing gate, 23 September 2026

The 16 representative shape/material rows were profiled with four TITAN RTX
GPUs. The profile ran for 109 seconds, included estimated cut-cell time steps
and measured 32/192-cell uniform runs where bounded. The large-model FP16
memory probe admitted microbatches 1, 2, 4, and 8, with batch 8 peaking at
16.16 GiB; the required choice would be 8.

Only nine rows had a complete compatible projection: eight dielectric and one
PEC. Seven PEC rows had at least one reference or candidate grid rejected by
the current conformal representation or exceeded a bounded profiling step
cap. At 192 cells, four of eight dielectric exemplars remained unsettled at
70 ns, and five of eight PEC exemplars exceeded the 50,000-step profiling cap.
Seven PEC exemplars had an incompatible tested grid at some budget or policy.
For the compatible rows alone, the projected data-phase elapsed time is
**18.16 hours for 128 lineages** or **36.31 hours for 256 lineages**
with all four GPUs. The projection assumes 140 ns candidate runs and four
384-cell independent reference probes; using longer durations where needed
would raise the cost. This is a partial projection under the 140 ns assumption,
not a rigorous lower bound or completed-campaign estimate. The saved machine record is
`runs_v2/profile_c0_c2/profile.json`.

A separate complete reference smoke for a dielectric circle needed the 512-cell
level. Its four independent probes passed after placing the monitor within
both PML thicknesses; the maximum reference variation was 0.1853%. This
escalation is additional evidence that the projection above is optimistic.
Short 32-cell star dielectric and smooth-lobed PEC cases agreed between the
CPU and CUDA backends within 5e-16 relative complex-field L2.

The approved data allocation is 14 hours, so neither balanced corpus is
eligible for bulk launch. No new teacher corpus, v2 checkpoint, frozen v2
test result, or positive accuracy claim exists yet. The launch command writes
`campaign_status.json` with `sizing_gate_stopped` and does not dispatch bulk
workers. Numerical incompatibility, insufficient settling, and budget
exhaustion are distinct statuses in the saved profile and case records.

Before a bulk run, revise and re-profile the reference cost and PEC scene/grid
compatibility without silently changing boundary locations or weakening the
reference gates. A larger time allocation is another option, but it does not
resolve incompatible PEC grids. Freeze the revised protocol and size before
creating teacher labels or exposing the test set.

## Commands and intended deliverables

```bash
.venv/bin/python -m pytest -q
.venv/bin/python scripts/run_mesh_cnn_v2.py launch --output runs_v2/c0_c2_pilot
```

If a future profile passes the sizing gate, the runner freezes a manifest,
imports qualified historical training examples, dispatches four GPU FDTD
workers, builds portable set-valued targets, trains large and small models on
separate GPUs, selects checkpoints by validation FDTD, and evaluates frozen
new test lineages. It enforces a 24-hour overall deadline and preserves
resumable state and incomplete statuses. Scientific success remains separate:
on new dielectric test conditions at least 75% must improve raw joint accuracy
by 5% or more and the median improvement must reach 1.2×. PEC is reported
independently. A failed gate requires a remediation report without retuning on
the frozen test set.

The report contract includes complex-field and width error versus cell count,
same-budget improvement by family and material, minimum tested cells meeting
both 2%, 5%, and 10% targets, cell-saving ratios, reference uncertainty,
time step, update count, runtime, memory, large-versus-small comparison,
penalty sensitivity, representative meshes and scattering curves, and counts
for incompatible, unconverged, budget-limited, and incomplete cases. Later
curriculum stages C3–C5 cover pairs, small gaps, and 3–10-object scenes; C6–C7
cover holes, cavities, thin sections, and multiscale details; C8–C9 are frozen
unseen engineering silhouettes and distributed scenes. Independent solver
comparisons, repeated seeds, larger tests, and confidence intervals precede
manuscript claims.
