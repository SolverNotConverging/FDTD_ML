# Mixed-budget physics campaign

The completed eight-scene pilot motivates the next physics-search/distillation
cycle. The frozen starting checkpoint is
`artifacts/training_mixed_v7_3000/pretraining/best.pt` (epoch 70), trained on the
combined 3,000-scene corpus. The campaign runs under
`artifacts/physics_distillation_mixed_v7_pml_v2`. The original run remains preserved
in `artifacts/physics_distillation_mixed_v7` with a superseded status.

## Scope and budget separation

Select 16 training and four validation scenes per family: 128 training and 32
validation scenes across separated, contact, overlap, nested, single_dielectric,
dielectric_gap, single_pec and pec_dielectric. Selection is deterministic by scene
ID, excludes the eight pilot scenes, and does not filter by teacher feasibility.
The IID test split remains untouched.

Reuse the pretraining budget plan: each training scene has four fixed square
budgets (48, 64, 96, 128), two intermediate random squares and a random rectangle
plus its transpose. Validation adds 80², 112², 72×104 and 104×72; these remain
excluded from gradient updates but participate in model selection. All axes
remain in [48,128]. This gives 1,024 training and 384 validation pairs, or 1,408
requested targets before failures. Sparse samples receive 2× sampling weight in
distillation; because this campaign contains equal numbers of dense/sparse scenes,
the resulting sparse sampling share is approximately two thirds.

The launch audit found 1,396 cached teacher pairs; the other 12 requested pairs
will be recorded as missing-teacher outcomes. Physics eligibility can reduce the
final target count further. The launcher, physics, pilot and training checks
passed 33 tests before handoff.

## Search and numerical policy

Four GPU workers each process one balanced scene partition. Each pair compares
true uniform, quasi-uniform, heuristic, CNN, three pairwise density mixtures,
and two deterministic CNN density perturbations: at most 12,672 candidate
evaluations. Identical meshes within a pair reuse their solver result.

Converged reference arrays and their actual extended duration are reused. The
launcher verifies reference identity and numerical settings, binding reference
file hashes into the workflow. Dielectric averaging stays sampled, with 8–32
samples per axis and tolerance 0.001. True uniform snaps PEC and rounds PML
thickness to whole cells (nearest interface, half-cell ties thinner); all learned mesh
targets preserve PEC, source and receiver anchors. Projection has a 30-second
limit per axis; candidates are capped at 256 billion cell updates.

Rank settled, legal candidates with
`max(receiver waveform error, receiver spectrum error) + 0.02 * cost_ratio`,
where `cost_ratio` is cell updates relative to the matching true uniform run.
The snapped uniform benchmark is excluded from learned target selection. Save
the Pareto front and underlying errors/costs. Blend the selected projected
density 50:50 with the existing heuristic teacher target.

A missing teacher, candidate projection failure, or absence of an eligible
candidate remains an explicit outcome. Failed pairs are not silently regenerated
on resume. Merge only usable targets, save coverage, require every requested pair
to have an outcome and every split/family to have target coverage before training.

## Distillation and the previous timeout

After search, fine-tune the starting checkpoint for at most 20 epochs, batch size
4, learning rate 3e-4, patience 8, sparse sampling weight 2. The optional training
repair loss is disabled (`repair_weight=0`): this avoids the training-time MILP
projection that terminated the old v6 workflow. It does not fix the optimizer's
timeout or relax mesh constraints. Search targets still come from legal projected
meshes; validation attempts eight projections per epoch and records failures.
Budget-group and family validation metrics are retained in physics datasets.

## Operation

```bash
.venv/bin/python scripts/start_physics_campaign_v7.py
python3 scripts/start_physics_campaign_v7.py --status
```

The launcher detaches the coordinator and workers. `workflow.json` binds the
checkpoint, teacher targets, scene selection, per-scene budget plan, reference
hashes and code identity. `stage.json` tracks preparation/search/merge/training;
`progress.json` refreshes every ten seconds during search. Candidate details are
under `lane*/search/<scene>/<Nx>_<Ny>/result.json`; target coverage is saved to
`coverage.json`, and the final model to `distillation/best.pt`.

The same launch command resumes an unchanged campaign. Completed pairs and
explicit failed pairs are retained. A new configuration requires a fresh output
directory. A failed worker or missing split/family coverage stops the workflow
with `failure.json` instead of silently starting incomplete training.

## PML correction and restart

The original run reused the strict reference-grid PML alignment check in the
uniform candidate. This caused intermediate budgets to fail cost normalization.
The corrected baseline keeps constant spacing at the requested Nx/Ny and rounds
PML interfaces by at most half a cell, recording `pml_snapping` diagnostics. Exact
reference geometry/thickness and constrained-mesh policies are unchanged.

All 1,408 scheduled pairs passed uniform construction and PML/probe/geometry
checks. Nine previously failed uniform cases passed actual CUDA runs. An aligned
case reproduced its saved uniform grid and receiver waveforms (rtol 1e-6,
atol 1e-12). Results are saved under `artifacts/physics_uniform_pml_fix`.

The corrected run was launched with
`--reuse-from artifacts/physics_distillation_mixed_v7`. It copies successful pairs
only when all numerical inputs match, PML thickness is unchanged, and the new
uniform mesh equals the cached mesh exactly. Imported results retain their
original policy diagnostics and carry explicit `cache_origin` provenance.
`reuse.json` lists those pairs. Failed/unfinished pairs are retried under v2.
