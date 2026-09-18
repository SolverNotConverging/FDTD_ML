# Stage 4: heuristic-teacher imitation

Implemented 2026-09-18 in package 0.5.0. This stage trains the existing nine-channel,
FiLM-conditioned ResU-Net to imitate legal meshes derived from the material/edge
heuristic. It does not claim that the heuristic is electromagnetically optimal.

## Target contract

For every selected scene and supported budget, target generation performs four
deterministic operations:

1. Rasterize the physical scene using the checkpoint input contract.
2. Construct the material/edge heuristic axis densities.
3. Pass those densities through the exact-budget deterministic mesher, including
   anchors, fixed PML collars, and adjacent-cell grading <=1.4.
4. Rebin equal mass for every learned interior cell into the CNN axis pixels.

The stored target is a normalized probability mass per raster bin. Fixed PML-only
regions carry no learned mass. Each target record contains scene ID/hash, split,
budget, and teacher projection diagnostics. The companion JSON binds the target file
to the dataset content identity, manifest hash, mesh policy, target hash, code
provenance, and teacher version. Modified or mismatched target files are rejected.

## Training contract

The primary loss is mean squared error between cumulative predicted and teacher
axis mass, averaged over x and y. Cumulative mass reflects the quantile operation
that converts density into line positions and is invariant to multiplying either
axis density by a constant. Every fourth batch by default also projects detached CNN
densities through the deterministic mesher and applies the existing repaired-CDF
loss with weight 0.05. Integer meshing and FDTD remain outside autograd.

Training uses AdamW, seeded epoch-specific shuffling, explicit train/validation
splits, and a configurable model width. Each epoch records train imitation/repair
loss, validation axis losses, and deterministic projection corrections for a bounded
validation subset. The best validation checkpoint includes model weights, raster and
mesh contracts, Git commit, dataset identity, teacher/training versions, configuration,
and validation metrics. A separate resume file stores optimizer state and history.
Resuming permits an increased epoch count; all other settings and the dataset must match.

## Evaluation contract

Held-out evaluation reports CDF imitation loss and deterministic line correction on
the requested split. It also runs a configurable number of scene/budget pairs through
the real CUDA solver using both heuristic and learned meshes and compares their
receiver waveforms and complex DFTs over a shared physical window. These are
teacher-agreement measurements. Only comparison against accepted Stage-3 reference
solutions can establish electromagnetic accuracy, and the present difficult v3
dataset does not yet have broad accepted-reference coverage.

## Measured run

Implementation commit `70c1a02` was trained from a clean Git state on the v3 dataset
`10c7b9dcaf1c9b47bf723d33c39bf354d659c9b7b63500b84446a8d4ebc1266f` using an
RTX 4070 Laptop GPU. The target set contains 192 scene-budget samples: 32 scenes and
two budgets from each of train, validation, and IID. Exact teacher projection took
55.70 seconds.

Training used width 16, batch size 4, AdamW learning rate 1e-3, repair weight 0.05
every fourth batch, and seed 2026. It ran 40 epochs in two resumable invocations,
about 312 seconds combined. The selected epoch-40 checkpoint has SHA-256:

```text
2a634488e298e1f6d22b0be5724901ce6fc574f565ab2429a4778c1e74648058
```

Best validation CDF MSE is 2.621e-5. On all 64 held-out IID scene-budget samples,
CDF MSE is 3.783e-5 versus 5.634e-4 for the seeded untrained network, a 14.9x
reduction. All 64 CNN densities produce legal exact-budget meshes. Across those
projections, normalized mean line correction averages 0.0155% in x and 0.0203% in y;
the largest per-sample mean correction is 0.0612% in x and 0.0964% in y. These small
repair values do not directly predict electromagnetic agreement.

Real CUDA FDTD compared learned and teacher meshes for the first budget of 16 held-out
IID scenes. Training improves both waveform and spectrum agreement in 11/16 scenes.
Median waveform disagreement falls from 5.50% for seeded random weights to 3.45%;
median complex-spectrum disagreement falls from 6.05% to 4.78%. The mean moves in
the opposite direction: waveform 6.37% to 7.62%, spectrum 11.81% to 13.61%. One
mesh-sensitive scene (`test_iid-00001`) dominates: normalized mean line displacements
are only 0.167% in x and 0.253% in y, yet waveform and spectrum disagreement are
67.96% and 140.78%. Its three dielectric constants are 2.10, 3.37, and 8.06 and its
window is 3.94 ns; no extreme material value explains the sensitivity.

This is the useful Stage-4 result: imitation learns the teacher density distribution
and often improves physical agreement, but lower density loss is not a reliable EM
objective. Stage 5 should use converged receiver physics to select/distill targets,
with sensitive outliers retained rather than averaged away. The comparison is against
the heuristic teacher, not a converged reference, so it establishes neither absolute
accuracy nor superiority to uniform meshing.

Generated targets, checkpoints, resume state, raw reports, the seeded-untrained
control and `training_and_repair.png` live below `artifacts/stage4/` and are ignored
by Git. The implementation, command contract, hashes, and measured summary are tracked.
