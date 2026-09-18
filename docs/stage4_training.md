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

The reproducible experiment and its measured results will be recorded here after the
implementation commit is trained from a clean Git state. Generated targets,
checkpoints, resume state, reports and figures live below `artifacts/stage4/` and are
ignored by Git; the implementation, command contract and measured summary are tracked.
