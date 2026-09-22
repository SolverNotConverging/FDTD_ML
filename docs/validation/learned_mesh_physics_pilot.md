# Learned-mesh physics pilot

The first residual U-Net checkpoint was evaluated with CUDA FDTD on all 16 held-out
condition/budget pairs from the exact-axis label pilot. Each prediction was
projected to the requested 32, 48, 64, or 96 cells per axis and checked against the
3.0 grading cap. The same complex far-field plus log-RCS score and soft Nt exponent
0.1 used for candidate labels were applied without retraining or loss reweighting.

All 16 simulations settled on the initial 70 ns attempt. No predicted axis needed
grading repair.

| Split | Accepted | >=1.05x wins vs uniform | Median improvement | Minimum improvement | Median score / teacher |
|---|---:|---:|---:|---:|---:|
| Validation | 8/8 | 7/8 | 3.135x | 0.893x | 1.356x |
| Test | 8/8 | 8/8 | 2.641x | 1.311x | 1.259x |

The diagnostic gate requires all cases settled, at least 75% meaningful wins in
each split, median improvement above 1.2x, and median teacher gap below 1.5x. The
pilot passes all four checks. The CNN even beats the searched teacher in two cases,
showing that density-profile interpolation is not limited to selecting one of the
six discrete policies.

The main limitation is one epsilon_r=26 validation case at 48x48 whose learned
score is 0.893x the uniform improvement and 3.19x the best teacher score. The pilot
contains only 36 training labels, so it is evidence for the representation and
training path rather than the final model. The next model is trained on the full
qualified simple-curriculum labels and must repeat this physics evaluation on its
frozen validation/test splits.
