# Stage 3: procedural datasets, reference convergence and evaluation

Implemented 2026-09-18 for project version 0.3.0. The stage-three workflow is
implemented; reference acceptance remains a per-scene result, not a blanket
certification of the generated dataset. No trained model or training loop is included.

## Reproduce

From the repository root in PowerShell, after the normal uv setup and CUDA build:

```powershell
.venv\Scripts\python.exe -m fdtdmesh.data generate --output artifacts/stage3/manifest.json --per-split 4 --seed 2026
.venv\Scripts\python.exe -m fdtdmesh.data evaluate --manifest artifacts/stage3/manifest.json --output artifacts/stage3/evaluation --split all --demo-cnn
.venv\Scripts\python.exe examples/plot_stage3.py --evaluation artifacts/stage3/evaluation
.venv\Scripts\python.exe -m pytest -q --basetemp=artifacts/pytest-stage3-full
```

Use a new empty evaluation directory on each run. `--split test_iid --limit 2`
selects a smaller experiment. Omit `--demo-cnn` for uniform/heuristic only, or pass
`--checkpoint path.pt` for a compatible supplied model. Checkpoint SHA-256 and
metadata are recorded; random weights are labelled untrained. The test suite does
not require a supplied model.

Reference controls: `--levels 64 128 256 512`, `--tolerance 0.02`,
`--consecutive-passes 2`, and `--max-cell-updates 8000000000`. Additional observation,
phase, memory and meshing controls are exposed through `EvaluationConfig` in Python.
Larger levels require an appropriate work budget; they are not a substitute for
checking whether observables are actually converging.

## Portable scene and dataset contract

`data/schema.py` defines a strict dataclass-backed JSON scene record with:

- Schema version, unique scene ID, lineage group, random seed, split and family.
- Physical domain, frequency band, duration, solver precision and raster resolution.
- Exact candidate budgets and explicit CPML cell counts, physical thickness and profiles.
- Material parameters, ordered continuous primitives, physical current source waveforms
  and physical point/fixed-count line receivers.
- Minimum unanchored feature size in raster pixels.

Dataset manifests contain generator settings, ordered scene records, the mandatory
mesh policy and SHA-256 content identity. Reading verifies the content hash, schema,
scene validity and split integrity. Provenance records Git revision/dirty state,
source-tree hash, native extension hash, platform, GPU and library versions. The
hash of the physical scene record also accompanies each evaluated reference.
Changed physics or ordering changes the dataset identity. JSON rejects NaN/Inf;
scene identifiers cannot contain path separators. NPZ files contain plain numeric
arrays and can be read with `allow_pickle=False`.

This first generator uses bounded-size rectangles/circles/convex triangles/polygons,
anchored PEC lines, positive rectangular gaps, touching objects and mixed objects.
Shapes live inside vacuum collars. Unanchored bounding spans and rectangular gaps
must cover at least four 64x64 raster pixels; touching has exactly zero gap and is
allowed. Explicit anchors protect zero-width PEC lines. Generated polygon templates
avoid narrow slivers. Bounding-span validation is not a general minimum-feature
proof for arbitrary user-authored concave polygons or overlapping geometry.

## Split design and leakage policy

| Split | Distribution |
|---|---|
| train, validation, test_iid | Independent seeds; rectangle, circle, anchored line and gap families |
| test_compositional | Mixed rectangle/circle and touching PEC/dielectric objects |
| test_geometry_ood | Triangles and five-vertex polygons |
| test_material_ood | Dielectric objects with epsilon_r 8–12 and conductivity 0.05–0.15 S/m |
| test_scale_ood | Lx 36–48 mm and f_max 30–40 GHz |
| test_budget_ood | 80x80 and 112x112 total cells, absent from ordinary budgets |

Ordinary scenes use Lx 18–24 mm, Ly/Lx 0.8–1.2, f_max 12–18 GHz,
dielectric epsilon_r 1.5–4, conductivity 0–0.02 S/m, and budgets 64x64/96x96.
Materials are nonmagnetic in this generator. Source y varies, source x is 0.23 Lx;
three probes lie at x=0.77 Lx with y/Ly=0.35,0.5,0.65. Wider probe-placement and
aspect-ratio OOD studies remain possible extensions, not results of this experiment.

Every seed lineage stays in one split. Additionally, 64x64 normalized occupied
geometry masks are compared across splits, ignoring material contrast and physical
scale. Masks differing on at most 0.5% of domain pixels are rejected as near duplicates.
Generation resamples seeds deterministically until this check passes, with a bounded
retry limit. This conservative raster rule can reject distinct small shapes and is
not a mathematical guarantee against every near-duplicate transformation. It is
explicit, tested and rerun on manifest loading. Pairwise validation is O(N²), so
larger datasets will need indexing and a richer similarity policy. Multiple budgets
of one scene always stay together. The generator does not use test errors to choose
or tune scenes or densities.

## Reference refinement

Each reference is a globally uniform tensor-product grid, with 64,128,256,512 cells
per axis by default. Constant physical collars occupy one eighth of each axis;
their counts increase 8,16,32,64 as the grid refines. Material primitives, excitation,
probe coordinates, profile parameters and physical duration remain unchanged.
Generated line anchors lie on a dyadic lattice. Custom anchors that a uniform grid
cannot represent fail explicitly; no adaptive reference is substituted silently.

At each refinement, receiver data is resampled onto 1,025 common physical times
including exact zero initial fields and the same requested final time. The source
is an integrated 1 mA z-current Gaussian-modulated sine, sampled at solver half steps;
its width is 1/f_max, delay 4/f_max, carrier 0.4 f_max, and duration 12/f_max.
Each run uses its own conservative CFL timestep; normalization preserves integrated
current across the differing cell areas and dt values.

A refinement passes only if **every receiver's waveform and complex-spectrum
relative L2 change is <=2%**. Two consecutive passing refinements are required.
The finer of the final pair becomes the accepted reference. This is empirical
convergence of the chosen observables over the selected time/frequency range;
it is not an a priori bound on continuum error, late-time behavior, other probes,
or broadband PML accuracy. Point-sampled interfaces can converge nonmonotonically.
Independent analytical cavity and stage-two CPML tests remain in the full suite.

Statuses are `converged`, `nonconverged` after exhausting levels, or `failed` for
solver/mesh/resource errors. Only converged references enable candidate scoring.
All statuses, attempted levels, metrics and errors are saved. The latest arrays are
named `reference_latest.npz` even after failure; only `reference.json` status marks
acceptance. Do not use these files for training without checking that status.

Default limits reject runs above 8 billion cell updates, 256 MB of raw receiver
history, or a conservative 1 GB field/coefficient estimate. Field-size checks precede
meshing; work/history checks precede CUDA allocation. These are configured estimates,
not measurements of peak host or GPU memory. MILP search retains a 30-second per-axis
limit followed by LP refinement. No silent precision reduction or CPU fallback occurs.

## Metrics and fair baseline comparison

All methods share the same physical source and receiver coordinates, duration,
1,025-point observation grid and 12 frequencies linearly spanning f_min to f_max.
Linear temporal interpolation uses the known zero sample at t=0 and rejects any
extrapolation beyond the recorded end. The common observation frequency band is
checked against both solver and observation-grid Nyquist limits.

Waveform relative L2 is computed per receiver, normalized by the greater of the
reference norm and `1e-8 V/m * sqrt(number_of_samples)`. Reported means weight
receiver samples equally; convergence uses the maximum. Peak waveform error is
also recorded. Spectra use a common Hann window and trapezoidal time integral on
the aligned grid. Complex-spectrum L2 uses a floor of `1e-8 V/m * duration`.
Phase is the circular difference at frequencies where reference magnitude exceeds
1% of that receiver's spectral peak and the absolute floor, and candidate magnitude
exceeds the floor. RMS phase is in degrees; undefined quiet cases return null with
zero valid phase samples. Phase is diagnostic, not an additional convergence gate.

Candidate collars retain **eight cells per side** at fixed physical thickness for
all budgets. Consequently the `uniform` strategy means uniform interior density
projected onto exact anchors, collars and mandatory <=1.4 grading. It can differ
from a globally uniform grid, especially at larger budgets. The `heuristic` baseline
uses axis averages of refractive-index, conductivity, PEC and material-edge weights.
These are explicit baseline calculations, not additional CNN inputs or trained
teacher targets. The `cnn` strategy uses the existing raw nine-channel model and
checkpoint contract. Every strategy goes through the same hard-constrained mesher.

Reports retain meshes, waveform/spectral errors, phase, dt/Nt, cell updates,
GPU stepping time, total solver wall time, meshing wall time, correction metrics
and transfer counters. GPU times are single-run observations including launch
costs, not repeated isolated performance measurements. Different scenes should not
be treated as one Pareto frontier; compare methods at matched scene and budget.

## Measured validation

Hardware: Windows / RTX 4070 Laptop GPU / CUDA 13.3 / existing native stage-two
extension. **93 tests pass, no skips** (13.74 seconds in the recorded full run).
The 15 new tests include deterministic generation, JSON round-trip/tampering,
lineage and near-geometry leakage, unsafe IDs, underresolved features/gaps,
reference collar refinement, analytic amplitude/phase metric cases, time alignment,
consecutive convergence and reset after failure, resource rejection, nonconverged
reference exclusion, and real CUDA reference-to-three-baseline integration.

The initial 32-scene dataset (seed 2026, four scenes per split) has identity:

```text
ab894d00f950296c5e96e3724a250c555f93de96c894fe6b6473baa22892eb14
```

| Split | Converged | Nonconverged |
|---|---:|---:|
| train | 2 | 2 |
| validation | 2 | 2 |
| test_iid | 2 | 2 |
| test_compositional | 0 | 4 |
| test_geometry_ood | 2 | 2 |
| test_material_ood | 3 | 1 |
| test_scale_ood | 1 | 3 |
| test_budget_ood | 3 | 1 |
| **Total** | **15** | **17** |

All **90 candidate runs** (15 scenes x 2 budgets x 3 strategies) succeed and report
zero transfers during native stepping. Across those 30 matched scene/budget pairs,
median waveform relative L2 is 0.00797 for uniform preference, 0.01069 for the
heuristic and 0.00590 for the random CNN. Median cell updates are 13.13, 15.17 and
13.42 million respectively. These small, convergence-selected results are workflow
validation, not evidence of a trained advantage. No compositional baseline accuracy
can be claimed from this experiment because none of its references passed.

The main remaining scientific task is increasing accepted reference coverage,
especially PEC/compositional and larger-electrical-size cases, without loosening
acceptance merely to admit more scenes. Extend refinement/work budgets where feasible,
inspect staircase convergence and PML sensitivity, expand seeds and test bands, and
retain coverage/failure reports before scaling teacher training.

Outputs for this run are under `artifacts/stage3/all_splits/`: full JSON report,
Markdown table, per-scene arrays, `scene_gallery.png`, and `accuracy_cost.png/.svg`.
Figures were visually inspected. Generated data and checkpoints are ignored by Git;
the schema, generator, evaluator, tests, plotting script and measured record are tracked.
