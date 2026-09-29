# Implementation plan: surrogate-assisted feasible mesh search

Prepared 29 September 2026 against source revision `3ec90ca`.
Target execution environment: **Linux server with NVIDIA TITAN RTX**, confirmed
by the user. Status: proposed work; this document does not implement the new API
or claim server validation or surrogate performance.

## Objective and scope

Add `strategy="surrogate_local"` to `optimize_mesh`. A small Gaussian process
(GP) will rank valid candidate meshes before one is evaluated by native CUDA
FDTD. Fit a separate GP during each geometry/budget/physics search. Geometry
processing, mesh projection, GP fitting, and acquisition run on the server CPU;
the TITAN RTX runs FDTD, DFT accumulation, and near-to-far conversion.

The research question is whether this selection achieves the same complex
far-field accuracy with fewer candidate solves and less total search time than
the existing feasible-local optimizer. A qualified numerical reference remains
necessary to measure each candidate's error. Removing that reference requirement,
training across geometries, and CNN meshing are outside the first implementation.

Preserve exact geometry, the existing boundary operator and stability bound,
fixed cell counts, anchors, grading, source, frequencies, and observation angles.
Retain hybrid optimization by default and allow an explicit strict-conformal
override. Do not relax fallback limits or reference qualification to make a
campaign complete. Keep the existing `feasible_local` strategy available.

The current tip-and-block searches provide a starting benchmark: 1.845% to
0.738% error at 120 x 120, and 1.458% to 0.792% at 160 x 160, each with 40 trial
evaluations. The star reference did not qualify. These results motivate the
experiment, but do not predict GP performance. See
[the measured study](hybrid_tip_optimization.md).

## Existing components to reuse

| Component | Existing responsibility | Planned change |
|---|---|---|
| `src/fdtdmesh/optimization/feasible.py` | Seed-relative proposals, projection, repair, backtracking, exact validation | Add stable mesh encoding and bounded candidate-pool support; preserve proposal semantics |
| `src/fdtdmesh/optimization/optimize.py` | Baselines, evaluation, parent pool, budgets, checkpoints, winner validation | Add surrogate selection and pool-random control through a shared local-search path |
| `src/fdtdmesh/optimization/common.py` | Physical/implementation identities, errors, caching | Reuse error definition and exact cache matching; add execution provenance where necessary |
| `src/fdtdmesh/optimization/reference.py` | Spatial, temporal, boundary and fallback qualification | Reuse unchanged numerical requirements |
| `src/fdtdmesh/storage.py` | Atomic HDF5 writes | Reuse for optimizer state and observations |
| `setup.py` | Linux/Windows CUDA extension build | Explicitly target TITAN RTX; qualify Linux build |

Do not change proposal radii, introduce cost penalties, or reallocate witness
indices during the first comparison. Those changes would make it harder to
identify the contribution of surrogate selection.

## Milestone 0: qualify the TITAN RTX server

TITAN RTX is a Turing device; compile for compute capability 7.5 (`sm_75`). The
current build defaults to `FDTDMESH_CUDA_ARCHS=89`, which targets the laptop and
must be overridden. NVIDIA documents Turing compilation with `compute_75` and
`sm_75` in its [compatibility guide](https://docs.nvidia.com/cuda/turing-compatibility-guide/index.html).
CUDA 13.0 retains Turing support; record the actual toolkit and matching driver
used rather than assuming any installed CUDA is sufficient.
[CUDA release notes](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/index.html).

The solver uses conditional WHILE graph nodes. Extension import and device
detection alone do not test this execution path. Include a real short solve
that creates, instantiates, and executes the graph. Consult the
[CUDA graph documentation](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html)
if the installed toolkit/driver rejects it. Do not silently replace the stopping
controller or numerical method during the surrogate experiment.

Record GPU name/UUID, available VRAM, driver, toolkit, host compiler, OS, Python,
CPU/thread settings, source revision, lockfile hash, native build flags, and
NumPy/SciPy/scikit-learn versions in `environment.json`. Distinguish the CUDA
version reported by `nvidia-smi` from the installed `nvcc` toolkit.

Suggested commands for the existing source, from a clean Linux checkout with
the required toolkit/compiler and `uv` already available:

```bash
nvidia-smi
nvcc --version
c++ --version
uv sync --locked --group dev
FDTDMESH_BUILD_CUDA=1 FDTDMESH_CUDA_ARCHS=75 \
  .venv/bin/python setup.py build_ext --inplace --force
.venv/bin/python -c 'from fdtdmesh.solver.tmz import cuda_backend; cuda_backend(); print("Native backend available")'
.venv/bin/python -m pytest tests/test_gpu_scattering.py -q -rs
.venv/bin/python -m pytest -q -rs
```

Select an allocated device with `CUDA_VISIBLE_DEVICES` before starting Python
when required by the server. Respect a scheduler's existing allocation. These
commands are a qualification procedure, not evidence that this server has been
tested. The backend probe must succeed and the GPU tests must execute, not skip.
Retain the existing precision policy and test both supported precisions; do not
switch to lower precision merely to improve TITAN timings.

Add a reproducible `scripts/build_cuda.sh` helper and a machine-readable preflight
script as implementation deliverables. A failed preflight stops GPU campaign
dispatch while CPU model and checkpoint development can continue.

Compare a small cylinder case and one identical tip mesh with the laptop results,
using identical physical inputs and precision. Investigate discrepancies against
the existing numerical tolerances. Requalify campaign references on the server.
Current identities hash the native `.pyd`/`.so` implementation, so Windows caches
must not be relabelled as matching Linux caches. Geometry recipes and axis arrays
can be transferred as experiment inputs.

## Milestone 1: define the surrogate's input and target

### Mesh input

Use the **actual mesh after projection and repair**, not its requested random
controls. Encode the displacement of free lines relative to one immutable seed:

\[
u_i^x=(x_i-x_i^{seed})/h_i^{seed},\qquad
u_j^y=(y_j-y_j^{seed})/h_j^{seed}.
\]

Reuse the local dual-width scale `FeasibleSpace.axes[axis].h`. Divide each axis
block by the square root of its number of free lines, then concatenate. This
makes Euclidean distance reflect axis RMS displacement rather than line count.
Omit fixed coordinates and empty blocks. Save the seed, fixed indices, scale,
feature version, and ordering; do not recompute these from changing parents.

Every observation must have the same counts and fixed coordinates as this search.
Baselines with another witness-index allocation remain comparison results but
are excluded from GP training. Exact candidate-axis hashes identify duplicates.
Near-duplicate meshes may have different cut topology: do not merge them merely
because their feature distance is small. If there are no movable lines, return
the measured seed with an explicit termination reason.

This representation can contain hundreds of coordinates. The model stays small
by using a shared kernel length scale and at most 80 initial observations, not
by claiming that constraints reduce the input to a known low-dimensional space.
The existing six-control smooth/local proposals provide correlated moves. Test
whether the resulting distance is informative before adding PCA or more features.

### Error target and GP

Use the current frequency-balanced relative complex far-field error from
`optimization.common.errors`, with target `log(max(error, 1e-12))`. Keep the raw
error, per-frequency error, worst-frequency error, and scattering-width error in
the report. The floor is numerical protection, not an accuracy tolerance.

Use scikit-learn `GaussianProcessRegressor` with a constant-amplitude Matérn 3/2
kernel, one shared length scale, normalized targets, and an initial diagonal
regularizer of `1e-6` in normalized target units. Bound hyperparameter fitting,
use a deterministic model RNG, and record fitted parameters. Refit after each
new successful unique observation. Limit numerical retries; record a fit failure
and use random valid selection when fitting or prediction remains non-finite.
Constant targets also trigger random selection until useful variation appears.

Treat the posterior standard deviation as uncertainty within this model. It is
not a calibrated physical error bound, and the regularizer is not reference
uncertainty. Donor/fallback changes can make the response nonsmooth. Use the
[official GP implementation](https://scikit-learn.org/stable/modules/gaussian_process.html)
instead of implementing a new linear algebra backend.

Add scikit-learn as an optional `surrogate` dependency and update `uv.lock` during
implementation. Ordinary meshing and `feasible_local` imports must continue to
work without it. A requested surrogate run should give a clear installation
message when that optional dependency is absent.

### Observation eligibility

Train only on successfully converged, unique meshes from the matching search
space, scored against the same qualified reference. Failed preparation, failed
time convergence, and unsolved proposals are not error labels. Preserve their
diagnostics separately; never regress on the existing `1e6` rejection sentinel.
Do not transfer historical strict-method labels into the current hybrid search.

## Milestone 2: add candidate selection

1. Evaluate the same three baselines as the existing optimizer. Require the
   geometry-aware seed to solve successfully before local search.
2. Collect 12 successful unique in-space observations, counting the seed, using
   the original one-at-a-time random feasible-local proposal policy. Other valid
   in-space baselines may count toward 12. If the budget ends first, retain the
   measured best result and report that warm-up was incomplete.
3. Fit the GP. Build a pool of up to 16 unique valid unsolved candidates using the
   existing parent selection, move modes, projection, repairs, and validator.
4. On 20% of selections, choose uniformly from that pool. Otherwise maximize
   expected improvement in **log error**, relative to the best measured eligible
   observation. Use the standard minimization formula, handle zero variance,
   and apply deterministic hash-based tie breaking. A rejected or unrelated
   baseline must not define the GP's incumbent.
5. Run FDTD for exactly one selected candidate. Only its measured result can
   update the reported winner, parent population, or GP observations. Update
   parent/radius state once per outer selection, not for every screened mesh.
6. Checkpoint and repeat until a budget is exhausted. Keep the existing stricter
   winner validation as a separately accounted evaluation.

Use the same pool machinery with uniform selection for a `pool_random` benchmark
control. It must use the same warm-up, candidate generation, repair, parent and
radius rules as `surrogate_local`; only selection/model work differs. Different
chosen parents will naturally cause later candidate sequences to diverge.

Bound each pool by 64 proposal attempts and initially 2 seconds of screening,
as well as the global remaining search time. Pass the remaining allowance into
projection where possible and check it between repair attempts. An individual
geometry check may overrun a deadline; record the overrun. Select from a partial
nonempty pool. An empty pool records a rejected outer evaluation and follows the
existing local failure response. Never loop indefinitely to fill a pool.

These defaults are pilot settings, not established optima. More CPU work per
iteration may outweigh saved FDTD solves. Pilot timings will determine whether
to reduce pool size before freezing the evaluation configuration.

## Public API and budget accounting

Proposed API; it does not exist at the source revision named above:

```python
from fdtdmesh.optimization import SearchSettings, SurrogateSettings, optimize_mesh

result = optimize_mesh(
    sim,
    reference,
    cells=(120, 120),
    directory="artifacts/surrogate/tip_120/seed_0",
    initial_mesh=seed_mesh,
    strategy="surrogate_local",
    settings=SearchSettings(
        max_evaluations=200,
        max_solves=40,
        max_seconds=600,
        seed=0,
    ),
    surrogate_settings=SurrogateSettings(
        warmup_solves=12,
        pool_size=16,
        max_pool_attempts=64,
        max_pool_seconds=2.0,
        exploration=0.2,
    ),
)
```

The caller supplies a qualified reference and a compatible seed, or the strategy
auto-generates the seed just as `feasible_local` does. `pool_random` accepts the
same pool settings but does not require scikit-learn. Validate incompatible
settings before launching work. `local_settings` continues to control proposal
generation; `surrogate_settings` controls screening and selection.

Keep existing meanings explicit:

- `max_evaluations` counts baseline and outer evaluation records, including failed
  outer selections. Screened-but-unsolved candidates have separate records.
- `max_solves` caps unique selected mesh evaluations, including matching cache
  hits and unsuccessful time convergence, as the existing API specifies.
- Warm-up and baseline evaluations consume these budgets. Exact duplicate
  candidate meshes do not trigger another solve or another GP observation.
- Report actual CUDA launches separately from cache hits and unique evaluations.
- `max_seconds` must cover preparation, screening, GP work, candidate solves and
  archive work inside the search, with explicit setup timing. On resume it is
  cumulative active search time; downtime does not count. Audit the current
  invocation-only deadline and version any compatibility change explicitly.
- Reference generation and tighter winner validation remain separately timed and
  reported; also show their contribution to end-to-end workflow cost.

Initially keep error as the objective. Record timestep, updates, GPU time and
wall time, but defer a learned runtime model or error/cost penalty until the
selection experiment establishes benefit.

## Milestone 3: checkpoints, diagnostics, and server runner

Add `src/fdtdmesh/optimization/surrogate.py` for settings, encoding, fitting and
acquisition. Extend exports in `optimization/__init__.py`. Keep feasibility in
`feasible.py` and actual solver evaluation in `optimize.py`. Refactor common local
loop code only as needed and preserve existing-strategy regression behavior.

Checkpoint with the existing atomic HDF5 writer. Save the eligible ordered
training mesh IDs and targets, feature contract, model hyperparameters, separate
proposal/selection/model RNG states, parent pool/radius, counters, accumulated
times, and dependency/source identities. Record every pool candidate's axes or
replayable representation, validation outcome, prediction, acquisition score,
selection reason and timing. Link solved meshes to their full result archives.

Persist a pending selected mesh before its solve. After interruption, recover its
matching cached result or rerun it, without selecting a different candidate or
adding the observation twice. Store model data and parameters rather than relying
on a pickled estimator. Reconstruct with frozen hyperparameters for the next
selection; refit only at the documented update boundary.

Test exact resume under the same software environment and no binding wall-time
limits. Runtime-based pool truncation and cross-device floating-point differences
can change a new run's trajectory; do not promise bitwise portability. Changing
the feature version, reference, solver/build, settings or dependencies produces a
distinct experiment identity and preserves the previous archive.

Add a headless `benchmarks/compare_surrogate.py` runner with a JSON configuration,
explicit `--run`, deterministic per-case/strategy/seed directories, resumability,
and a report-only mode. CPU preparation should work without `--run`. Add a
`benchmarks/surrogate_study.json` configuration and notebook
`notebooks/06_surrogate_mesh_optimization.ipynb` for inspecting the same archives.
Notebook execution must not be required for the server campaign.

Run one FDTD job at a time per allocated TITAN RTX initially. Limit CPU BLAS thread
counts to the allocated cores and record them. With multiple allocated GPUs,
distribute independent cases/seeds to separate processes and directories; do not
introduce concurrent writes or distributed GP training. Use the site's scheduler
or an existing persistent session rather than assuming a particular scheduler.
Handle termination between solves where possible; retain the last atomic state
when a running process is killed.

After the optional dependency and runner are implemented, the intended commands
are:

```bash
uv sync --locked --group dev --extra surrogate
FDTDMESH_BUILD_CUDA=1 FDTDMESH_CUDA_ARCHS=75 \
  .venv/bin/python setup.py build_ext --inplace --force
.venv/bin/python benchmarks/compare_surrogate.py benchmarks/surrogate_study.json
.venv/bin/python benchmarks/compare_surrogate.py benchmarks/surrogate_study.json --run
```

Retain large HDF5 files under ignored `artifacts/`; commit compact configuration,
summary JSON, figures and an interpretation report after the campaign. Report
preparation failures, reference failures, timeouts and nonconvergence explicitly.

## Milestone 4: verification and scientific comparison

### Implementation verification

Add meaningful CPU tests for encoding real projected axes, fixed-line invariance,
duplicate detection, rejection of mismatched spaces, and exclusion of failed
solves from GP training. Test acquisition against hand-computed cases, finite
predictions with near-duplicate inputs, constant targets, missing dependencies,
and bounded fit failure fallback. Include genuine GP fitting, not only mocks.

Use the existing synthetic-solver fixture for budget and resume tests: interrupt
before/after selection and result writing, then compare selected hashes,
observations, RNG state and winners with an uninterrupted run. Test all-invalid
pools, partial pools, warm-up exhaustion, cached evaluations, zero mobility and
termination limits. CPU tests validate control flow, not EM accuracy.

On TITAN RTX, run the existing numerical/CUDA suite and a short surrogate search
that exercises actual graph stepping, far-field scoring and winner validation.
Check exact budgets, spacing/ratio constraints, fixed coordinates, boundary policy,
and no duplicate launches. Archive the test summary and ensure CUDA checks ran.

### Experimental protocol

Compare three methods: existing `feasible_local`, `pool_random`, and
`surrogate_local`. Use the same geometry, seed mesh, reference, precision,
proposal settings, physics and stopping tolerances. Give each method the same
baseline and warm-up observations where the policies coincide. Reset each GP
for each run; do not share outcomes from competing methods or later seeds.

1. Pilot on tip-and-block at 0 degrees, budgets 120 x 120 and 160 x 160, seed 0,
   with 40 unique candidate evaluations per method. Measure memory and CPU/GPU
   timings before scheduling a larger job. Existing 40-trial reports are context;
   rerun all methods under the new accounting for controlled comparisons.
2. Freeze the GP and pool configuration using development cases. Compare seeds
   0 through 4 at the two tip budgets. Add a smooth cylinder, a slotted/notched
   PEC body, and a rotated engineered case as held-out cases, only after their
   current-method references qualify. Do not tune on these held-out results.
3. For the full solve-budget study, use up to 80 unique candidate evaluations and
   report prefixes at 20, 40 and 80. Do not rerun shorter prefixes unnecessarily.
   Separately enforce equal wall-time budgets selected from pilot timings; a
   run cut short by time is a censored solve-budget result.

Keep the star out of accuracy claims until its reference qualifies. Old strict
reports are not matching hybrid baselines. Reference sensitivity limits how
small a claimed gain can be; repeat an independent finer comparison for borderline
results and retain the tighter temporal winner check. If a reference fails,
report the failure rather than changing the tolerance or substituting its last
mesh as truth.

For each case and across seeds report best measured error versus unique candidate
evaluations, actual GPU launches, and total elapsed search time; time/evaluations
to a predeclared accuracy threshold; final error spread; fit/screen/solve/archive
time; timestep and update counts; reference cost; and winner-validation outcome.
Predeclare case thresholds after reference qualification and before search, with
adequate separation from observed reference sensitivity. Include target misses
when reporting success rates and time-to-target, rather than dropping them.

Use uncached candidate runs for live timing comparisons, randomize method order
across seeds, and avoid simultaneous GPU jobs that contaminate timings. Exact
cache hits are useful for resume and report reconstruction but are not new
speedup measurements. Keep reference construction shared and separately charged.

A practical pilot target is at least 25% fewer candidate evaluations to reach
the same predeclared accuracy, accompanied by a reduction in median search wall
time across the qualified evaluation cases. This is a go/no-go target, not a
promised effect or formal significance test. With five seeds, report variability
and per-case regressions. If selection saves solves but loses time, report that
tradeoff; if it does not reliably help, retain `feasible_local` as the default.

## Delivery order and completion criteria

| Order | Deliverable | Acceptance evidence |
|---|---|---|
| 0 | TITAN/Linux build helper, preflight and environment record | Real CUDA tests run; graph execution and numerical checks pass |
| 1 | Mesh encoding and CPU GP module | Encoding, eligibility, acquisition and numerical-failure tests pass |
| 2 | Surrogate strategy and pool-random control | Bounded selection, original constraints and compatible existing strategy behavior |
| 3 | Checkpoints and headless runner | Interrupted/resumed synthetic and short GPU runs produce consistent state |
| 4 | Notebook and compact report | Reproducible three-way, multi-seed cost/accuracy comparison with all failures included |

Update `docs/mesh_optimization_api.md`, `docs/mesh_optimization.md` and `README.md`
when the implementation exists. Run appropriate tests, lint, and the native suite
for affected code. Keep this plan linked as the design record and mark measured
results separately. Completion requires both working software and an honest
comparison; superiority of the GP is an experimental outcome, not a prerequisite
for documenting a completed implementation.

After the first comparison, consider cost-aware acquisition, geometry/patch
features, or a reduced displacement basis only if measured diagnostics justify
them. Cross-case learning and reference-free error indicators remain separate
research milestones.
