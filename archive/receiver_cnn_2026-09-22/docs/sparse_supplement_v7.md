# Sparse reference supplement and combined training corpus

The v6 generator always constructs 8–16 bodies and often distributes them across
the physical interior. Generator v7 adds isolated and localized configurations;
v6 scene definitions and their 2,000 accepted references are retained.

The requested supplement is **1,000 converged scenes**, not 1,000 attempts:

| Family | Train | Validation | IID test |
|---|---:|---:|---:|
| One dielectric | 200 | 25 | 25 |
| Two dielectrics with a gap | 200 | 25 | 25 |
| One rectangular PEC | 200 | 25 | 25 |
| Rectangular PEC and one dielectric | 200 | 25 | 25 |
| Total | 800 | 100 | 100 |

Each dielectric may be a circle, rectangle, triangle or convex polygon. Pairs
include both matching and different primitive types. The three requested object
span bands are 8–12.5%, 12.5–22%, and 22–30% of the domain axis. Lattice rounding
and the physical aspect ratio of circles can increase individual spans. The
minimum span is at least 10.24 input pixels, or 81.92 cells at resolution 1024.
Whole clusters receive random translations and reflections; pair axes vary.
Physical domain size and electrical size vary independently as in v6.

Pair bounding boxes have 4, 6 or 8 input pixels of vacuum separation (32, 48 or
64 reference cells at 1024); curved/polygonal surfaces may be farther apart.
This deliberately avoids pretending that a gap below the input-raster resolution
can be learned by the current CNN. Permittivity varies from 1.05 to 30; electric
loss varies independently, including lossless scenes. Permeability stays one and
magnetic conductivity zero. Three receivers and one current source vary in vacuum
with clearance from the objects, and all probe coordinates are anchored.
PEC rectangle coordinates occupy the 64-line lattice and receive four exact
anchors, preserving faces through all reference refinements. PML stays in vacuum.

The sparse corpus has independent seeded v7 identities and preserves train,
validation and test lineage. Workers reserve geometry under a shared lock before
running the solver so near duplicates in different splits are skipped. Duplicate
checking requires both at most 0.5% domain mismatch and at least 90% foreground
intersection-over-union: shared empty space alone is insufficient. Finalization
rechecks splits; the combined dataset checks across both input corpora as well.

## References and later training

The existing reference policy is retained: sampled material averaging with
8–32 samples, 1e-3 quadrature tolerance, minimum accepted grid 1024, maximum 2048,
2% receiver convergence and 1% tail threshold, with two successive spatial passes.
Unsettled tails can extend duration up to the existing six-extension limit;
pure spatial failures at the ceiling are skipped. Per-scene time/resource limits
and rejected records remain explicit. There are no 4096 attempts.

```bash
.venv/bin/python scripts/start_sparse_reference_campaign.py
```

The detached workflow writes `artifacts/reference_sparse_v7_1000/`, then audits
and combines both completed campaigns into `artifacts/reference_combined_v7_3000/`.
The combination retains original scene hashes and reference paths and publishes
a new dataset ID. Its split totals are 2,400 train, 300 validation, and 300 IID
test. No old reference waveform is regenerated. Check `workflow_stage.json`,
`progress.json`, lane logs and `workflow_failure.json` for status.

Read current progress with system Python (no virtual environment required):

```bash
python3 scripts/campaign_status.py
python3 scripts/campaign_status.py --watch 10
python3 scripts/campaign_status.py --json
```

The default is the sparse v7 campaign; `--campaign PATH` selects another reference
campaign. Reports include accepted/remaining counts, split totals, recent aggregate
throughput, approximate remaining reference time, per-case timings, accepted grid
and Nt statistics, and the latest recorded case on each GPU. Counts come directly
from committed decisions, so they can be newer than `progress.json`. The estimate
uses up to 40 recent decisions and includes failures and time since the last
completion; difficult future cases can change it. GPU rows show persisted state,
not a process-liveness check. `--watch` refreshes until Ctrl+C.

### Probe/PEC roundoff repair

The first run stopped with 411 converged references after three scenes expressed
a probe coordinate and a PEC face with adjacent floating-point values. The
reflection formula `L - k*L/64` and direct probe formula `j*L/128` differed by one
ULP, causing the uniform mesher to see two anchors for one line.

`SceneSpec.build` now uses the existing face coordinate for probe coincidences
within four ULPs, and anchors the built probe exactly there. PEC faces are retained
exactly; distinct geometry anchors and probe separations beyond roundoff remain
distinct. This applies consistently to reference and learned/heuristic meshes.
Stored scene records, scene hashes and existing accepted waveforms are unchanged.

The repair helper `scripts/repair_sparse_anchor_campaign.py` requires both workflow
and campaign locks, verifies this specific failure and successful remeshing at all
reference levels, archives the failed attempts, and updates source provenance.
On 2026-09-20 it archived the three failed references and verified 824 committed
decision/reference JSON files were unchanged. Regression checks passed: 30 tests,
with one CUDA test skipped in the sandbox. Restarted workflow PID: 2102406.
The status script suppresses ETA on failure and measures recent throughput only
from completions since the latest launch, excluding downtime.

The workflow stops after publishing the combined corpus. For later training:

```bash
.venv/bin/python scripts/start_training_v6.py \
  --campaign artifacts/reference_combined_v7_3000 \
  --output artifacts/training_combined_v7_3000 --workers 12 --gpu 0
```

This uses fresh target/model output, width 16 and budgets 48/64/96/128. All accepted
training and validation scene-budget pairs are eligible for teacher preparation;
IID test scenes remain excluded. Feasibility failures remain recorded, so report
actual family/size coverage after preparation rather than assuming every budget
survives. Sparse training pairs receive **2× sampling weight**; dense pairs receive
1×. For the full 800-sparse/1,600-dense training split, this gives approximately
50% sparse exposure, subject to teacher feasibility. The weighted sampler draws
with replacement and preserves the epoch sample count. The default weight is 2
in the training launcher and is configurable with `--sparse-sample-weight`.
Validation/test distributions are unchanged. Sampling probabilities by family are
recorded in the training report and model metadata. Probe anchors remain required.

The physical-search launcher discovers all families in the new manifest
and selects 16 training plus four validation scenes per family with feasible
targets at all budgets. With eight families this is 128 training and 32 validation
scenes (640 total scene-budget targets), including all four sparse families.
Subsequent distillation also uses 2× sparse sampling weight. Since that search
subset has equal sparse/dense scene counts, sparse pairs receive about two thirds
of distillation training exposure.

Before subsequent distillation, address the prior v6 distillation failure: a
30-second mesh-projection timeout stopped its repair-loss step. Its completed
physics-search targets remain available. This supplement does not establish a
physical-accuracy improvement by itself. Evaluate sparse and dense family errors,
size/gap strata and receiver waveform/spectrum errors separately after training.

## Geometry inspection

```bash
.venv/bin/python scripts/preview_sparse_scenes.py
```

This produces a 12-scene gallery and reproducible preview manifest under
`artifacts/sparse_v7_preview/`. These are geometry previews, not convergence claims.
