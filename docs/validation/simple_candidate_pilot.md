# Simple candidate-label pipeline pilot

Campaign `simple_candidate_pilot_e38c11dcf8b35fc8` exercises the complete label
path on dataset `simple_dk_3ea40e8117434e5c`. It selects four geometries spanning
two training lineages, validation, and test; expands 30 geometry/illumination/budget
conditions; and evaluates nine uniform, region, interface, hybrid, and randomized
mesh policies. The 270 cases use a common 70 ns duration and analytic complex
dielectric-cylinder targets.

Each case stores its exact x/y axes, native complex far field, derived scattering
width, both loss components, actual `Nx*Ny*Nt`, material/feature metadata, split,
lineage, geometry, and illumination IDs. Campaign membership is deliberately absent
from the numerical fingerprint so a later superset campaign can reuse identical
cases. Atomic records are validated against the dataset condition and numerical
source version before reuse.

The pilot accepts 266 cases and retains four unsettled randomized candidates as
rejections. All uniform baselines settle, so all ten illumination groups receive
labels. Pareto ranking is performed across 32/48/64 budgets within each geometry and
illumination, and each uniform update count defines a compute cap.

Two nonuniform budget wins occur, both on the held-out epsilon_r=5 test geometry:

- incidence set 0: a lower-budget strong-region mesh improves the 64² uniform-cap
  loss by **2.003x**;
- incidence set 1: the corresponding improvement is **1.648x**.

The selected train and validation subset has no positive budget win. This is an
important pilot finding rather than a reason to relabel the test data: the full
campaign must include all training lineages, especially epsilon_r=4.5, before model
training. `simple_candidate_full_d535ccac015b3cd8` covers all 32 geometries, 352
conditions, and 3,168 cases while preserving the frozen grouped splits.

Run or resume the pilot, then summarize labels, with:

```bash
MPLCONFIGDIR=/tmp/scattermesh-mpl \
  .venv/bin/python scripts/run_simple_candidate_campaign.py \
  --device cuda:0 --shard 0 --shards 4
MPLCONFIGDIR=/tmp/scattermesh-mpl \
  .venv/bin/python scripts/run_simple_candidate_campaign.py --summarize
```

Use `--campaign configs/simple_candidate_full.json` with the same output root to
grow the verified cache into the full campaign. Local records and labels are under
`runs/simple_candidate_pilot/`.
