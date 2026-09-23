# Intermediate-resolution candidate search

The first full candidate campaign sampled both uniform and nonuniform grids only at
32, 48, 64, and 96 cells per axis. A focused grid at the same axis count frequently
reduced scattering error, but its smaller cells increased `Nt`. It therefore
exceeded the matched `Nx*Ny*Nt` cap, while the next lower sampled grid discarded too
much spatial resolution. This resolution gap caused uniform to be selected even
when nonuniform placement was useful.

Campaign `simple_candidate_scaled_train_b7d0de8eb710babc` decouples the uniform
target budget from the candidate axis count. It evaluates 15 frozen combinations of
policy and cell factor on six geometries from three training lineages. All 1,152
records completed: 967 settled and 185 aggressive 32-cell candidates were retained
as unsettled failures. The campaign produced 59 meaningful wins in 72 labels,
covering all three lineages and every budget. The median selected improvement is
1.481x and the maximum is 2.912x.

Four candidates appear in selected labels:

| Candidate | Selected training labels | Meaningful affordable comparisons |
|---|---:|---:|
| `region_wide_f92` | 39 | 57 / 71 |
| `region_medium_f84` | 14 | 37 / 54 |
| `region_medium_f87` | 5 | 11 / 45 |
| `region_strong_f78` | 1 | 17 / 54 |

Those four policies were frozen before evaluating campaign
`simple_candidate_scaled_holdout_6eef28266cf0126f`. Its 320 cases cover all eight
geometries in the untouched validation and test lineages. It completed with 272
settled and 48 unsettled records. Validation changed from zero wins in the original
coarse-resolution search to 19/32 wins, with a 1.303x median and 1.927x maximum.
Test has 28/32 wins, with a 1.741x median and 2.343x maximum.

The combined 1,472-case audit passes every frozen strict-compute headroom check.
Training wins span three lineages, four candidate names, and all four budgets; both
held-out splits have substantial headroom. This proves that resolution sampling,
rather than an absence of useful nonuniform meshes, caused the original failure.
It does not define the final CNN target: the product contract requires exactly the
requested `Nx` and `Ny`, so scaled candidates remain diagnostic evidence.

The saved equal-resolution results also quantify a softer compute objective. If
candidate ranking uses `loss * (updates/uniform_updates)^lambda`, the number of
meaningful wins over the original 352 labels is:

| lambda | Train | Validation | Test |
|---:|---:|---:|---:|
| 0.0 | 265/288 | 32/32 | 32/32 |
| 0.1 | 259/288 | 32/32 | 32/32 |
| 0.25 | 249/288 | 32/32 | 32/32 |
| 0.5 | 229/288 | 29/32 | 31/32 |
| 1.0 | 183/288 | 25/32 | 31/32 |

The CFL time step itself remains at the stable grid limit. Cost softening belongs in
candidate ranking; raw physics loss, `Nt`, cell updates, and wall time remain stored
separately so no rerun is needed when selecting the cost exponent.

The adopted final ranking fixes every candidate at the requested `Nx` and `Ny` and
uses `lambda=0.1` initially. The first decorrelated-pool pilot compares exact-axis
uniform, interface-wide, region-wide, region-medium, region-strong, and hybrid-wide
policies. Strict matched-update labels continue as a secondary diagnostic.
