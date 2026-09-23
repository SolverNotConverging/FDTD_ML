# Exact-budget factorial pilot

Campaign `simple_factorial_exact_candidate_pilot_6fb40ec9fe0fe154` tested the
six exact-axis policies on dataset `simple_factorial_69168792914bfe03`. Its five
geometries include a lossless epsilon_r=2 control, lossy epsilon_r=8 and 30 training
cases, an epsilon_r=26 validation case, and an epsilon_r=28 test case. The 52
illumination/budget conditions expand to 312 simulations.

All 312 cases settled on the initial 70 ns attempt. No case needed the available
140 ns or 560 ns retries. All candidates preserved the requested Nx and Ny. Labels
were ranked with

`joint_scattering_loss * (Nt / Nt_uniform)^0.1`.

The pilot produced 49 nonuniform selections among 52 budget labels. Using the
predeclared 1.05x meaningful-improvement threshold:

| Split | Meaningful wins | Budget labels | Median improvement | Maximum improvement |
|---|---:|---:|---:|---:|
| Train | 32 | 36 | 2.257x | 11.170x |
| Validation | 8 | 8 | 4.700x | 6.045x |
| Test | 8 | 8 | 4.804x | 19.770x |

Training wins span three lineages, all five nonuniform policy families, and every
32/48/64/96 budget. Validation and test each have headroom at all four budgets.
The frozen readiness gate returns `ready_for_m5_pilot`, so the same candidate set,
ranking, and duration schedule can be applied to the full simple curriculum.
