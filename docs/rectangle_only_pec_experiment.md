# Rectangle-only PEC diagnostic

This is an ablation of train-v5-000000. Its PEC circle and seven-sided PEC polygon were removed; its one axis-aligned PEC rectangle, five dielectric objects, domain, source, receivers, pulse, losses and physical simulation duration are unchanged. It is a diagnostic variant with a separate scene identity, not a replacement for the original pilot scene.

Both mesh families use sampled dielectric averaging (adaptive 8–32), float64 fields, identical physical PML thickness with refinement-scaled collar counts, and 128/256/512/1024 cells per axis. The anchored family retains the existing source/receiver anchors plus the rectangle’s four face coordinates. Its 128-grid base was reused from the preceding face-anchor experiment because all relevant mandatory coordinates are unchanged; subsequent grids bisect each interval.

| Refinement | Uniform waveform | Anchored waveform | Uniform spectrum | Anchored spectrum |
|---|---:|---:|---:|---:|
| 128→256 | 2.89491% | 0.18443% | 3.36155% | 0.22577% |
| 256→512 | 1.63669% | 0.07012% | 1.98206% | 0.08316% |
| 512→1024 | 0.28033% | 0.02835% | 0.32974% | 0.03422% |

Metrics are worst-receiver relative L2 differences between consecutive meshes, not analytical absolute errors.

uniform: 2 consecutive spatial passes; final decay check True; required resolution 1024; meets the configured reference criteria at the final grid: **True**.

anchored: 3 consecutive spatial passes; final decay check True; required resolution 1024; meets the configured reference criteria at the final grid: **True**.

The anchored grid uses 115,059 timesteps at 1024 versus 90,797 for the uniform grid (26.7% more). These diagnostics shared GPU 3 with the original pilot; their wall times are not uncontended speed benchmarks.

The comparison supports axis-aligned rectangular PEC as a useful first proof-of-concept restriction. Retain dielectric shape diversity and material averaging. To keep all exposed PEC interfaces orthogonal, PEC must take precedence over overlapping dielectric objects; a later circle or slanted polygon cutting into a PEC rectangle can reintroduce curved/slanted PEC boundaries. Overlapping axis-aligned PEC rectangles still form orthogonal boundaries.

This is one diagnostic scene. No new bulk corpus was generated and no production generator restriction was applied. The original mixed-PEC pilot remains separate. A broader rectangle-only pilot is needed to establish corpus acceptance rate.

Evidence: artifacts/rectangle_only_pec/{manifest.json,uniform,anchored,summary.json,comparison.png}. Run with scripts/compare_sampled_pilot.py --manifest and scripts/check_pec_anchors.py --manifest --base-mesh.
