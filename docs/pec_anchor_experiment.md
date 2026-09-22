# PEC-face anchor experiment

`Scene2D.add_pec_anchors()` adds optional mandatory mesh coordinates after scene
construction. The default `mode="axis_aligned"` anchors the fixed coordinate and
endpoints of horizontal/vertical PEC polygon edges. Thin PEC lines retain their
existing exact anchors. Later ordinary primitives whose bounds overlap earlier
PEC bodies are included conservatively, since they can expose PEC cutout faces.

`mode="features"` additionally anchors polygon vertex coordinates and circle
centers/extrema. These coordinates do not make a Cartesian grid conform to a
curved or slanted interface. All supplied coordinates remain exact; the method
never silently merges nearby anchors or drops them to satisfy a budget.

```python
# Add all materials, bodies, sources and receivers first.
report = simulation.add_pec_anchors(mode="axis_aligned")
simulation.mesh_from_density([1], [1])
```

The caller must remesh after adding anchors. An existing mesh missing any new
anchor is rejected by the usual validation. This experiment does not change the
running pilot or automatically convert the uniform reference campaign into a
nonuniform one. Such a policy change needs an explicit, separately identified
reference configuration.

## Controlled check

A 20×15 mm rectangular domain contains a PEC wall at x=0.371 Lx. Its vacuum cavity
has a known analytical fundamental frequency. We compare uniform point-mapped
PEC against a mesh anchored exactly at the wall, using identical cell budgets
and the production Yee coefficients, including temporal dispersion.

At 512×512, relative frequency error changes from **0.020781%** to **0.00046980%**
(44.2× lower). At 128×128 it changes from 0.863532% to 0.00751745%. Material
averaging is enabled in both cases; all non-PEC material here is vacuum.

## Pilot scene 000000

The scene has a rectangular PEC body, a PEC circle, and a PEC polygon. The face
policy adds two x and two y coordinates for the rectangle. Existing source,
receiver, boundary and PML anchors remain mandatory. The 128×128 base mesh uses
uniform density preference, exact cell budgets and maximum adjacent spacing
ratio 1.4; construction took 13.1 seconds. We then bisect all intervals for nested
256, 512 and 1024 grids. Only floating-point roundoff in PML lines is reset to the
canonical fixed-collar representation. Physical PML thickness and scaled cell
counts match the uniform baseline; interior anchors stay fixed exactly.

Both sequences use the same sampled dielectric averaging (8–32 samples), float64
fields, excitation, receiver positions and 9.43982 ns duration. The baseline is
the preceding material-averaging run, not the older point-material run.

| Refinement | Uniform waveform | Anchored waveform | Uniform spectrum | Anchored spectrum |
|---|---:|---:|---:|---:|
| 128→256 | 18.0248% | 8.4443% | 17.4407% | 7.9578% |
| 256→512 | 7.6777% | 2.7272% | 8.0358% | 3.3716% |
| 512→1024 | 3.3948% | 2.5423% | 3.5294% | 2.4947% |

These are maximum receiver L2 differences, not errors against analytical truth.
At the finest comparison, waveform and spectrum differences improve by about
25% and 29%, respectively, but **neither passes the 2% spatial threshold**.
Decay checks pass. No converged anchored reference is claimed.

The 1024 grid needs **115,059 timesteps versus 90,797** for the uniform grid,
26.7% more, because of its smaller minimum cell spacing. Runtime comparisons
would also include meshing/setup and GPU contention; no uncontended wall-time
speedup is claimed. Changing the grid redistributes resolution at other
interfaces too, so the mixed-scene comparison does not isolate rectangle error
as cleanly as the analytical test.

The fuller feature policy creates 23 anchors per axis, including one ordinary
primitive that overlaps earlier PEC. The 128-grid mesher reached its 30-second
optimization limit without producing a mesh. This is **not proven infeasibility**
and no feature-policy electromagnetic result is reported. Arbitrary vertex
anchors should not be made a blanket default on the strength of this experiment.

## Evidence

`python -m pytest -q tests/test_pec_anchors.py tests/test_scene.py`: **14 passed**.
The four anchored GPU refinements completed with finite observations and exact
anchor/mesh/PML validation. Existing production solver defaults are unchanged.

Reproduction: `scripts/check_pec_anchors.py`; saved base grids, anchor reports,
configurations, waveforms, metrics and figures: `artifacts/pec_anchor_check/`.
GPU-3 diagnostic contention is recorded in the original pilot's timing metadata.

## Rectangle-only follow-up

A subsequent diagnostic removed the PEC circle and polygon while retaining the
rectangle and five dielectric objects. Both uniform and anchored sequences meet
the reference convergence criteria at 1024. The anchored sequence reaches
0.02835% waveform and 0.03422% spectrum differences, versus 0.28033%/0.32974% on
the uniform sequence. See [rectangle-only PEC diagnostic](rectangle_only_pec_experiment.md).
This supports a broader axis-aligned rectangle-only PEC pilot; it does not
constitute a generator change or a full-corpus acceptance measurement.
