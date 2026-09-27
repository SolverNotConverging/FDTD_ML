# Geometry-first simulation and meshing

```python
from fdtdmesh import Simulation, DomainPolicy

sim = Simulation(fmin=0.9e9, fmax=1.1e9)  # No predefined domain
sim.add_circle((0, 0), 0.17, material="PEC")
sim.add_circle((0.0756, -0.009), 0.15, material="air")
mesh = sim.apply_mesh("geometry_aware", max_cells=(512, 512))
result = sim.solve()  # Native CUDA
result.save("result.h5")
```

[Notebook 04](../notebooks/04_geometry_aware_meshing.ipynb) shows the exact recipe,
large mesh/discretization figures, construction report, GPU solve, and refinement
comparisons. No scattering optimization or reference field is used to build a mesh.

## Automatic domain

Omitting `Simulation.size` enables automatic domain allocation. Frequency arguments
must be supplied by name in this form. Geometry can use negative coordinates and
has no construction canvas. `Geometry()` also creates a standalone unbounded recipe.
Ordered PEC/air overlays and touching-material rules are unchanged. Bounds come from
the final continuous PEC material, not a raster or the union of primitive boxes.

The domain is resolved when needed for `size`, `layout`, plotting, saving, or mesh
preparation. Empty final PEC geometry cannot define a domain. Editing geometry
invalidates both the grid and the resolved domain; a subsequent request recomputes
the domain. Interior mesh changes alone never change it.

`domain=DomainPolicy(...)` configures allocation. It cannot be combined with explicit
`size`, `pml`, or `layout`; the existing explicit-domain API remains supported.

| Argument | Default | Meaning |
|---|---|---|
| `exterior_spacing` | `None` | Fixed exterior cell width in metres; `None` resolves to `C0 / fmax / 24`. |
| `pml_cells` | `12` | PML cells per side; positive integer. Physical thickness is count × exterior spacing. |
| `pml_to_contour` | `6` | Cells between the inner PML boundary and integration contour; minimum 2. |
| `contour_to_tfsf` | `4` | Cells between contour and TFSF; minimum 2. |
| `scatterer_to_tfsf` | `5` | Cells from final PEC bounding box to TFSF; minimum 3. |
| `phase_origin` | `None` | Input-coordinate phase origin in metres. `None` uses the final PEC bounding-box centre. An explicit origin must lie inside the derived TFSF box. |

`sim.geometry` preserves the original input recipe. `sim.computational_geometry`
is its exact translated copy in the zero-based solver domain. The mapping is
`solver_coordinate = input_coordinate + sim.coordinate_offset`. Plot coordinates,
mesh coordinates, and `result.geometry` use the solver frame. `result.input_geometry`
recovers the original recipe. HDF5 saves the policy, recipe and transform; loading a
simulation restores automatic allocation and editable original coordinates. The
phase origin translates with the geometry, preserving its relative definition.

All reserved axis coordinates are fixed across interior refinements. Tensor-product
lines continue tangentially through the exterior, so the complete 2D exterior grid
is not identical when interior coordinates differ.

## `apply_mesh("geometry_aware", ...)`

This strategy chooses the cell counts. It also works with an explicit-domain
simulation after `fit_domain()`. Standard `apply_mesh` strategies still require
their existing `cells=(Nx, Ny)` argument.

| Argument | Default | Meaning |
|---|---|---|
| `target_spacing` | `C0 / fmax / 32` | Positive interior spacing target in metres. Not a fixed budget or a strict per-cell upper bound; boundary transitions and geometry repairs can change local spacing. |
| `constraints` | `None` | `AxisConstraints`. Default minimum 0, maximum `max(target_spacing, exterior_spacing)`, adjacent ratio 1.4. A maximum smaller than fixed exterior spacing is rejected. |
| `max_cells` | `(512, 512)` | Maximum total cells on each axis, including exterior/PML. |
| `max_passes` | `20` | Maximum mesh construction/repair passes, including failed projections. |
| `time_limit` | `30.0` | Positive construction time allowance in seconds, checked between passes and allocated to axis projections. Geometry/operator validation can overrun it. |

`cells`, `density`, `checkpoint`, `strict=True`, external `anchors`, and a nondefault
`anchor_assignment` are incompatible with this strategy. Invalid combinations raise
`ValueError`; there is no silent fallback to another strategy.

Construction probes primitive interiors, including air cutters, to expose enclosed
features. It scans exact vacuum/PEC intervals on both axis families and proposes
interior witness nodes for missed segments or multiple crossings. For small cut
faces with unavailable donors, it proposes three interior nodes in the associated
air interval to create separate full donor edges. The global axis projector enforces
fixed exterior lines, spacing and grading, using local anchor assignments. Infeasible
repair proposals are rolled back before retrying with more interior cells.

Every new grid is rescanned in both directions; changing one axis can introduce
violations on the other. Returned meshes pass the exact edge scan and the actual
conformal enlarged-cell operator. Geometry is never moved relative to itself,
rounded, inflated, or rasterized for these tests. Enlargement remains mandatory
where required by the solver.

The construction is a bounded heuristic, not a proof of existence or a minimum-cell
algorithm. Difficult cusps, tiny gaps, fixed exterior transitions and local anchor
assignment may prevent success. In particular, a tested sun at 30 degrees still
exhausts the default repair limit. Successful geometric validation also does not
establish scattering accuracy; resolution and boundary convergence remain separate.

## Reports and failures

Successful construction returns the usual `Mesh`; inspect
`mesh.metadata["geometry_aware"]` for the target, caps, pass history, topology/donor
violation counts, elapsed time and final `status="valid"`. This report is independent
of the GPU convergence status.

`GeometryMeshingError` (from `fdtdmesh.geometry_mesher`) is a `MeshInfeasibleError`
subclass. Its `.report` contains attempted cell counts, projection messages and
unresolved edge coordinates (`axis`, `fixed`, `edge`), plus donor details where
applicable. Exhausting resources means no valid mesh was found by this construction;
it is not proof that the geometry has no possible mesh. A failed `apply_mesh` keeps
the previously applied grid intact. User-input and non-geometric runtime errors
continue to propagate normally.
