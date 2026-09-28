# Geometry-aware meshing

Geometry-aware meshing builds a tensor-product mesh around the final continuous
PEC geometry. It is a CPU construction step; the subsequent solve uses the
native CUDA conformal solver.

```python
from fdtdmesh import Simulation

sim = Simulation(fmin=0.9e9, fmax=1.1e9)
sim.add_circle((0.0, 0.0), 0.17, material="PEC")
mesh = sim.apply_mesh("geometry_aware", max_cells=(512, 512))
result = sim.solve()
```

The automatic domain is derived from final PEC bounds. Domain policy distances
are wavelengths, and exterior cell widths are rounded outward once. Geometry is
never rasterized, rounded, inflated, or moved by this strategy. Air overlays
remain part of the exact recipe.

## `apply_mesh("geometry_aware", ...)`

The strategy accepts either an exact total cell budget or an automatic spacing
target with axis caps:

| Argument | Default | Units | Meaning |
|---|---:|---|---|
| `cells` | `None` | cells `(Nx, Ny)` | Exact total budget, including PML and exterior cells. |
| `target_spacing` | `C0 / fmax / 32` | m | Interior spacing target when `cells` is omitted. |
| `max_cells` | `(512, 512)` | cells | Per-axis cap for automatic construction. |
| `options` | `MeshOptions()` | — | Advanced constraints, anchors, pass and time limits. |

`cells` and `target_spacing` are mutually exclusive. Advanced options are not
top-level mesh arguments; pass `MeshOptions(constraints=..., time_limit=...)`.
The exact budget includes fixed PML collars and required layout coordinates.
Construction validates scanline intersections, spacing constraints, cut-face
donors, and enlarged-cell topology. It retries bounded repairs and reports
unresolved edges when no valid mesh is found.

## Related strategies

`uniform` requests a constant grid and can reject a fixed-layout request.
`quasi_uniform` projects the same preference through the mesh constraints.
`density` accepts positive axis density vectors at an exact budget. `custom`
accepts only `mesh=existing_mesh`. These strategies all preserve exact geometry.

Every generated mesh obeys mandatory adjacent-cell grading (`max_ratio <= 1.4`)
and the requested PML/exterior layout. A failed `apply_mesh` leaves any previous
mesh unchanged. `MeshInfeasibleError` means the requested constraints could not
be satisfied; `MeshOptimizationError` means construction did not finish within
its resource limits. Neither proves that another budget or anchor assignment
could not work.

## Reports and accuracy

Successful construction stores target spacing, caps, pass history, topology and
donor counts, and a final validity status in `mesh.metadata["geometry_aware"]`.
Mesh validity is a geometric/operator check, separate from DFT convergence,
boundary sensitivity, and scattering accuracy. Hybrid boundary accuracy remains
under targeted validation; no geometry-aware construction result qualifies it.
