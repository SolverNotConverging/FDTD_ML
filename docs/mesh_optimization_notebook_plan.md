# Notebook 03: numerical mesh optimization against converged references

Proposed implementation plan. No optimization runs or reference solutions have
been produced by this planning step. The fixed-exterior redesign is deferred.

## Deliverable

`notebooks/03_reference_and_optimized_mesh.ipynb` will run a selected shape and
incidence orientation end to end, then offer a resumable catalog sweep. Each
qualified case produces its exact geometry, fine-grid reference, uniform and
deterministic baselines, best feasible mesh found at a fixed cell budget, error
metrics, polar scattering plots, optimization history and HDF5 archives.

The notebook should orchestrate reusable modules rather than contain the solver,
geometry library and optimizer implementation itself. Proposed helpers:
`benchmarks/shapes.py`, `benchmarks/reference.py`, `benchmarks/optimize.py`, and
an accompanying resumable command-line sweep for later server runs.

## Geometry catalog

| Family | Initial shapes | Continuous representation |
|---|---|---|
| Smooth/simple | Circle, ellipse, rectangle | Analytic circle/ellipse, polygon |
| Convex | Triangle, regular pentagon, irregular convex polygon | Polygon |
| Concave | L, U, star | Polygon or ordered elementary-shape overlays |
| Symbols | Wi-Fi, crescent moon, sun | Circular bands/dot, circle subtraction, disk and polygon rays |

Add analytic ellipse support and rigid transforms before presenting ellipses as
exact geometry. Rotated ellipses need analytic membership and axis-line quadratic
intersections. Polygons and circles already have continuous representations.
Rotate about the common phase origin; preserve physical scale, shape parameters,
material order and feature widths across every mesh in a comparison.

Use procedural symbols first: no external image assets are needed. Keep all
components and gaps in a safe bounding circle inside TFSF for every orientation.
Specify each shape's physical feature dimensions, not merely its raster size.
Do not close narrow gaps or smooth tips to rescue an optimizer candidate.

Optional later image import should trace contours and holes once, simplify using
a stated physical tolerance, and freeze the resulting polygon recipe for all
meshes. A traced bitmap is an approximation of the image; the resulting continuous
polygon is the authoritative simulation geometry. Never retrace at solver-grid
resolution or silently treat pixels as conformal boundaries.

## Incidence orientations

Initial sweep: 0, 30, 60 and 90 degrees per shape. Expand asymmetric shapes to a
full azimuth sweep on the server. Separate optimizations at each orientation are
the initial objective; a shared mesh minimizing a multi-incidence loss is a later
experiment.

The current source propagates along +x. For physical incidence angle alpha,
rotate the object by -alpha about the phase origin and use solver observation
angles theta-alpha. Report theta in the original object frame. This is a rotated
object/domain experiment, not arbitrary-angle TFSF on a fixed lab-frame grid.
References and candidates must use exactly the same orientation and origin.
Proven symmetry-equivalent cases may share cached results; do not assume symmetry
for irregular shapes. A circle is a useful rotation-invariance check.

## Reference qualification

Temporal DFT convergence is necessary but does not establish spatial convergence.
Use at least three successively finer conformal enlarged-cell meshes, for example
48, 72 and 108 nominal cells per centre wavelength, increasing further when needed.
These are starting resolutions, not accuracy guarantees. Preserve the physical
domain, geometry, TFSF, contour, source band and PML thickness; increase PML cell
count when refining a truly uniform reference grid.

Require both adjacent refinement comparisons to meet a configurable complex
far-field tolerance, initially 0.2%, across the band. Also inspect per-frequency
errors and scattering-width errors, and check PML thickness/contour sensitivity
and temporal-tolerance sensitivity on the finest candidate reference. A small
successive-grid difference is evidence, not a rigorous error bound. Tighten the
reference if optimized candidates approach its observed uncertainty.

Use the cylinder series to audit the reference procedure on the circle. For other
shapes label accepted data a "converged numerical reference", not exact ground
truth. If resolution, donor topology, convergence, memory or runtime caps prevent
qualification, retain diagnostics and mark the case unqualified. Do not optimize
against it as though it were certified truth.

## Objective and controls

For normalized complex far field S, minimize the square root of the mean over
frequencies of `sum_theta |S_candidate-S_reference|^2 / sum_theta |S_reference|^2`.
Use the same periodic angular samples and all 21 DFT bins over 0.9-1.1 GHz to
start. Define an explicit absolute normalization floor for nearly zero-scattering
references. Avoid pointwise relative errors and dB-based losses at pattern nulls.
Do not fit away amplitude scaling or phase: they are part of the error.

Report complex error, scattering-width error, worst frequency, minimum spacing,
timestep, update count, GPU execution time, preparation time and total cell count.
This is currently an end-to-end discretization comparison: outside-PML coordinates,
contour quadrature and incident-line discretization may change with the mesh.
The postponed fixed-exterior work is not assumed to exist.

For a stricter spatial comparison, add an explicit timestep setting and a common
safe timestep derived from imposed minimum spacing and checked against every
candidate's conformal stability bound. Do not let each optimizer trial silently
choose a different timestep in that mode. Keep the same time-based checkpoint
spacing when timesteps differ; the current API would need an extension for this.
Alternatively label the initial native-CFL mode an accuracy-versus-cost study
and report the varying timestep explicitly.

## Optimizer

Use 6 smooth log-density controls per axis initially, with one additive degree
of freedom removed per axis because density scale is irrelevant. Interpolate,
exponentiate, then pass both positive densities through the existing exact-budget
projector. Preserve anchors, PML collars, minimum/maximum spacing and ratio <=1.4.

Start from feasible uniform and deterministic baselines. Use bounded differential
evolution with a deliberately small explicit population and `polish=False`.
Optional bounded Powell refinement may spend a separately allocated evaluation
budget. Gradient descent is not the initial method: integer anchor allocation,
PEC node membership and enlarged-pair changes make the implemented objective
nonsmooth and potentially discontinuous.

Reject known geometric/mesh infeasibility before launching CUDA. Treat failed
temporal convergence as an invalid trial, never as a good low-scattering result.
Record failure categories; propagate unexpected software/CUDA failures instead
of disguising bugs as optimization penalties. Retain the best feasible evaluated
mesh, including baseline candidates, so a failed search cannot replace a better
baseline. Call it "best found", not a proven global optimum.

PyTorch is optional, not a requirement for the first optimizer. Wrapping mesh
controls as torch tensors does not create derivatives through SciPy projection
and native CUDA stepping. A valid end-to-end gradient would require a derived
backward/adjoint implementation and treatment of topology/assignment changes.
Do not claim that autograd differentiates the current solver.

If measured evaluation costs justify it, a later PyTorch/BoTorch Gaussian-process
surrogate can propose new black-box trials by optimizing an acquisition function.
Those gradients belong to the surrogate, not FDTD; real solver evaluations still
determine feasibility and accuracy. This is an optimizer surrogate, not a CNN
mesher. Benchmark it against the same derivative-free budget before adopting it.

Cache by exact geometry/orientation, full solver configuration and actual mesh
coordinates, not only density parameters: different proposals may project to the
same grid. Save every trial and RNG/optimizer state for true resumability, or
clearly distinguish a warm restart from continuation if solver state cannot be
serialized. Use one GPU evaluation at a time on the laptop.

## Notebook flow and figures

1. Configure shape, incidence, physical scale, coarse budget and run profile.
2. Display the exact geometry and all requested orientations; check representability.
3. Generate/load reference data and display spatial/temporal qualification evidence.
4. Run uniform and deterministic baselines with the same coarse budget.
5. Optimize; show best-so-far error, feasible/invalid trial counts and elapsed time.
6. Re-evaluate the winner with stricter stopping and against the finest reference.
7. Plot exact geometry with each mesh, zoomed cut faces/enlarged pairs, and axis
   spacing. Plot all angular far-field/width comparisons on polar axes.
8. Produce a shape-by-incidence gallery of best-found meshes plus a summary table
   of baseline error, optimized error, improvement, reference status and GPU cost.
9. Save HDF5 case/trial results and PNG figures; allow plotting saved sweeps without GPU.

## Laptop and server execution

Default notebook execution selects one shape and one orientation, not the entire
catalog. Start with 50-100 expensive evaluations, an independent wall-time cap,
and a short pilot to measure preparation and GPU time. Mesh MILP preparation can
cost more than the field solve, so avoid claiming runtimes from GPU milliseconds
alone. The full catalog and all orientations are opt-in, cached, resumable sweeps.
Server profiles can increase evaluation budgets and use multiple independent
seeds. No absolute-optimum guarantee accompanies any fixed evaluation budget.

## Implementation order

1. Geometry catalog, analytic ellipse and orientation/frame tests.
2. Reference runner, qualification metrics, caching and HDF5 records.
3. Density parameterization and derivative-free search, with failure accounting.
4. Third notebook, polar comparisons, optimized-mesh gallery and server CLI.

Algorithm documentation: [SciPy differential evolution](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html)
and [SciPy optimization guide](https://docs.scipy.org/doc/scipy/tutorial/optimize.html).
