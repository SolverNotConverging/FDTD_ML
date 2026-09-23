# Agent guidance

Delegate bounded, repetitive work that does not require scientific or architectural
judgment to `gpt-6-luna` with `reasoning_effort="high"`. Use a Luna sub-agent when
the task is independently checkable and substantial enough that delegation saves
primary-agent attention.
Good examples include mechanical file inspection, formatting, routine test
execution, collecting metrics from already-defined outputs, verifying manifests,
and applying a well-specified repetitive edit. Trivial one-command checks may stay
with the primary agent when delegation would add more overhead than the work.

Keep architectural decisions, numerical-method changes, ambiguous debugging,
scientific interpretation, convergence analysis, and final integration with the
primary agent or a stronger model. A delegated task must name its exact scope,
inputs, expected output, files it may edit, and objective completion criteria.
Prefer read-only delegation for audits and metric collection. The primary agent
reviews and integrates all delegated changes and remains responsible for tests and
conclusions.

Do not delegate work merely to create concurrency. Delegate only independent work
whose result can be checked objectively, and never let multiple agents edit the
same file concurrently.
