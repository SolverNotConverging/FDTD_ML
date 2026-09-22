# Agent guidance

Use `gpt-5.6-luna` for bounded, repetitive work that does not require strong
reasoning. Good examples include mechanical file inspection, formatting, routine
test execution, collecting metrics from already-defined outputs, and applying a
well-specified repetitive edit.

Keep architectural decisions, numerical-method changes, ambiguous debugging,
scientific interpretation, convergence analysis, and final integration with the
primary agent or a stronger model. A delegated task must name its exact scope,
inputs, expected output, and files it may edit. The primary agent reviews and
integrates all delegated changes and remains responsible for tests and conclusions.

Do not delegate work merely to create concurrency. Delegate only independent work
whose result can be checked objectively, and never let multiple agents edit the
same file concurrently.
