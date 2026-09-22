# Dataset generator v4: low-budget strata

Generator v4 keeps the version-3 scene, material, source, receiver, and duration
distributions and changes the ordinary scene budgets to 32x32, 48x48, 64x64, and
96x96. The budget-OOD split remains 80x80 and 112x112, so it is still disjoint from
training budgets.

The 32x32 and 48x48 cases target allocation under scarce cells. They do not relax
PML collars, geometry anchors, or the maximum adjacent-cell ratio of 1.4. A
scene-budget pair with no legal mesh is an explicit infeasible sample and must not
be silently repaired by dropping anchors or weakening grading. Training should
balance budget strata and validation should compare strategies at equal budgets.

Existing generator-v3 manifests and their dataset IDs remain unchanged. New
manifests record generator version 4 and include all four ordinary budgets.
