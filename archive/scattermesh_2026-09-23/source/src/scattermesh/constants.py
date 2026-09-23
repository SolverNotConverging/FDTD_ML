"""SI constants used consistently by source, updates and analytic benchmarks."""

import numpy as np

EPS0 = 8.8541878128e-12
MU0 = 1.25663706212e-6
C0 = 1 / np.sqrt(EPS0 * MU0)
Z0 = np.sqrt(MU0 / EPS0)
