# This code is a Qiskit project.
#
# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""How close is ``linearly_independent_rows`` to the dense floor, and does blocking help?

Compares, on a real design matrix:

* the shipped blocked Gram-Schmidt at its current block size and at several others,
* LAPACK's column-pivoted QR of the transpose, which is the standard dense rank-revealing
  factorization and does *not* honour the prefix-greedy row preference, and
* an unpivoted QR, as a lower bound on any dense factorization of the same size.
"""

import sys
import time
from pathlib import Path

import numpy as np
import scipy.linalg as sla

sys.path.insert(0, str(Path("benchmarks").resolve()))

from qnl_bench import case_by_name  # noqa: E402
from qnl_bench.cases import build_gate_set, build_model  # noqa: E402

from qiskit_noise_learning.experiment_builder import Experiment  # noqa: E402
from qiskit_noise_learning.experiment_builder.stages import (  # noqa: E402
    Depth1Paths,
    EvenDepthPaths,
    RankReducePaths,
    SPAMPaths,
)
from qiskit_noise_learning.math import indexed_matrix  # noqa: E402

CASE = sys.argv[1] if len(sys.argv) > 1 else "grid64"
SPAM = ("P", "M")

case = case_by_name(CASE)
gate_set, topology = build_gate_set(case)
model = build_model(case, gate_set, topology)
empty = Experiment(fidelity_model=model, shots=case.shots, randomizations=case.randomizations)
experiment = empty
for stage_class in (EvenDepthPaths, Depth1Paths):
    for name, gate in gate_set.model_gate_set.items():
        if name in SPAM:
            continue
        experiment = experiment + RankReducePaths().run(stage_class(gates=[gate]).run(empty))
experiment = experiment + RankReducePaths().run(SPAMPaths().run(empty))

A = experiment.design_matrix
m, n = A.data.shape
print(f"{CASE}: {m} x {n}")

original = indexed_matrix._ROW_REDUCTION_BLOCK_SIZE  # noqa: SLF001
rank = None
for block in (64, 128, 256, 512, 1024):
    indexed_matrix._ROW_REDUCTION_BLOCK_SIZE = block  # noqa: SLF001
    t = time.perf_counter()
    reduced = A.linearly_independent_rows()
    elapsed = time.perf_counter() - t
    rank = reduced.shape[0]
    flops = 2 * m * n * rank
    marker = "  <- shipped" if block == original else ""
    print(f"  lir block={block:>5}  {elapsed:7.3f}s  {flops / elapsed / 1e9:6.0f} GFLOP/s{marker}")
indexed_matrix._ROW_REDUCTION_BLOCK_SIZE = original  # noqa: SLF001

At = np.asfortranarray(A.data.T)
t = time.perf_counter()
sla.qr(At, mode="economic", pivoting=True)
t_pivoted = time.perf_counter() - t
t = time.perf_counter()
sla.qr(At, mode="economic", pivoting=False)
t_plain = time.perf_counter() - t
print(f"  LAPACK pivoted QR of A.T   {t_pivoted:7.3f}s  (does not honour row order)")
print(f"  LAPACK plain QR of A.T     {t_plain:7.3f}s  (no rank revealing at all)")
print(f"  rank = {rank}")
