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

"""Measure the bordered-block structure of the design matrix, and this machine's dgemm peak.

The design matrix's columns are generator indices, which carry a gate name.  This probe asks:

* Does each row's support fall inside a single layer gate's columns plus a shared border?
* How big is the border?
* What is the sum of the per-block ranks, versus the global rank?
* What fraction of the global ``linearly_independent_rows`` flop count would a block-aware
  elimination avoid?

and separately measures peak dgemm throughput so that "already BLAS-bound" is a measurement.
"""

import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

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

CASE = sys.argv[1] if len(sys.argv) > 1 else "grid32"
SPAM = ("P", "M")


def dgemm_peak() -> float:
    """Best observed double-precision GFLOP/s over a few square sizes."""
    best = 0.0
    for n in (512, 1024, 2048):
        a, b = np.random.rand(n, n), np.random.rand(n, n)
        a @ b  # warm up
        t = time.perf_counter()
        a @ b
        elapsed = time.perf_counter() - t
        best = max(best, 2 * n**3 / elapsed / 1e9)
    return best


case = case_by_name(CASE)
gate_set, topology = build_gate_set(case)
model = build_model(case, gate_set, topology)
layer_gates = [name for name in gate_set.model_gate_set if name not in SPAM]

empty = Experiment(fidelity_model=model, shots=case.shots, randomizations=case.randomizations)
experiment = empty
for stage_class in (EvenDepthPaths, Depth1Paths):
    for name in layer_gates:
        partial = stage_class(gates=[gate_set.model_gate_set[name]]).run(empty)
        experiment = experiment + RankReducePaths().run(partial)
experiment = experiment + RankReducePaths().run(SPAMPaths().run(empty))

A = experiment.design_matrix
data = A.data
n_rows, n_cols = data.shape

# Column -> gate name, and the block each column belongs to.
col_gate = np.empty(n_cols, dtype=object)
for generator_index, idx in A.column_index_map.items():
    col_gate[idx] = generator_index.gate_name
blocks = {name: np.nonzero(col_gate == name)[0] for name in {*layer_gates, *SPAM}}
border = (
    np.concatenate([blocks[name] for name in SPAM]) if any(len(blocks[n]) for n in SPAM) else []
)

print(f"{CASE}: design matrix {data.shape}, nnz={np.count_nonzero(data)}")
for name in layer_gates:
    print(f"  block {name:<10} {len(blocks[name]):>6} columns")
print(f"  border (P,M)        {len(border):>6} columns")

# How many layer-gate blocks does each row touch?
nz_rows, nz_cols = np.nonzero(data)
touched = Counter()
row_block = {}
per_row_blocks = [set() for _ in range(n_rows)]
for r, c in zip(nz_rows, nz_cols):
    gate = col_gate[c]
    if gate not in SPAM:
        per_row_blocks[r].add(gate)
for r, gates in enumerate(per_row_blocks):
    touched[len(gates)] += 1
    if len(gates) == 1:
        row_block[r] = next(iter(gates))
print(f"  rows by number of layer-gate blocks touched: {dict(sorted(touched.items()))}")
border_rows = int((data[:, border] != 0).any(axis=1).sum())
print(f"  rows touching the border: {border_rows} / {n_rows}")

# Per-block rank versus global rank.
t = time.perf_counter()
global_rank = np.linalg.matrix_rank(data)
t_global_rank = time.perf_counter() - t
block_ranks = {}
flops_blocks = 0
for name in layer_gates:
    rows = np.array([r for r, g in row_block.items() if g == name])
    cols = np.concatenate([blocks[name], border])
    sub = data[np.ix_(rows, cols)]
    block_ranks[name] = int(np.linalg.matrix_rank(sub))
    flops_blocks += 2 * sub.shape[0] * sub.shape[1] * block_ranks[name]
    print(
        f"  block {name:<10} rows={sub.shape[0]:>6} cols={sub.shape[1]:>6} rank={block_ranks[name]}"
    )
print(
    f"  sum of block ranks = {sum(block_ranks.values())}, global rank = {global_rank} "
    f"(numpy rank took {t_global_rank:.1f}s)"
)

flops_global = 2 * n_rows * n_cols * global_rank
print(f"\n  lir flops, global      : {flops_global:.3e}")
print(f"  lir flops, per block   : {flops_blocks:.3e}  ({flops_global / flops_blocks:.1f}x fewer)")

print(f"\n  dgemm peak on this machine: {dgemm_peak():.0f} GFLOP/s")
t = time.perf_counter()
A.linearly_independent_rows()
t_lir = time.perf_counter() - t
print(f"  lir measured: {t_lir:.3f}s -> {flops_global / t_lir / 1e9:.0f} GFLOP/s effective")
