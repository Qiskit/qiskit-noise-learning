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

"""Confirm the bordered-block-diagonal structure at a larger size, and price the two-level idea.

Uses ``linearly_independent_rows`` itself for ranks rather than an SVD, so it is affordable on the
64- and 128-qubit cases.
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
from qiskit_noise_learning.math import IndexedMatrix  # noqa: E402

CASE = sys.argv[1] if len(sys.argv) > 1 else "grid64"
SPAM = ("P", "M")

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

col_gate = np.empty(n_cols, dtype=object)
for generator_index, idx in A.column_index_map.items():
    col_gate[idx] = generator_index.gate_name
block_cols = {name: np.nonzero(col_gate == name)[0] for name in layer_gates}
border = np.nonzero(np.isin(col_gate.astype(str), SPAM))[0]

print(
    f"{CASE}: design matrix {data.shape} nnz={np.count_nonzero(data)} "
    f"density={np.count_nonzero(data) / data.size:.2e}"
)
print(
    f"  {len(layer_gates)} layer-gate blocks of {len(block_cols[layer_gates[0]])} columns, "
    f"border {len(border)} columns"
)

nonzero = np.abs(data) > 0
row_block_hits = np.stack([nonzero[:, block_cols[name]].any(axis=1) for name in layer_gates])
counts = Counter(row_block_hits.sum(axis=0).tolist())
print(f"  rows by layer-gate blocks touched: {dict(sorted(counts.items()))}")
print(f"  rows touching the border: {int(nonzero[:, border].any(axis=1).sum())} / {n_rows}")

row_order = {idx: label for label, idx in A.row_index_map.items()}

t = time.perf_counter()
whole = A.linearly_independent_rows()
t_whole = time.perf_counter() - t
print(f"\n  lir whole      {data.shape} -> {whole.shape}  {t_whole:.3f}s")

t_blocks = 0.0
kept = []
block_rank_sum = 0
for gate_idx, name in enumerate(layer_gates):
    rows = np.nonzero(row_block_hits[gate_idx])[0]
    cols = np.concatenate([block_cols[name], border])
    sub = IndexedMatrix.from_index_lists(
        [row_order[i] for i in rows], list(cols), data[np.ix_(rows, cols)]
    )
    t = time.perf_counter()
    reduced = sub.linearly_independent_rows()
    t_blocks += time.perf_counter() - t
    block_rank_sum += reduced.shape[0]
    kept.extend(reduced.row_index_map)
    print(f"    block {name:<10} {sub.shape} -> {reduced.shape}")

spam_only = np.nonzero(~row_block_hits.any(axis=0))[0]
print(f"    spam-only rows: {len(spam_only)}")

print(
    f"  lir per block  sum of block ranks = {block_rank_sum}, global rank = {whole.shape[0]}, "
    f"excess = {block_rank_sum - whole.shape[0] + len(spam_only)}"
)
print(f"  time per block {t_blocks:.3f}s  (speedup {t_whole / t_blocks:.1f}x)")
print(
    f"  per-block selection is a superset of the global one: "
    f"{set(whole.row_index_map).issubset(set(kept) | {row_order[i] for i in spam_only})}"
)
