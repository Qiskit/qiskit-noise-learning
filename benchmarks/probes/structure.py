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

"""Probe the structure of the overall design matrix: sparsity and block decomposition.

Answers three level-1 questions with measurements rather than argument:

1. How sparse are the two operands of the big ``_matmul_matrix``, and what would a sparse
   product cost instead of the dense one?
2. Does the overall design matrix decompose into independent row/column blocks, and if so
   how much of ``linearly_independent_rows`` would that save?
3. Is the answer identical either way?
"""

import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

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
from qiskit_noise_learning.sequences import LogPathMap  # noqa: E402

CASE = sys.argv[1] if len(sys.argv) > 1 else "hex32"


def build(case):
    gate_set, topology = build_gate_set(case)
    model = build_model(case, gate_set, topology)
    layer_gates = {
        name: gate for name, gate in gate_set.model_gate_set.items() if name not in ("P", "M")
    }
    empty = Experiment(fidelity_model=model, shots=case.shots, randomizations=case.randomizations)
    experiment = empty
    for stage_class in (EvenDepthPaths, Depth1Paths):
        for gate in layer_gates.values():
            partial = RankReducePaths().run(stage_class(gates=[gate]).run(empty))
            experiment = experiment + partial
    experiment = experiment + RankReducePaths().run(SPAMPaths().run(empty))
    return model, experiment


model, experiment = build(case_by_name(CASE))
paths = experiment.paths
print(f"{CASE}: {len(paths)} paths")

# --- 1. operand sparsity -------------------------------------------------------------------
t = time.perf_counter()
P = LogPathMap(model.output_space).rows(paths)
t_P = time.perf_counter() - t

t = time.perf_counter()
M = model.rows(P.column_index_map.keys())
t_M = time.perf_counter() - t

Pd, Md = P.data, M.data
print(
    f"  LogPathMap.rows  {Pd.shape} nnz={np.count_nonzero(Pd):>9}  "
    f"density={np.count_nonzero(Pd) / Pd.size:.2e}  {t_P:.3f}s"
)
print(
    f"  model.rows       {Md.shape} nnz={np.count_nonzero(Md):>9}  "
    f"density={np.count_nonzero(Md) / Md.size:.2e}  {t_M:.3f}s"
)

# The library contracts over shared labels; here they line up exactly by construction.
shared = [k for k in P.column_index_map if k in M.row_index_map]
assert len(shared) == len(P.column_index_map) == len(M.row_index_map)
left = Pd[:, [P.column_index_map[k] for k in shared]]
right = Md[[M.row_index_map[k] for k in shared], :]

t = time.perf_counter()
dense = left @ right
t_dense = time.perf_counter() - t

Ls, Rs = sp.csr_matrix(left), sp.csr_matrix(right)
t = time.perf_counter()
sparse = Ls @ Rs
t_sparse = time.perf_counter() - t

flops_dense = 2 * left.shape[0] * left.shape[1] * right.shape[1]
print(f"  dense  product   {t_dense:.3f}s  ({flops_dense / t_dense / 1e9:.0f} GFLOP/s)")
print(f"  sparse product   {t_sparse:.3f}s  (speedup {t_dense / t_sparse:.0f}x)")
print(f"  agree: {np.abs(sparse.toarray() - dense).max():.2e}")
print(f"  product nnz={sparse.nnz} density={sparse.nnz / (dense.shape[0] * dense.shape[1]):.2e}")

# --- 2. block decomposition ---------------------------------------------------------------
A = experiment.design_matrix
Ad = A.data
S = sp.csr_matrix(Ad)
n_rows, n_cols = Ad.shape
# Connected components of the bipartite row/column graph.
bip = sp.bmat([[None, S], [S.T, None]], format="csr")
n_comp, labels = sp.csgraph.connected_components(bip, directed=False)
row_labels, col_labels = labels[:n_rows], labels[n_rows:]
sizes = []
for comp in range(n_comp):
    r = int((row_labels == comp).sum())
    c = int((col_labels == comp).sum())
    if r or c:
        sizes.append((r, c))
sizes.sort(reverse=True)
print(f"\n  design matrix {Ad.shape} nnz={S.nnz} density={S.nnz / Ad.size:.2e}")
print(f"  bipartite components: {len(sizes)}")
for r, c in sizes[:12]:
    print(f"    rows={r:>6} cols={c:>6}")
if len(sizes) > 12:
    print(f"    ... {len(sizes) - 12} more")

# --- 3. cost of lir, whole vs per block ---------------------------------------------------
t = time.perf_counter()
reduced = A.linearly_independent_rows()
t_whole = time.perf_counter() - t
print(f"\n  lir whole      {Ad.shape} -> {reduced.shape}  {t_whole:.3f}s")

row_order = {idx: label for label, idx in A.row_index_map.items()}
t_blocks = 0.0
kept = []
for comp in range(n_comp):
    r_idx = np.nonzero(row_labels == comp)[0]
    c_idx = np.nonzero(col_labels == comp)[0]
    if len(r_idx) == 0 or len(c_idx) == 0:
        continue
    sub = IndexedMatrix.from_index_lists(
        [row_order[i] for i in r_idx], [i for i in c_idx], Ad[np.ix_(r_idx, c_idx)]
    )
    t = time.perf_counter()
    sub_reduced = sub.linearly_independent_rows()
    t_blocks += time.perf_counter() - t
    kept.extend(sub_reduced.row_index_map.keys())
print(
    f"  lir per block  {len(kept)} rows kept       {t_blocks:.3f}s  "
    f"(speedup {t_whole / t_blocks:.1f}x)"
)
print(f"  same row set: {set(kept) == set(reduced.row_index_map)}")
print(f"  same count:   {len(kept)} vs {len(reduced.row_index_map)}")
