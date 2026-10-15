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

"""Clean dgemm-peak measurement, and dense vs sparse cost at the measured design-matrix shapes.

The shapes and per-row nonzero counts are taken from the recorded benchmark runs, so the sparse
timings answer "what would the big ``_matmul_matrix`` cost if the operands stayed sparse" without
having to rebuild the 256-qubit experiment.

Caveat: because ``SHAPES`` is recorded rather than measured live, it can drift from the real calls.
The ``hex128`` inner dimension below is larger than the actual in-build product
(``(9816, 7720) @ (7720, 5242)``, 2.665s), so the dense figure there overstates the true cost by
roughly 2x.  Treat the dense column as an upper bound and re-derive ``SHAPES`` before quoting it;
the sparse column is the load-bearing one and is conservative either way.
"""

import time

import numpy as np
import scipy.sparse as sp

rng = np.random.default_rng(0)


def dgemm_peak() -> None:
    print("dgemm peak (double precision):")
    for n in (1024, 2048, 4096, 8192):
        a, b = rng.random((n, n)), rng.random((n, n))
        a @ b
        t = time.perf_counter()
        a @ b
        elapsed = time.perf_counter() - t
        print(f"  n={n:>5}  {elapsed:8.3f}s  {2 * n**3 / elapsed / 1e9:6.0f} GFLOP/s")


# (case, n_paths, n_fidelities, n_generators, nnz per LogPathMap row, nnz per model row)
# nnz-per-row figures come from the measured hex32/grid32 probes: the path map holds one entry
# per distinct fidelity index in the path, the model one entry per anticommuting generator.
SHAPES = [
    ("grid32 ", 4440, 3659, 2320),
    ("hex128 ", 9816, 16232, 5242),
    ("grid128", 19440, 16683, 10144),
    ("hex256 ", 19996, 15826, 10673),
    ("grid256", 40000, 34795, 20864),
]

PATH_NNZ = 2.4  # measured: 5632 nonzeros over 2324 rows on hex32
MODEL_NNZ = 21.0  # measured: 37104 nonzeros over 1763 rows on hex32


def random_sparse(n_rows: int, n_cols: int, nnz_per_row: float) -> sp.csr_matrix:
    """A CSR matrix with about ``nnz_per_row`` uniformly placed nonzeros per row."""
    per_row = max(1, int(round(nnz_per_row)))
    rows = np.repeat(np.arange(n_rows), per_row)
    cols = rng.integers(0, n_cols, size=n_rows * per_row)
    return sp.csr_matrix((np.full(rows.size, 2.0), (rows, cols)), shape=(n_rows, n_cols))


def compare() -> None:
    print("\ndense vs sparse for the design-matrix product:")
    print(
        f"  {'case':<8} {'shape':<26} {'dense (s)':>10} {'sparse (s)':>11} "
        f"{'speedup':>8} {'dense GiB':>10} {'sparse GiB':>11}"
    )
    for name, m, k, n in SHAPES:
        left = random_sparse(m, k, PATH_NNZ)
        right = random_sparse(k, n, MODEL_NNZ)
        t = time.perf_counter()
        product = left @ right
        t_sparse = time.perf_counter() - t

        flops = 2 * m * k * n
        # Time the dense product directly when it fits comfortably, else project from measured peak.
        dense_bytes = (m * k + k * n + m * n) * 8
        if dense_bytes < 6e9:
            ld, rd = left.toarray(), right.toarray()
            t = time.perf_counter()
            ld @ rd
            t_dense = time.perf_counter() - t
            note = ""
        else:
            t_dense = flops / (PEAK * 1e9)
            note = "*"
        print(
            f"  {name:<8} ({m},{k})@({k},{n})".ljust(37)
            + f"{t_dense:9.3f}{note:1} {t_sparse:11.4f} {t_dense / t_sparse:7.0f}x "
            f"{(m * k + m * n) * 8 / 2**30:9.2f} "
            f"{(product.nnz * 12 + left.nnz * 12) / 2**30:10.4f}"
        )
    print("  * projected from the measured dgemm peak rather than run (operands exceed 6 GB).")


PEAK = 0.0
if __name__ == "__main__":
    dgemm_peak()
    n = 8192
    a, b = rng.random((n, n)), rng.random((n, n))
    a @ b
    t = time.perf_counter()
    a @ b
    PEAK = 2 * n**3 / (time.perf_counter() - t) / 1e9
    compare()
