# Probes

Throwaway-shaped but preserved measurement scripts backing the numbers in
`experiment_building_performance.md`. These are **not** part of the benchmark suite: `run_bench.py`
measures end-to-end build time and memory, while each probe here answers one specific question about
*why* the time is where it is. They are kept so that every claim in the plan document can be
re-derived rather than trusted.

Run them from the **repository root** (they insert `benchmarks/` on `sys.path` themselves), using the
environment the package is installed into — the probes reach into internals and need the same Qiskit
version the library does:

```bash
.venv/bin/python benchmarks/probes/structure.py hex32   # or: -m benchmarks.probes.structure hex32
```

Each probe takes an optional case name (`hex32`, `grid32`, `hex64`, `grid64`, `hex128`, `grid128`,
`hex256`, `grid256`) and defaults to a small one. `blas.py` takes no argument. Costs scale the way
the baseline table does — `grid256` probes take ten minutes and ~10 GiB, `hex32` takes seconds.

## What each probe produced

| probe | question | numbers it produced in the plan document |
|-------|----------|------------------------------------------|
| `structure.py` | How sparse are the two `_matmul_matrix` operands, what would a sparse product cost, does the design matrix split into independent blocks, and is the result identical either way? | §2.A operand densities (LogPathMap 2.4 nnz/row, model 21 nnz/row, product 27); §3.1 sparse-vs-dense product times with the exactness check; the finding that the bipartite sparsity graph has exactly **1** connected component, so naive component splitting saves nothing |
| `border.py` | Given that there is one component, is the coupling confined to a small border? | §2 block/border column counts, the row block-touch histogram, per-block vs global rank, the `n * n_blocks / (n_per_block + border)` flop-reduction estimate (8.2x heavy-hex, 14.8x grid), and this machine's dgemm peak |
| `bbd.py` | Does the bordered-block structure hold at 64+ qubits, and what does exploiting it actually buy? | §3.4: `grid32`/`grid64` histograms `{0:32, 1:4408, ≥2:0}` and `{0:64, 1:9376, ≥2:0}`; measured `grid64` rank reduction 3.199 s → 0.410 s (7.8x); selection verified a superset of the global one, with excess exactly `2×|border|` |
| `lir.py` | Is `linearly_independent_rows` near the dense floor, and would a different block size or a LAPACK factorization beat it? | §2.B block-size sweep (64→3.49 s, **128→3.27, shipped and optimal**, 256→3.27, 512→3.70, 1024→5.27) and LAPACK QR of `A.T` (unpivoted 3.40 s, pivoted 13.38 s) — i.e. it already beats an unpivoted dense factorization of the same matrix, so only *fewer flops* help |
| `propagate.py` | How far is `ModelGate.clifford_propagate` from a properly indexed table lookup? | §3.2: 59.7 → 1.5 µs (`hex32`, 39x), 69.2 → 1.6 (`hex128`), 76.6 → 1.8 (`hex256`), 78.6 → 1.7 (`grid256`, 46x); the replacement is flat in qubit count, the current one is not. Output verified identical against the shipped method on 80 Paulis per case |
| `blas.py` | Clean dgemm peak, plus dense-vs-sparse at the recorded design-matrix shapes without rebuilding the big experiments | §2 peak of **345 GFLOP/s** double precision (M3 Pro, 12 cores); the observation that the in-build product runs at 221 GFLOP/s, i.e. genuinely BLAS-bound |

## Caveats worth knowing before trusting a re-run

- **`blas.py` shapes are recorded, not live.** Its `SHAPES` table came from instrumented runs, and
  the `hex128` inner dimension (`k=16232`) is larger than the actual in-build call
  (`(9816, 7720) @ (7720, 5242)`, 2.665 s), so `blas.py` **overstates** dense cost there by about
  2x. The plan document's recommendation estimates are based on the real in-build times, using
  `blas.py`'s sparse timings only as conservative upper bounds. Re-derive `SHAPES` before quoting its
  absolute numbers.
- **`border.py` uses `numpy.linalg.matrix_rank`** (an SVD) for ranks, which is why it defaults to
  `grid32`. `bbd.py` is the same question answered with `linearly_independent_rows` instead, which is
  what makes 64- and 128-qubit cases affordable.
- **`lir.py` monkeypatches** `indexed_matrix._ROW_REDUCTION_BLOCK_SIZE`. If that constant is renamed
  or the reduction is rewritten, the probe needs updating rather than believing.
- **The block structure `border.py` and `bbd.py` measure is not durable.** A planned `EvenDepthPaths`
  change will emit paths spanning two layers, after which rows touch up to two blocks instead of one.
  Both probes will still run; the `≥2` bucket of the histogram is the thing to look at once it lands.
  See §3.4 and §4.1 of the plan document.
