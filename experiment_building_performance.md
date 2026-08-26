# Experiment-building performance

Where the time goes when building a noise-learning experiment, and what to do about it. All numbers
measured on an Apple M3 Pro (12 cores, 36 GiB), one case per process. Peak double-precision dgemm on
that machine is **345 GFLOP/s**, which is the ceiling to compare the linear-algebra stages against.

---

## 1. The benchmarked examples

`benchmarks/` holds a reproducible, credential-free suite. A case is `(topology family, qubit count)`
and nothing else; from those two numbers the harness derives a coupling map, an edge colouring into
disjoint two-qubit layers, a `Target`, a `QiskitGateSet` with one `cz` layer gate per colour, and a
2-local `PauliLindbladModel` over those gates with 1-local `P`/`M`.

```bash
python benchmarks/run_bench.py --self-check    # validate the synthetic topologies
python benchmarks/run_bench.py --all --json out.json
```

The workload mirrors the experiment-builder cell of
`notebooks/utility_benchmark_v2/executor_utility_benchmark_qnl_advanced_more.ipynb`, which needs a
live `QiskitRuntimeService` and so cannot be re-run at will. The suite makes that shape of workload
runnable offline, at several sizes, indefinitely.

| case | topology | qubits | edges | edges/qubit | layer gates | generators | per layer gate |
|------|----------|-------:|------:|------------:|------------:|-----------:|---------------:|
| `hex32`   | heavy-hex   | 32  | 33  | 1.03 | 3 |  1243 |  393 |
| `grid32`  | square grid | 32  | 52  | 1.62 | 4 |  2320 |  564 |
| `hex64`   | heavy-hex   | 64  | 68  | 1.06 | 3 |  2540 |  804 |
| `grid64`  | square grid | 64  | 112 | 1.75 | 4 |  4928 | 1200 |
| `hex128`  | heavy-hex   | 128 | 142 | 1.11 | 3 |  5242 | 1662 |
| `grid128` | square grid | 128 | 232 | 1.81 | 4 | 10144 | 2472 |
| `hex256`  | heavy-hex   | 256 | 291 | 1.14 | 3 | 10673 | 3387 |
| `grid256` | square grid | 256 | 480 | 1.88 | 4 | 20864 | 5088 |

* **`heavy_hex`** — the IBM "brick" lattice used by Eagle and Heron. Degree ≤ 3, 3 edge colours.
  `heavy_hex_lattice(8, 16)` reproduces `FakeFez`'s coupling map exactly (176 edges, zero symmetric
  difference), asserted by `--self-check`. Sizes that are not a whole number of bands are cut down by
  breadth-first truncation, leaving a full lattice plus a ragged partial band — the shape a real
  device subset has.
* **`grid`** — a square lattice, as on Nighthawk-class devices (`ibm_miami` is exactly 12x10).
  Degree ≤ 4, 4 edge colours. All four sizes factor exactly (4x8, 8x8, 8x16, 16x16).

Both families hit the four sizes **exactly**, so a `hex`/`grid` pair at the same size differs only in
coupling density. That is the comparison the suite is built for: generator count — and therefore
design-matrix width — scales with **edges**, not qubits, so the grid family carries roughly 1.7x the
parameters per qubit and one extra layer gate.

Both topologies are generated analytically, so a case is byte-identical on any machine with the same
library versions. Every run also digests its own output (path list ordered and as a set, instruction
sequences, relations, and a permutation-blind design-matrix digest), so an optimization can be checked
for having left the answer alone.

`hex128` is the case to compare against history: 15.4 s, 1662 generators per layer gate, 5114 paths,
against 15.4 s / 1659 / 5104 for the real 127-qubit device the notebook used. The synthetic suite
reproduces the real workload.

### Baseline

| case | seconds | peak RSS | design matrix | density | paths | seqs |
|------|--------:|---------:|---------------|--------:|------:|-----:|
| `hex32`   |   1.4 |  306 MiB | 2324 x 1243   | 2.2e-02 |  1211 | 48 |
| `grid32`  |   3.5 |  792 MiB | 4440 x 2320   | 2.0e-02 |  2288 | 75 |
| `hex64`   |   3.7 | 1088 MiB | 4752 x 2540   | 1.1e-02 |  2476 | 49 |
| `grid64`  |  13.2 | 2369 MiB | 9440 x 4928   | 1.0e-02 |  4864 | 78 |
| `hex128`  |  15.4 | 3114 MiB | 9816 x 5242   | 7.4e-03 |  5114 | 49 |
| `grid128` |  72.4 | 8689 MiB | 19440 x 10144 | 5.1e-03 | 10016 | 78 |
| `hex256`  |  91.9 | 7813 MiB | 19996 x 10673 | 3.8e-03 | 10417 | 49 |
| `grid256` | 600.7 |10325 MiB | 40000 x 20864 | 2.6e-03 | 20608 | 78 |

`grid256` is the practical ceiling on a 36 GiB machine: 10 minutes, 10 GiB resident, and a **40.7 GiB
peak memory footprint**. It does not swap, but it has no headroom.

---

## 2. What is actually bottlenecking the time

Four costs account for essentially the whole build. Seconds, per case:

| | `hex32` | `grid32` | `hex64` | `grid64` | `hex128` | `grid128` | `hex256` | `grid256` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **A** design-matrix products | 0.07 | 0.37 | 0.42 | 2.88 | 3.38 | 24.2 | 28.0 | **284.0** |
| **B** rank reduction, overall | 0.10 | 0.44 | 0.53 | 3.21 | 3.85 | 27.2 | 34.2 | **238.2** |
| **B'** rank reduction, per gate | 0.08 | 0.18 | 0.28 | 0.83 | 1.38 | 4.4 | 8.5 | 27.2 |
| **C** path generation | 0.85 | 1.85 | 1.76 | 4.25 | 3.99 | 9.5 | 8.7 | 21.4 |
| **D** everything else | 0.25 | 0.64 | 0.74 | 2.05 | 2.80 | 7.1 | 12.5 | 29.9 |
| total | 1.4 | 3.5 | 3.7 | 13.2 | 15.4 | 72.4 | 91.9 | 600.7 |

The regime shifts sharply with size. Below ~64 qubits the build is **C**, pure-Python Pauli
bookkeeping (53–63% of the build). At 128+ it is **A** and **B**, two dense BLAS calls (up to 85%).
Denser topology arrives in the second regime sooner, because generator count scales with edges.

### A. A dense matrix product on operands that are ~1% dense

`Experiment._compute_design_matrix` builds `LogPathMap(model.output_space) @ model` and calls
`.rows(paths)`. `ComposedLinearMap.rows` materializes `LogPathMap.rows(fidelities)` as a **dense**
array and hands it to `IndexedMatrix._matmul_matrix`, which does a dense BLAS `left @ right`.

Measured nonzero structure (`hex32`, exact):

| operand | shape | nnz | density | per row |
|---|---|---:|---:|---:|
| `LogPathMap.rows` | (2324, 1763) | 5 632 | 1.4e-03 | 2.4 |
| `model.rows` | (1763, 1243) | 37 104 | 1.7e-02 | 21 |
| product | (2324, 1243) | 63 556 | 2.2e-02 | 27 |

A path-map row holds one entry per distinct fidelity index in the path — a handful. A model row holds
one entry per generator anticommuting with the row's Pauli — about 21, **independent of qubit count**.
Neither density grows with the problem; the dense flop count grows as its cube.

**The fundamental problem is sparse-sparse matrix multiply.** Its cost is
`sum_i nnz(P_i) * nnz-per-model-row ≈ 2.4 * 21 * n_paths` — linear in path count, about 50
multiply-adds each. At `grid256` that is 2e6 operations against the 5.8e13 the dense path performs: a
factor of 3e7 in arithmetic. The single `grid256` call is 263.0 s of the 600.7 s build, and it runs at
221 GFLOP/s — it is not a missed-vectorization problem, it is the wrong algorithm.

This is also the memory ceiling. At `grid256` the three dense arrays involved in that one product are
11.1 + 5.8 + 6.7 = **23.1 GiB**, the bulk of the 40.7 GiB footprint.

### B. Rank reduction, which is already at the dense floor

`IndexedMatrix.linearly_independent_rows` is a blocked Gram-Schmidt. On the real `grid64` design
matrix (9440 x 4928, rank 4864):

| method | time |
|---|---:|
| blocked Gram-Schmidt, block 64 | 3.49 s |
| **blocked Gram-Schmidt, block 128 (current)** | **3.27 s** |
| blocked Gram-Schmidt, block 256 | 3.27 s |
| blocked Gram-Schmidt, block 512 | 3.70 s |
| blocked Gram-Schmidt, block 1024 | 5.27 s |
| LAPACK `dgeqrf` of `A.T`, unpivoted, not rank-revealing | 3.40 s |
| LAPACK pivoted QR of `A.T` (does not honour row order) | 13.38 s |

The current block size is already optimal, and the current routine is **faster than an unpivoted
LAPACK QR of the same matrix** while additionally revealing the rank and honouring the prefix-greedy
row preference. It sustains 145 GFLOP/s at both `grid128` (3.95e12 flops / 27.17 s) and `grid256`
(3.44e13 / 238.2 s) — bandwidth-bound, exactly like LAPACK on this shape.

**There is no constant factor to win here.** The only lever is doing fewer flops.

That lever exists, and it is structural. Design-matrix columns are `GeneratorIndex`, which carry a
gate name, so columns partition by gate. Measured:

| | `grid32` | `grid64` |
|---|---|---|
| design matrix | 4440 x 2320 | 9440 x 4928 |
| layer-gate blocks | 4 x 564 cols | 4 x 1200 cols |
| border (`P`, `M`) | 64 cols = 2/qubit | 128 cols = 2/qubit |
| rows touching 0 / 1 / **≥2** layer-gate blocks | 32 / 4408 / **0** | 64 / 9376 / **0** |

**Every row touches at most one layer-gate block.** The matrix is exactly *bordered block diagonal*:
one block per layer gate, coupled only through 2 border columns per qubit. The handful of rows
touching no layer block are the pure-SPAM rows. Note the bipartite sparsity graph has exactly **one**
connected component — the border connects everything — so naive component splitting does not find
this; the border has to be recognized *as* a border.

**This particular structure is not durable.** It holds because every path today involves a single
layer gate. A planned update to `EvenDepthPaths` will generate paths in which two layers appear in the
same path, and those rows will touch two layer-gate blocks — so the matrix stops being bordered block
diagonal. What survives is weaker but still strong: rows touch **at most two** of `B` blocks. See §3.4
for what that does and does not preserve.

### C. Propagating a weight-≤4 Pauli through a dense N-qubit representation

`ModelGate.clifford_propagate` converts its argument to a `SparseObservable`, takes
`.pauli_bases()[0]` — a **dense `Pauli` over all N qubits** — evolves it through every Clifford whose
qubits intersect the support, and after each evolve rebuilds the support with a set comprehension over
`dense_pauli.to_label()[::-1]`, an O(N) Python string per step. It also tests *all* of the gate's
Cliffords for overlap; a 256-qubit layer gate has 98.

Measured 59.7 µs/call at `hex32` rising to 78.6 µs at `grid256` — growing with N, which is the O(N)
label work. Under `cProfile` at `grid32` it is 20 368 calls and 3.28 s cumulative, 48% of the build;
`FidelityIndex.from_gate` is 2.97 s cumulative, essentially all of it propagation. It is the
overwhelming majority of row **C**. (Qiskit's `_evolve_clifford` also recomputes `clifford.adjoint()`
on every call — 9 036 `_conjugate_transpose` calls in one `grid32` build — but that is a symptom, not
a separate item.)

**The fundamental problem** is: propagate a Pauli through a fixed layer of Cliffords each acting on at
most 2 qubits. A 2-qubit Clifford's action on Paulis is a map from 16 phaseless local Paulis to 16
local Paulis, tabulable once per Clifford. Propagation is then O(support size) and **independent of
N**: read the support, visit only the Cliffords indexed to those qubits, splice in the table's answer.

### D. Everything else

Sequence generation, merging, completion, binding, and model construction. 18% of `hex32` and 5% of
`grid256`; `model/generators` is the largest single piece at 256 qubits (4.5–6.5 s). Nothing here is a
bottleneck at any size, and prior work already established that the merge-sequence grouping is
provably optimal on measured instances.

---

## 3. Recommendations, in increasing order of how large the change is

Estimated seconds after each change, cumulative down the table, for every case. Derived from the
per-case measurements in §2 with the measured speedup of each change applied to the cost it removes;
directly measured where a row says so.

**3.1–3.3 are actionable now**; they depend only on properties that do not change (operand sparsity,
the locality of Clifford action). **3.4–3.5 are deferred** until the two-layer `EvenDepthPaths` update
lands, because they exploit a structure that update removes. **3.6 needs a contract decision** and a
prototype that does not exist yet. So the firm near-term deliverable is:

| | `hex32` | `grid32` | `hex64` | `grid64` | `hex128` | `grid128` | `hex256` | `grid256` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 1.4 | 3.5 | 3.7 | 13.2 | 15.4 | 72.4 | 91.9 | 600.7 |
| after 3.1–3.3 | 0.6 | 1.6 | 2.0 | 6.9 | 8.8 | 40.6 | 57.1 | **299** |
| speedup | 2.3x | 2.2x | 1.9x | 1.9x | 1.8x | 1.8x | 1.6x | **2.0x** |

— roughly 2x everywhere, plus the memory ceiling removed, which is the more valuable half. Everything
beyond that is upside contingent on the two open questions.

### 3.1 Sparse design-matrix product, dense result — *internal to one method*

Have `_matmul_matrix` convert both operands to CSR, multiply, and densify the result.
`IndexedMatrix.data` stays an `np.ndarray` exactly as documented; no other code changes.

The sparse product measures **0.001–0.011 s** at every size in the suite, against 0.07–284 s dense.
The residual cost is the densification: measured 0.08 s / 1.47 GiB at `grid128` and **0.65 s / 6.22
GiB** at `grid256`. Correctness is checkable — on real `hex32` operands the sparse and dense products
agree to 0.0e+00.

| | `hex32` | `grid32` | `hex64` | `grid64` | `hex128` | `grid128` | `hex256` | `grid256` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 1.4 | 3.5 | 3.7 | 13.2 | 15.4 | 72.4 | 91.9 | 600.7 |
| after 3.1 | 1.3 | 3.1 | 3.3 | 10.4 | 12.1 | 48.4 | 64.2 | **318** |
| speedup | 1.0x | 1.1x | 1.1x | 1.3x | 1.3x | 1.5x | 1.4x | **1.9x** |

Smallest possible diff, and it removes the largest single cost in the largest case. Do this first.

### 3.2 Table-driven Clifford propagation — *internal to `ModelGate`*

Build, once per gate, a per-Clifford code table plus a `qubit -> Cliffords` index, and propagate by
lookup over the support. No dense `Pauli` anywhere, and only overlapping Cliffords are visited.

Prototyped and verified against the current output on 80 Paulis per case:

| case | Cliffords in gate | current | table | speedup |
|---|---:|---:|---:|---:|
| `hex32`   |  12 | 59.7 µs | 1.5 µs | 39x |
| `hex128`  |  48 | 69.2 µs | 1.6 µs | 42x |
| `hex256`  |  98 | 76.6 µs | 1.8 µs | 42x |
| `grid256` | 106 | 78.6 µs | 1.7 µs | 46x |

Flat in N where the current cost grows. Table construction is one-off per gate and should be built
from the Clifford's symplectic matrix directly (the prototype built it by calling the slow path, hence
its 12–132 ms, which a real implementation would not pay).

Estimates below assume ~75% of row **C** is propagation, which is the conservative end of what
`cProfile` shows:

| | `hex32` | `grid32` | `hex64` | `grid64` | `hex128` | `grid128` | `hex256` | `grid256` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| after 3.1 + 3.2 | 0.7 | 1.8 | 2.1 | 7.3 | 9.2 | 41.5 | 57.8 | 302 |
| cumulative speedup | **2.0x** | **2.0x** | 1.8x | 1.8x | 1.7x | 1.7x | 1.6x | 2.0x |

This is the only item that helps the 32–64 qubit cases much, where path generation is the build.

### 3.3 Sparse storage inside `IndexedMatrix` — *one class, one documented type*

Store either a dense array or CSR, densifying lazily in `.data`. `LogPathMap.rows` and
`PauliLindbladModel.rows` build CSR directly (they already receive `IndexedVector` rows, i.e. COO
triples), the product stays sparse, and `linearly_independent_rows` densifies what it needs.

The time win on top of 3.1 is small — it removes the `add_rows` dense assembly (0.06–2.0 s) and the
densification 3.1 still pays. **The win is memory**, and memory is what caps the suite at 256 qubits:

| dense intermediates in the overall product | `hex32` | `grid32` | `hex64` | `grid64` | `hex128` | `grid128` | `hex256` | `grid256` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| now | 73 MB | 280 MB | 312 MB | 1.3 GiB | 1.3 GiB | 5.4 GiB | 5.5 GiB | **23.1 GiB** |
| sparse | ~0.9 MB | ~1.7 MB | ~1.9 MB | ~4 MB | ~5 MB | ~10 MB | ~10 MB | **~30 MB** |

Peak footprint at `grid256` falls from 40.7 GiB to a few GiB, which is what makes 256 qubits — and
beyond — comfortable rather than marginal. Wall-clock lands within ~5% of the 3.1 + 3.2 row above.

**Risk to check:** any refactor that builds the dense array differently may change **column order**,
currently insertion order. Column order feeds `linearly_independent_rows`, so it can change *which
rows are selected* — a behavioural change, not just a numerical one. Verify against the suite's
fingerprints, do not assume.

### 3.4 Pre-partition the rank reduction — *deferred until `EvenDepthPaths` is updated*

**Status: revisit after the two-layer `EvenDepthPaths` change lands.** The principle below — cut the
flop count by partitioning the problem, without touching the numerical method — is the largest
remaining lever, and the measurement below shows the approach works. But the specific partition it
exploits is a property of today's path set, and the planned update to `EvenDepthPaths` (paths in which
two layers appear together) removes it. Re-measuring on the new path set is the first step, not
implementing this as written.

#### What was measured on today's structure

Because every row touches at most one layer-gate block, the reduction splits:

1. **Per block**, greedy prefix selection over (that block's columns + the border). Independent per
   block, so also trivially parallel.
2. **A correction pass** over cross-block dependencies. A cross-block dependency requires a
   combination whose block part cancels within each participating block, so it lives entirely in the
   border's 2n-dimensional space. The phase-2 problem is at most `n_blocks x |border|` by `|border|` —
   512 x 128 at `grid64`. Negligible.

Measured on `grid64`: whole-matrix 3.199 s → **per-block 0.410 s, 7.8x**, with the per-block selection
a verified **superset** of the global one. The excess is exactly `2 x |border|` in both cases measured
(128 at `grid32`, 256 at `grid64`) — a structural quantity, which is what makes phase 2 small.

Flop reduction is `n * n_blocks / (n_per_block + border)` and is essentially size-independent:
**8.2x for heavy-hex** (3 layer gates), **14.8x for grid** (4). The `grid64` measurement realizes 53%
of its flop bound, the shortfall being lower BLAS efficiency on narrower blocks; larger cases have
wider blocks and should do better, so the estimates below use the measured 53% and are conservative.

| | `hex32` | `grid32` | `hex64` | `grid64` | `hex128` | `grid128` | `hex256` | `grid256` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| after 3.1–3.4 | 0.6 | 1.4 | 1.7 | 4.5 | 6.3 | 17.8 | 31.5 | **94** |
| cumulative speedup | 2.3x | 2.5x | 2.2x | **2.9x** | 2.4x | **4.1x** | 2.9x | **6.4x** |

`grid64`'s figure is the one directly measured (3.199 → 0.410 s on the real matrix); the rest apply
its efficiency ratio to the per-case flop bound. **Treat this whole table as an upper bound that the
`EvenDepthPaths` change will erode**, by how much depending on the next subsection.

#### What survives two-layer paths

The two-level scheme itself generalizes; only its arithmetic advantage shrinks. With rows touching at
most two of `B` blocks:

* Rows touching **one** block still reduce independently within (that block + border), exactly as
  above, and they will still be the majority.
* Rows touching **two** blocks form a coupling problem that is no longer confined to a `2n`-dimensional
  border. Phase 2 grows from `n_blocks x |border|` by `|border|` to something whose width is the
  border plus the columns of the block pairs actually co-occurring — bounded by, but likely much
  smaller than, the full matrix.

So the shape becomes: cheap independent phase 1 over single-layer rows, plus a phase 2 sized by *how
many two-layer paths there are and which layer pairs they use*. If two-layer paths are a modest
fraction of the total, most of the win survives; if they dominate, little does. That fraction is not
knowable until the change lands, which is exactly why this is deferred rather than estimated.

Two things worth setting up so the answer is cheap to get later:

1. The row/block incidence is a **hypergraph partitioning** question, not a special-case trick. Each
   row is a hyperedge over the blocks it touches; the goal is an ordering with a small separator. If
   two-layer paths cluster on a few layer pairs (plausible — adjacent layers are the interesting ones),
   merging those pairs into supernodes restores a bordered structure over `B/2`-ish coarser blocks.
   Worth measuring the co-occurrence pattern first thing.
2. The benchmark suite already reports the ingredients. Add a per-row block-touch histogram to the
   harness now, so the day `EvenDepthPaths` changes, the new structure is a single run away rather
   than a fresh investigation.

Where the block structure should live is the design question, and it is unchanged by any of this.
`IndexedMatrix` knows nothing about its column labels beyond hashability, so it cannot discover a
partition itself. Either `linearly_independent_rows` takes an optional column-partition argument,
supplied by `RankReducePaths` which does know the gate structure (additive, opt-in, no behaviour
change), or a dedicated block-structured matrix type is introduced (more invasive, cleaner). The
argument form is the better bet precisely because it does not hard-code *which* partition — it would
survive the `EvenDepthPaths` change with a different partition passed in.

### 3.5 Restructure the stages around the block structure — *deferred with 3.4*

**Status: same gate as 3.4.** The first half of this (stop discarding the design-matrix cache) is
independent of path structure and can be done any time. The second half (per-gate ownership of the
reduction) inherits 3.4's assumption and waits with it.

Two related facts:

* `Experiment.__add__` builds a fresh `Experiment` and `replace` clears `_design_matrix_cache`
  (`experiment.py:240`). So `RankReducePaths` discards the matrix it just consumed, and each `+`
  discards both operands'. At `grid256` that is 31.4 s of per-part design-matrix work thrown away
  before the overall product spends 269.7 s recomputing the union.
* The per-gate reductions (row **B'**, 27.2 s at `grid256`) compute exactly what 3.4's phase 1 needs,
  and then throw it away too.

Put together: **let each path builder own its block's reduction, and make composition do only the
coupling correction.** The pipeline becomes per-gate path generation and reduction (independent, and
parallelizable across gates) → a cheap coupling correction → sequence generation. The monolithic
overall design-matrix and overall rank-reduce stages stop existing.

With two-layer paths this no longer partitions cleanly by gate — a two-layer path belongs to no single
gate's builder. The restructure then needs a notion of a *group* of gates owning a reduction, which is
the supernode idea from 3.4 arriving at the pipeline level. Worth designing with that in mind rather
than around single gates, since the change is known to be coming.

| | `hex32` | `grid32` | `hex64` | `grid64` | `hex128` | `grid128` | `hex256` | `grid256` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| after 3.1–3.5 | 0.5 | 1.2 | 1.4 | 3.6 | 4.9 | 13.3 | 22.9 | **66** |
| cumulative speedup | 2.7x | 2.9x | 2.7x | 3.7x | 3.1x | 5.4x | 4.0x | **9.1x** |
| plus across-gate parallelism | — | — | — | — | — | ~6 s | ~10 s | ~30 s |

A note on the "carry the Gram-Schmidt record across stages" idea specifically: measured, it is worth
much less than it looks. The overall reduction is already fed pre-reduced rows — at `grid128` the four
per-gate blocks arrive as 2380 + 2356 + 2344 + 2344 and the overall pass still drops 19440 rows to
10016, so *those 19440 are the survivors*; there is nothing for a carried basis to skip. Each
surviving row still costs a full orthogonalization against the accumulated basis, `2 * n_cols * rank`,
regardless of how it arrived. The saving is real but bounded by the per-gate passes' own work. It
falls out for free once 3.4 is in place — phase 1 *is* the per-gate reduction — which is the version
worth pursuing.

### 3.6 Priority tiers: keep the preference semantics, drop the total order — *changes a documented contract*

The row-order guarantee in `linearly_independent_rows` is **not** an implementation artifact. It is the
mechanism by which a user expresses *preferences* over which paths are retained — most concretely, that
unbound paths are preferred over bound ones because they yield more informative model estimates, and
plausibly other preferences of the same kind. A user orders path insertion by preference and the
algorithm honours it. Dropping the guarantee outright is therefore not on the table.

What *is* on the table is weakening it from a total order to an ordered partition, because the faster
numerical methods only need the weaker thing:

> Partition rows into **priority tiers**. Process tiers in priority order; within a tier, use any
> method that returns a maximal independent extension of what has already been accepted.

**This is exactly equivalent to today's behaviour at tier granularity, not an approximation.** Linear
independence of rows is a matroid, priorities are weights, and prefix-greedy is the matroid greedy
algorithm — so it returns a maximum-weight basis. Every maximum-weight basis of a matroid has the same
weight profile, so **the number of rows drawn from each priority tier is invariant** no matter how ties
inside a tier are broken. Tier-order-respecting selection cannot do worse on any preference the tiers
encode. The only thing that becomes unspecified is *which* rows are taken from within a single tier —
which is precisely the content of calling that tier equal-priority.

This is the "combination method" framing, and the practical structure is close to ideal for it: the
preference structure is a handful of coarse tiers (unbound before bound), and the `EvenDepthPaths`
output is one large equal-priority block. So almost all the rows sit in one tier, where any fast
unordered method is legal, and the sequential tier-by-tier logic only runs a few times.

What that unlocks:

* **Sparse LU with Markowitz pivoting**, which chooses its own pivot order and can exploit the 1e-3
  density directly rather than densifying. This is the method most likely to beat everything above,
  and today's contract is the only thing ruling it out.
* **LAPACK pivoted QR** within a tier. Measured 13.38 s against 3.27 s at `grid64`, so on *dense* data
  it is a loss and not worth it — noted only so it is not re-proposed.

Not estimated, deliberately: a sparse-LU rank reduction on this structure has not been prototyped, and
the honest thing is that it could plausibly be another large factor on the dominant remaining cost or
could disappoint. It is the one item here where the measurement does not exist yet.

Three things to pin down before prototyping: whether the tiers a user actually wants are expressible as
an ordered partition of the insertion sequence (if a preference is *pairwise* rather than tier-based,
this framing does not capture it); whether anything downstream depends on the specific rows chosen
rather than on the tier counts (the matroid argument protects the counts, nothing more); and that the
selection stays at **row** granularity, since the matroid argument fails once a path contributes several
rows that must be kept or dropped together — see §4.2.

---

## 4. Constraints from planned changes

Three known changes bear on the recommendations above. None of them invalidates 3.1–3.3.

### 4.1 Two-layer paths from `EvenDepthPaths`

Removes the bordered-block-diagonal structure that 3.4 and 3.5 exploit. Covered in place, in §3.4.

### 4.2 Unbound-path pre-factors: one path, two rows

The exponential fit to an unbound path yields both a decay rate and a constant pre-factor. Only the
rate is used today, as an estimate of the repeatable fragment's fidelities. The pre-factor also carries
information — about SPAM, mixed with the path's non-repeated boundary fragments — and a planned
depth-0-like addition would use it. So **an unbound path will contribute two design-matrix rows, not
one.**

What this does not affect:

* **The block structure.** A pre-factor row touches the SPAM border plus, at most, the boundary
  fragment's own gate columns. That is still "at most one block plus border", so this change is
  compatible with bordered block diagonality — unlike 4.1. If anything it thickens the border's row
  population, which grows 3.4's phase 2 but does not break it.
* **What drives inclusion of an unbound path.** That remains the repeatable fragment, i.e. the rate row.
  The pre-factor row is a consequence of running the experiment, not a reason to run it.

What it does affect, and the one thing worth being careful about:

**Row selection and path selection decouple, so a single global row-level greedy is no longer
coherent.** Today `RankReducePaths` selects rows and maps the survivors straight back to paths. With
two rows per unbound path that breaks in both directions: a row-level greedy could retain a pre-factor
row whose path was not selected (meaningless), and keeping a path means accepting both of its rows
whether or not the second one adds rank.

Worse, selection over *groups* of rows is **not** a matroid, so the greedy-optimality guarantee that
§3.6 leans on does not survive naive grouping. Concretely, with rows `v1, v2` grouped as one path `A`,
and singleton paths `B = {v1}`, `C = {v2}`: `{B, C}` is independent and `{A}` is independent, but
neither `B` nor `C` can be added to `{A}`. Augmentation fails, so tier counts are no longer invariant
and greedy no longer returns a maximum-weight solution.

The fix is to keep the staging the user's description already implies, rather than to generalize the
greedy:

1. Select unbound paths by their **rate rows only** — a plain row matroid, so all of §3.6 applies.
2. Admit the selected paths' **pre-factor rows into the accumulated span for free**, without treating
   them as selectable.
3. Select fixed-depth paths against that **augmented** span.

Each stage is greedy on a matroid over a fixed prior subspace, so ordering preferences and tier
equivalence hold within each stage. What must be avoided is one undifferentiated greedy over mixed
rate, pre-factor, and fixed-depth rows.

Step 3 is the substantive consequence: **the extra SPAM information must be accounted for when
choosing which fixed-depth experiments to keep**, or redundant ones will be retained. Since the border
is only `2 * n_qubits` columns wide (64 at 32 qubits, 512 at 256) and thousands of unbound paths are
retained, their pre-factor rows may cover much of the SPAM space on their own — which would *reduce*
the fixed-depth and dedicated-SPAM workload rather than add to it. Net effect on build time is
plausibly negative; worth measuring once implemented, since fewer retained rows is cheaper everywhere.

One scale caveat: pre-factor estimates are not statistically comparable to rate estimates, so their
rows will carry different magnitudes. That is another instance of the tolerance problem noted at the end
of this section, and it maps naturally onto §3.6's tiers — a pre-factor row is lower-information than a
rate row and belongs in a lower tier if it is ever made selectable.

### 4.3 The design matrix will stop being integer-valued

Planned fidelity → fidelity maps take **convex combinations** of fidelities, with row entries of the
form `k_i / n` for integers `k_i, n` and `sum_i k_i = n`. Today every entry is an exact small integer
in a float array (`2.0` from `PauliLindbladModel._row`, small multiplicities from `LogPathMap`), and
integer-valued float arithmetic is exact below 2^53.

**No recommendation above requires integrality.** `_data` is already `dtype=float` end to end, 3.2
touches no matrix values at all, and CSR multiply and bordered-block elimination are dtype-agnostic.
Three consequences do need recording:

**Validation.** The evidence for 3.1 includes "sparse and dense agree to 0.0e+00" — that
bit-exactness is *because* the operands are integer-valued. With `k_i/n` entries, CSR and dense BLAS
accumulate in different orders and will differ in the last bits without either being wrong. The
acceptance test must become a tolerance comparison. Same for the harness:
`benchmarks/qnl_bench/fingerprint.py` digests row and column absolute sums rounded to 9 decimals,
exact for integers but marginal for rationals (worst-case accumulation over a 40000-row column is
`n * ulp ≈ 4e-8`, larger than the rounding grid). Loosen it to ~6 decimals when the maps land.

**Exact rank stays available.** Each row becomes `1/n` times an integer vector, and scaling rows
individually by nonzero scalars changes **neither the rank, nor which subsets are independent, nor the
prefix-greedy selection**. So clearing denominators per row makes the matrix integer, and exact
integer or fraction-free (Bareiss) elimination remains on the table for 3.6 — no common `n` needed
across maps, since the scaling is per row. An exact method needs no tolerance at all.

**One structural question for 3.4.** A convex combination over fidelities *of a single gate* preserves
"every row touches at most one block". One that mixes fidelities from **different layer gates** does
not: the matrix stays bordered block diagonal, but the border absorbs the mixed columns and phase 2
grows with it — the algorithm survives, the 8.2x/14.8x does not necessarily. Keeping the new maps
gate-local is cheap insurance for the largest speedup on this list.

### A correctness issue worth fixing first, independent of performance

`linearly_independent_rows` uses an **absolute** `tol=1e-8` on the norm of the residual row, with rows
never normalized (`indexed_matrix.py:271-272`). Rows have norm O(1)–O(10) today, so that is enormous
margin. Convex combinations shrink entries by `1/n`: a genuinely independent row built from a
fine-grained combination can have a residual norm near or below 1e-8 and be silently **dropped as
dependent**, losing a real degree of freedom. A relative criterion — compare against the row's own
pre-orthogonalization norm — is the fix, and it is worth making before the new maps land rather than
after. It also interacts with 3.4, where blocks reduced independently each see their own local scale.
