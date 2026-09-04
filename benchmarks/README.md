# Experiment-building benchmarks

A reproducible, credential-free benchmark suite for the cost of building a noise-learning
experiment. Nothing here is part of the shipped library, nothing imports it, and `pytest` does not
collect it (`testpaths` in `pyproject.toml` covers only `qiskit_noise_learning` and `test`).

## Running

```bash
python benchmarks/run_bench.py --self-check      # validate the synthetic topologies
python benchmarks/run_bench.py --list            # show the suite
python benchmarks/run_bench.py hex32 grid32      # run named cases
python benchmarks/run_bench.py --all --json out.json
```

Useful flags:

| Flag | Effect |
|------|--------|
| `--rank` | Also compute `design_matrix.rank`. **Off by default on purpose** — its dense SVD costs about as much as the entire build, and leaving it on hides everything else. |
| `--no-fingerprint` | Skip the output digests. |
| `--json PATH` | Dump every measurement, including the per-call log. |
| `--quiet` | Suppress per-phase progress lines. |
| `--profile [N]` | Also run under `cProfile` and print the top N functions by self time. Call overhead inflates every number in such a run — use it to find hot Python, not to quote timings. |
| `--pstats PATH` | With `--profile`, dump raw `pstats` for `snakeviz`/`gprof2dot`. |

## What a case is

A case is `(topology family, qubit count)` and nothing else — no backend, no service, no
credentials. From those two numbers the harness derives a coupling map, an edge colouring into
disjoint two-qubit layers, a `Target`, a `QiskitGateSet` with one `cz` layer gate per colour, and a
2-local `PauliLindbladModel` over those gates with 1-local `P`/`M`.

The workload mirrors the experiment-builder cell of
`notebooks/utility_benchmark_v2/executor_utility_benchmark_qnl_advanced_more.ipynb`, which needs a
live `QiskitRuntimeService` and so cannot be re-run at will. The suite exists to make that same
shape of workload runnable offline, at several sizes, forever.

### The suite

| case | topology | qubits | edges | edges/qubit | layers |
|------|----------|-------:|------:|------------:|-------:|
| `hex32`  | heavy-hex | 32  | 33  | 1.03 | 3 |
| `hex64`  | heavy-hex | 64  | 68  | 1.06 | 3 |
| `hex128` | heavy-hex | 128 | 142 | 1.11 | 3 |
| `hex256` | heavy-hex | 256 | 291 | 1.14 | 3 |
| `grid32`  | square grid | 32  | 52  | 1.62 | 4 |
| `grid64`  | square grid | 64  | 112 | 1.75 | 4 |
| `grid128` | square grid | 128 | 232 | 1.81 | 4 |
| `grid256` | square grid | 256 | 480 | 1.88 | 4 |

Both families hit the four sizes **exactly**, so a `hex`/`grid` pair at the same size differs only
in coupling density. That is the comparison the suite is built for: the number of 2-local
generators — and therefore the design matrix's column count — scales with edges, not qubits, so
the grid family carries roughly 1.7x the parameters per qubit and one extra layer gate.

### The topologies

* **`heavy_hex`** — the IBM "brick" lattice used by Eagle and Heron. Degree ≤ 3, 3 edge colours.
  `heavy_hex_lattice(8, 16)` reproduces `FakeFez`'s coupling map **exactly** (176 edges, zero
  symmetric difference); `--self-check` asserts this whenever `qiskit-ibm-runtime` is installed.
  Sizes that are not a whole number of bands are cut down by breadth-first truncation from qubit
  0, which leaves a full lattice plus a ragged partial band — the same shape a real device subset
  has.
* **`grid`** — a square lattice, as on Nighthawk-class devices (`ibm_miami` is exactly a 12x10
  grid). Degree ≤ 4, 4 edge colours. All four sizes factor exactly (4x8, 8x8, 8x16, 16x16).

Both are generated analytically, so a case is byte-identical on any machine with the same library
versions.

### Measured baseline

Recorded 2026-08-26 on an Apple M3 Pro (12 cores, 36 GiB), one case per process, `--quiet`.
Peak dgemm throughput on that machine is **345 GFLOP/s** double precision, which is the number to
compare the linear-algebra stages against.

| case | seconds | peak RSS | generators | design matrix | density | paths | seqs |
|------|--------:|---------:|-----------:|---------------|--------:|------:|-----:|
| `hex32`   |   1.4 |  306 MiB |  1243 | 2324 x 1243   | 2.2e-02 |  1211 | 48 |
| `grid32`  |   3.5 |  792 MiB |  2320 | 4440 x 2320   | 2.0e-02 |  2288 | 75 |
| `hex64`   |   3.7 | 1088 MiB |  2540 | 4752 x 2540   | 1.5e-02 |  2476 | 49 |
| `grid64`  |  13.2 | 2369 MiB |  4928 | 9440 x 4928   | 1.0e-02 |  4864 | 78 |
| `hex128`  |  15.4 | 3114 MiB |  5242 | 9816 x 5242   | 7.4e-03 |  5114 | 49 |
| `grid128` |  72.4 | 8689 MiB | 10144 | 19440 x 10144 | 5.1e-03 | 10016 | 78 |
| `hex256`  |  91.9 | 7813 MiB | 10673 | 19996 x 10673 | 3.8e-03 | 10417 | 49 |
| `grid256` | 600.7 |10325 MiB | 20864 | 40000 x 20864 | 2.6e-03 | 20608 | 78 |

`grid256` is the practical ceiling on a 36 GiB machine: 10 minutes, 10 GiB resident, and a **40.7
GiB peak memory footprint** (it did not swap, but it has no headroom). Almost all of that footprint
is two dense intermediates — see below.

`hex128` is the case to compare against history: it measures 15.4 s with 1662 generators per layer
gate and 5114 paths, against 15.4 s / 1659 / 5104 for the real 127-qubit device the notebook used.
The synthetic suite reproduces the real workload.

Where the time goes, as a share of each build:

| | 32q | 64q | 128q | 256q |
|---|---:|---:|---:|---:|
| path generation (`mult`+`add`/`*`/`paths`) | 63% / 53% | 47% / 32% | 26% / 13% | 10% / 4% |
| `overall/design_matrix` + `overall/rank_reduce` | 15% / 27% | 28% / 48% | 46% / 70% | 64% / 85% |

(heavy-hex / grid in each cell.) The regime shifts with size: below ~64 qubits the build is
pure-Python Pauli bookkeeping, above ~128 it is two dense BLAS calls. Denser topology moves a case
into the second regime sooner, because the generator count — and so the design matrix — scales with
edges.

## What gets measured

Each build is recorded two ways at once.

**Stage view.** The pipeline's stages are invoked one at a time rather than composed with `+`, so
each can be timed. `ExperimentBuilder._run` is itself just a loop over `stage.run`, so this changes
nothing about the work done. Labels are `phase/subject/step`, and the report collapses the subject:
`mult/*/paths` is the total over all layer gates.

Two deliberate splits:

* `experiment.design_matrix` is forced under its own timer before each `RankReducePaths`. The
  property is lazily cached, so this separates *building* the design matrix from the *Gram-Schmidt
  elimination* that consumes it, and the stage then finds the matrix already built. Without this,
  `RankReducePaths` looks like one undifferentiated cost — and in the 127-qubit profile it was 49%
  of the build, so knowing which half is which matters.
* `IndexedMatrix.rank` is never computed unless `--rank` is passed. See the table above.

**Component view.** `qnl_bench/instrument.py` monkeypatches five hot methods —
`PauliLindbladModel.rows`, `LogPathMap.rows`, `IndexedMatrix.add_rows`,
`IndexedMatrix._matmul_matrix`, `IndexedMatrix.linearly_independent_rows` — for the duration of the
build. Nested instrumented calls are subtracted from their caller's self-time, so the self-times
sum without double counting and `add_rows` called from inside `rows` is charged only to `add_rows`.
The gap between that sum and the wall total is the *unattributed* line: pure-Python path and
sequence bookkeeping, which is exactly the part of the build no prior profile had broken down.

Every call is also logged individually with its own self-time and operand shapes, which is how the
report can name the handful of specific calls that dominate a build.

## Fingerprints

Every run digests its own output — path list (ordered and as a set), instruction sequences,
relations — so an optimization can be checked for having left the answer alone. Compare the
`fingerprint` block across runs; `paths` differing while `path_set` matches means a reordering
rather than a change of content.

`qnl_bench/fingerprint.py` serializes content explicitly and **raises** on any type it has not been
taught. This is not fussiness:

* `Path.__hash__` and `FidelityIndex.__hash__` go through `hash()` of strings, which Python
  randomizes per process, so they cannot be compared across runs.
* `QubitSparsePauli.__repr__` is not reliably content-based, so a digest that falls back on `repr`
  can silently be a digest of a memory address — which looks like a working fingerprint while
  guaranteeing every comparison fails.

A fingerprint that refuses to cover something is much better than one that covers an address.

The design matrix digest is deliberately blind to row and column permutation (it uses shape,
nonzero count, and sorted per-row/per-column nonzero counts and absolute sums), because insertion
order is an implementation detail an optimization may legitimately change while the linear algebra
may not.

## Layout

```
benchmarks/
  run_bench.py            CLI: --self-check, --list, case names, --json
  qnl_bench/
    lattices.py           heavy-hex and grid generators, edge colouring
    cases.py              BenchmarkCase, SUITE, gate-set and model construction
    build.py              timed_build: the instrumented pipeline
    instrument.py         stage timeline and monkeypatched component timers
    fingerprint.py        process-independent digests of build output
```
