# qiskit-noise-learning — tiered-review checklist

Package-specific checklist and reference anchors for the `tiered-review` skill.
Each section corresponds to one tier. Anchors are files that *exemplify* the
pattern — read them live to ground findings in the current code, since this file
can drift. Keep this file to **stable, durable** conventions; do not encode
in-flight refactor decisions here (the skill reviews cold).

Domain in one line: randomization-based quantum noise characterization on Qiskit —
run twirled circuit sequences at varying depths, fit exponential fidelity decays,
and solve a linear system to recover per-gate Pauli-Lindblad noise rates.

Two public API tiers:
- **Low-level**: user composes core objects directly (gate sets, fidelity models,
  paths, sequences, experiment-builder stages, analysis stages).
- **High-level**: `NoiseLearner(backend, options).run(instructions)` → a
  `NoiseLearnerJob` whose `NoiseLearnerResult.to_dict()` yields
  `{gate_name: PauliLindbladMap}`.

---

## Tier 1 — design (approach & intent)

Highest altitude. Judge strategy, not code. No bug/style findings.

- **Is the method sound?** Fidelity-decay estimation, twirling/randomization, and
  linear recovery of Pauli-Lindblad rates — does the change respect that pipeline's
  assumptions?
- **Does it belong here?** Distinguish work that belongs upstream in Qiskit
  (`quantum_info`, transpiler) or in `samplomatic` (twirling/injection) from work
  that belongs in this package. Prefer reusing upstream primitives.
- **Does it duplicate an existing abstraction?** Check before adding new concepts:
  `GateSet` / `ModelGateSet`, `FidelityModel`, `Path` / `BaseSequence`,
  `ExperimentBuilderStage`, `AnalysisStage`, `IndexedMatrix`. A "new" idea is often
  a new stage or a new model subclass, not a new framework.
- **Which API tier does it target**, and is that right? Low-level composition vs.
  the `NoiseLearner` high-level wrapper. New user-facing capability usually needs
  both a composable primitive and (optionally) a hook in the high-level pipeline.
- **Is there a simpler strategy** that reuses the existing stage pipelines rather
  than adding a parallel mechanism?

Anchors: `noise_learner/noise_learner.py`, `models/fidelity_model.py`,
`gate_sets/gate_set.py`, `sequences/path.py`, `README`.

---

## Tier 2 — architecture (structure & API)

Assume the approach is settled. Judge structure, interfaces, placement.

- **Stage-pipeline conformance.** The two parallel stage designs share a template:
  public `run()` validates + copies, abstract `_run()` does the work, `__add__`
  builds a flattening composite.
  - `ExperimentBuilderStage`: declares `required_fields` / `populates_fields`; work
    goes in `_run()`. Anchor: `experiment_builder/experiment_builder_stage.py`.
  - `AnalysisStage`: declares `input_level` / `output_level` (data-level *types*);
    pipeline validates level connectivity. Anchor:
    `analysis/analysis_pipeline.py`.
- **ABC / Generic contracts.** Subclasses must satisfy the base:
  - `FidelityModel(Generic[ParameterIndex], ABC)` → implement `row_from_fidelity`.
    Anchor: `models/fidelity_model.py` (see `PauliLindbladModel`).
  - `GateSet(Mapping[str, GateType])` → implement `model_gate_set`. Anchor:
    `gate_sets/gate_set.py`.
- **Hashing is load-bearing.** Objects used as array labels / dict keys need
  correct, consistent `__eq__` + `__hash__` (and follow the lazy `_hash` cache
  pattern where the type already uses it). Anchors: `sequences/fidelity_index.py`,
  `sequences/path.py`, `math/indexed_matrix.py`.
- **Serialization convention.** Data containers → xarray (`Dataset`/`DataTree`),
  constructed via `from_arrays` classmethods; pydantic is used **only** for
  `LearningOptions`. Don't introduce a new serialization scheme casually. Anchors:
  `data/raw_data.py`, `noise_learner/learning_options.py`.
- **Qiskit-mirroring API.** Execution returns a job object with `.result()` (mirrors
  Qiskit primitives); results export native `qiskit.quantum_info.PauliLindbladMap`.
  Keep public shapes consistent with Qiskit conventions.
- **Encapsulation conventions.** Attributes as `self._x` behind read-only
  `@property`; keyword-only args (`*`) in high-level containers (`Experiment`,
  `Fit`); `replace()`-style validated updates for immutable-ish containers.

---

## Tier 3 — correctness (bugs; delegates to code-review)

Assume design + architecture are settled. Prime code-review with these invariants.

Domain invariants to guard:
- **Fidelity range:** `f > 0` before `-log(f)`; existing guards clamp with
  `max(fidelity, 1e-300)` and `np.clip(..., 1e-10, 1-1e-10)`. New fidelity math must
  preserve them.
- **Positivity of rates:** Pauli-Lindblad rates `r ≥ 0` (enforced by `NNLSSolve`,
  bounded `LSQLinearSolve`, or the `PositivityMinSolve` convex program).
  `free_indices` (params off the constraint boundary) drive which get covariance —
  a subtle correctness point in error propagation.
- **Design-matrix construction:** coefficient `2` per anticommuting generator;
  depth-scaling differs for bound vs. unbound paths; `IndexedMatrix.add_rows`
  **drops all-zero rows**, so `b` / `sigma_b` must stay aligned to the surviving
  `row_index_map`.
- **Curve-fit statistics:** single-sample std uses binomial `sqrt(p(1-p))` with
  `p=(mean+1)/2`; multi-sample uses standard error `std(ddof=1)/sqrt(n)`;
  `absolute_sigma=True`; deliberate NaN handling, sigma clipping, and a
  `RuntimeError` fallback path.
- **Qubit indexing & dim contracts:** `qubit_subset ⊆ range(num_qubits)`; gates must
  act within the subset. `RawData` dims are `("randomization","shot","bit")` with a
  `("randomization","shot")` mask for ragged shots; `measurement_map` uses
  **physical** qubit indices. `IndexedMatrix` index-maps must be a permutation of
  `range(shape)`.
- **`None` as legitimate failure vs. raise:** some helpers return `None` to mean
  "cannot" (e.g. `extend_permutations`) rather than raising — don't "fix" these into
  exceptions.
- **Unsupported physics** raises `NotImplementedError` (e.g. midcircuit
  measurement) — keep it that way rather than silently mishandling.

Numeric hotspots (read before reviewing): `analysis/model_solve.py`,
`analysis/curve_fit_observables.py`, `models/pauli_lindblad_model.py`,
`math/indexed_matrix.py`, `sequences/partial_pauli_permutation.py`,
`sequences/path.py`.

---

## Tier 4 — polish (conventions; delegates to code-review)

Assume everything above is settled. CLAUDE.md rules made executable, plus de-facto
conventions.

- **Copyright header** `(C) Copyright IBM 2026.` on every non-`docs/` file (a year
  *range* like `2025, 2026.` only when the file predates the current year). See
  `tools/verify_headers.py`.
- **Docstrings:** `__init__` args documented in the **class** docstring; classes/
  functions referenced as prose or Sphinx cross-refs (`:class:`~.X``), not bare
  backticks in Sphinx-rendered docstrings.
- **`from __future__ import annotations`:** CLAUDE.md says avoid it. There is legacy
  drift (several `experiment_builder/` files, `visualizations/gate_set_topology.py`)
  — new code must **not** add more; flag it if introduced.
- **Errors:** prefer `ValueError` with an f-string naming the offending value/stage;
  `TypeError` for wrong-type guards; `NotImplementedError` for unsupported physics;
  `ImportError` via `optionals.py`. Validation centralized in `_validate_*` free
  helpers.
- **Optional deps** gated through `optionals.py` (`HAS_CVXPY`, `HAS_PLOTLY`):
  `HAS_X.require_now("...")` then a local import inside the method; heavy/optional
  imports behind `TYPE_CHECKING`.
- **Typing:** PEP 604 (`X | None`), `Self`, `Generic`/`TypeVar` (`bound=Hashable`
  for label types), `Literal` for enum-like options, `@overload` for dispatch.
- **Naming:** `PascalCase` classes; `snake_case` module named after its primary
  class; `_leading_underscore` privates; `UPPER_SNAKE` constants; stage classes are
  verb/noun phrases (`ComputeObservables`, `MergeInstructionSequences`).
- **Tests** (`test/unit/`, mirrors the package tree): public-API imports only;
  `conftest.py` fixture hierarchy; `@pytest.mark.parametrize`; `np.isclose`/
  `allclose` with explicit `atol`; test docstrings describe the scenario/expected
  math. **Doctests run in CI** (`--doctest-modules` + scipy-doctest) — every
  docstring example must actually execute.
- **Rich repr:** `_repr_html_` via `utils/html_repr.HTMLTable`; `draw()` for plotly;
  concise `__repr__`.

---

## Reference: subpackage roles

| Subpackage | Role |
|---|---|
| `gate_sets/` | Gate/GateSet abstractions; Qiskit↔model bridge |
| `sequences/` | Domain primitives: FidelityIndex, Path, InstructionSequence, permutations |
| `models/` | Fidelity models (linear parameterization of log-fidelities) |
| `experiment_builder/` | Stage-based pipeline building an `Experiment` |
| `circuit_generator/` | `ExecutorCircuitGenerator` + `ExecutorDataMapper` (executor/data-mapper) |
| `aer_executor/` | Local Aer executor mimicking the runtime `Executor` interface |
| `analysis/` | Stage-based analysis pipeline (raw→model); numerically sensitive |
| `data/` | xarray-backed leveled data containers (`RawData`→`ObservableData`→`AveragedData`→`ModelData`) |
| `math/` | `IndexedMatrix`, `IndexedVector` — arrays with hashable row/col labels |
| `noise_learner/` | High-level `NoiseLearner`, `LearningOptions` (pydantic), result, job |
| `visualizations/` | plotly topology + latex label helpers (optional dep) |
| `utils/` | `html_repr` (HTMLTable for `_repr_html_`) |
