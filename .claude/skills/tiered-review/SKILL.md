---
name: tiered-review
description: >-
  Perform a tiered review of the current branch as a PR, analyzing changes against main.
  Runs ONE review tier per invocation — design → architecture → correctness → polish —
  each assuming the previous tiers are already settled. Invoke as
  `/tiered-review <tier> [scope] [effort]` (e.g. `/tiered-review design`,
  `/tiered-review correctness new-models high`).
disable-model-invocation: false
---

# Tiered review

A review workflow proceeding in four tiers, one per invocation. Each tier is a fresh run (it
re-diffs the current tree), and each tier **assumes the tiers above it are already settled**
and refuses to drift downward.

```
Tier 1  design       — approach & intent            (chat only)
Tier 2  architecture — structure & API              (chat only)
Tier 3  correctness  — bugs        (delegates to the built-in code-review skill)
Tier 4  polish       — conventions (delegates to the built-in code-review skill)
```

The user controls the transitions between tiers: they invoke a tier, resolve what
it surfaces on their own time, then invoke the next tier. Do **not** run multiple
tiers in one invocation — do exactly the one requested and recommend the next.

## What this package is

Toolkit for randomization-based quantum noise characterization within the Qiskit ecosystem:
building and running learning experiments, analyzing the data, and solving for model parameters.

Two public API tiers:
- **Low-level**: User composes core objects directly, useful for research applications.
- **High-level**: Preset configurations of low-level workflows accessible through the
  `NoiseLearner` interface.

## Arguments

`/tiered-review <tier> [scope] [effort]`

- **tier** (required): one of `design`, `architecture`, `correctness`, `polish`.
  Accept obvious synonyms (`arch`→architecture, `bugs`→correctness,
  `nits`/`style`→polish). If no tier is given, default to `design` and say so.
- **scope** (optional): a git ref or diff range. Default `main...HEAD` (merge-base
  diff of the current branch against `main` — only what this branch changed). A
  bare branch name `X` means `main...X`. Accept an explicit range verbatim.
- **effort** (optional): `low` | `medium` | `high` | `xhigh` | `max`. Default
  `high`. Only meaningful for tiers 3–4 (passed through to `code-review`); for
  tiers 1–2 it scales how deep you trace.

## Step 0 — set up (every tier)

Do this first, regardless of tier:

1. **Resolve scope.** Compute the diff range (default `main...HEAD`). Run
   `git diff --stat <range>` and `git diff <range>` to get the change. If the diff
   is empty, say so and stop.
2. **Frame intent.** In 2–4 sentences, state what the change is *trying to
   accomplish*, inferred from the diff, commit messages (`git log main..HEAD
   --oneline`), and any touched docstrings. Every tier judges the change against
   this intent.
3. **Load context.** Read the root `CLAUDE.md`, then work through the section for the
   requested tier under [Tier logic](#tier-logic) below.
4. **Review cold.** Judge the diff on its own merits. Do not make assumptions based on other
   branches or external context unless the diff, CLAUDE.md, or the user says so.

## The escalation contract

Each tier stays in its lane. This is what keeps the phases from collapsing into
one undifferentiated review:

- **A tier assumes every higher tier is settled.** `correctness` does not
  relitigate the API shape; `polish` does not raise design concerns.
- **A tier refuses to drift downward.** If while doing `design` you notice a bug,
  do **not** report it as a design finding — note in one line that lower tiers
  exist for it and move on. The `design` tier explicitly does **not** flag bugs,
  style, or nitpicks.
- If you believe a *higher* tier was mis-resolved (e.g. during `correctness` you
  realize the whole approach is wrong), that IS worth raising — surface it briefly
  and suggest re-running the higher tier, rather than reviewing downward on a
  foundation you doubt.

## Tier logic

For **all** tiers: produce a **ranked** list (most important first). Each finding
gets a one-line summary, a `file:line` anchor, a short rationale, and — where
useful — a concrete alternative or fix. Rank by impact on the change's goal, not
by how easy it is to describe. If a tier finds nothing, say so plainly.

### Tier 1 — `design` (chat only)
Judge strategy and design choices harshly, but do not comment on code details.

- **Do the changes make conceptual sense?** Does it fit into the design of the package, or introduce
  new abstractions that are useful without introducing unnecessary complexity? Is there a simpler
  approach? Does it duplicate any existing abstractions?
- **Does it belong here?** Distinguish work that belongs upstream in Qiskit
  (`quantum_info`, transpiler) or in `samplomatic` (twirling/injection) from work
  that belongs in this package. Prefer reusing upstream primitives.

Report to chat. Do not edit files.

Anchors: `noise_learner/noise_learner.py`, `models/fidelity_model.py`,
`gate_sets/gate_set.py`, `sequences/path.py`, `README`.

### Tier 2 — `architecture` (chat only)
Concrete structure and public surface, **assuming design is settled**. Judge
structure, interfaces, and placement: ABC/Generic contracts, hashing correctness
for label/key objects, serialization convention, Qiskit-mirroring API shape,
placement of responsibility, naming of public surfaces, backward-compat. Report to
chat. Do not edit files.

### Tier 3 — `correctness` (delegates to code-review)
Bugs and logic errors, **assuming design and architecture are settled**. This
includes **test coverage of the change itself**: new or changed behavior, branches,
and edge cases must be exercised — an untested path is a latent correctness gap, not
a style nit. (Whether the *existing* tests follow project conventions is Tier 4.)
Delegate the heavy lifting to the built-in review skill.

### Tier 4 — `polish` (delegates to code-review)
Conventions, nits, docs, and micro-cleanups, **assuming everything above is settled**
(CLAUDE.md rules made executable, plus de-facto conventions). Invoke the built-in
**code-review** skill steered toward convention/simplification findings against the
checklist below, and away from design/architecture/correctness (those are other
tiers). Present ranked findings; offer `--fix`.

Do **not** spend findings on anything `ruff` + pre-commit already enforce — they
fail CI on their own: copyright headers (`verify_headers.py`), formatting and
line-length, import order (isort), `Optional`/`Union` → PEP 604, private-member
access (SLF), and trailing-whitespace/EOF. The checklist below is deliberately the
*residue* those tools cannot catch.

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
- **Typing:** `Self`, `Generic`/`TypeVar` (`bound=Hashable` for label types),
  `Literal` for enum-like options, `@overload` for dispatch.
- **Naming:** `PascalCase` classes; `snake_case` module named after its primary
  class; `_leading_underscore` privates; `UPPER_SNAKE` constants; stage classes are
  verb/noun phrases (`ComputeObservables`, `MergeInstructionSequences`).
- **Test-writing conventions** (`test/unit/`, mirrors the package tree) — *how* the
  tests that exist are written, not *whether* the change is covered (that is Tier 3):
  public-API imports only; `conftest.py` fixture hierarchy; `@pytest.mark.parametrize`;
  `np.isclose`/`allclose` with explicit `atol`; test docstrings describe the
  scenario/expected math.
- **Rich repr:** `_repr_html_` via `utils/html_repr.HTMLTable`; `draw()` for plotly;
  concise `__repr__`.

## Closing every run

End with a **next-tier recommendation**: name the next tier and the exact command
(e.g. "Design looks settled — when ready, run `/tiered-review architecture`").
If the current tier surfaced blocking concerns, recommend resolving them and
re-running the *current* tier rather than advancing.

## Reference: subpackage roles

| Subpackage | Role |
|---|---|
| `gate_sets/` | Gate/GateSet abstractions; Bridge between qiskit instructions and this package. |
| `sequences/` | Domain primitives: representation of fidelities, sequences of fidelities, and sequences of instructions. |
| `math/` | Basis-labelled representations of linear algebra objects. |
| `models/` | Fidelity "models": specific linear mappings from arbitrary model parameters into log fidelity space. |
| `experiment_builder/` | Stage-based pipeline for building a description of an experiment: both sequences of instructions as well as what can be learned about a model from their data. |
| `circuit_generator/` | Management of internal experiment representation to job submission inputs, and job result data into internal data representations (another bridge between this package and qiskit). |
| `analysis/` | Stage-based data analysis tools, ultimately generating estimates of model parameters. |
| `data/` | Data containers the analysis tools act on. |
| `noise_learner/` | High-level interface providing curated configurations of the other subpackages. |
| `visualizations/` | Visualization tools. |
| `utils/` | Miscellaneous helpers that don't belong to a specific subpackage (e.g. `html_repr.HTMLTable` for `_repr_html_`). |
