---
name: tiered-review
description: >-
  Gated, escalating code review for qiskit-noise-learning. Runs ONE review tier
  per invocation — design → architecture → correctness → polish — each assuming
  the higher tiers are already settled. Use when reviewing a branch's changes and
  you want to resolve high-level concerns before drilling into detail. Invoke as
  `/tiered-review <tier> [scope] [effort]` (e.g. `/tiered-review design`,
  `/tiered-review correctness new-models high`).
disable-model-invocation: false
---

# Tiered review

A **gated, escalating** review workflow. Instead of one review that mixes design
critique with bug-hunting and nitpicks, this runs in four tiers, one per
invocation. Each tier is a fresh run (it re-diffs the current tree), and each tier
**assumes the tiers above it are already settled** and refuses to drift downward.

```
Tier 1  design       — approach & intent            (chat only)
Tier 2  architecture — structure & API              (chat only)
Tier 3  correctness  — bugs        (delegates to the built-in code-review skill)
Tier 4  polish       — conventions (delegates to the built-in code-review skill)
```

The user drives the gates: they invoke a tier, resolve what it surfaces on their
own time, then invoke the next tier. Do **not** run multiple tiers in one
invocation — do exactly the one requested and recommend the next.

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
3. **Load the project checklist.** Read `conventions.md` next to this file — it
   holds the `qiskit-noise-learning`-specific checklist and reference anchors for
   each tier. Use the section for the requested tier.
4. **Re-derive live anchors** (don't trust stale notes): read the root
   `CLAUDE.md`, and for the tier at hand skim the anchor files named in
   `conventions.md` so your findings cite what the code *currently* looks like.
5. **Review cold.** Judge the diff on its own merits. Do not assume any in-flight
   refactor is "already decided" unless the diff or CLAUDE.md says so.

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
Highest altitude, **zero code-detail findings**. Work through the `design` section
of `conventions.md`. Focus: is this the right problem framing and solution
strategy; does it belong in this package; does it duplicate an existing
abstraction; which API tier does it target and is that right; is there a
fundamentally simpler approach. Report to chat. Do not edit files.

### Tier 2 — `architecture` (chat only)
Concrete structure and public surface, **assuming the approach is settled**. Work
through the `architecture` section of `conventions.md`: stage-pipeline
conformance, ABC/Generic contracts, hashing correctness for label/key objects,
serialization convention, Qiskit-mirroring API shape, placement of
responsibility, naming of public surfaces, backward-compat. Report to chat. Do
not edit files.

### Tier 3 — `correctness` (delegates to code-review)
Bugs and logic errors, **assuming design and architecture are settled**. Delegate
the heavy lifting to the built-in review skill, primed with domain context:

1. Read the `correctness` section of `conventions.md` (domain invariants +
   numeric hotspots) so you know what to watch for.
2. Invoke the built-in **code-review** skill over the same scope at the resolved
   effort, and explicitly steer it toward those invariants and hotspots and away
   from design/architecture/style (those are other tiers).
3. Present the confirmed findings, ranked. Offer `--fix` as a follow-up; only
   apply fixes if the user asks.

### Tier 4 — `polish` (delegates to code-review)
Conventions, nits, docs, and micro-cleanups, **assuming everything above is
settled**. Read the `polish` section of `conventions.md`, then invoke the
built-in **code-review** skill steered toward convention/simplification findings
(CLAUDE.md adherence, `from __future__` drift, error-raising style, optional-dep
gating, test/doctest conventions). Present ranked findings; offer `--fix`.

## Closing every run

End with a **next-tier recommendation**: name the next tier and the exact command
(e.g. "Design looks settled — when ready, run `/tiered-review architecture`").
If the current tier surfaced blocking concerns, recommend resolving them and
re-running the *current* tier rather than advancing.
