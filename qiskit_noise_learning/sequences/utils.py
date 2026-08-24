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

"""Utilities for working with sequences."""

from collections import defaultdict
from collections.abc import Hashable, Sequence
from typing import Literal, get_args

import numpy as np

from .apply_gate import ApplyGate
from .instruction_sequence import InstructionSequence
from .partial_pauli_permutation import (
    NUM_COMPLETE_PERMUTATIONS,
    PartialPauliPermutation,
    consistency_matrix,
)

_POPCOUNT = np.array([bin(value).count("1") for value in range(256)], dtype=np.uint8)
"""The number of set bits in each possible value of a ``uint8``."""

InstructionSequenceOrder = Literal[
    "most-constrained-first", "least-constrained-first", "qubitwise-lexicographic", "input"
]
"""The order in which instruction sequences are offered to the groups by :func:`merge_groups`."""

MergingStrategy = Literal["first", "most-constrained", "least-impacted"]
"""Which of the groups a sequence can join :func:`merge_groups` merges it into."""


def _merge_candidate_data(sequence: InstructionSequence) -> tuple[Hashable, np.ndarray]:
    """Return a key and a concatenation of the sequence's partial permutation indices.

    The returned key comes with the following guarantees:
    - If the keys for two instruction sequences are different, then they are not mergeable.
    - If the keys for two instruction sequences are equal, the concatenation of the partial
      permutation indices has the same length, and whether or not they are mergeable is determined
      completely by the consistency of those partial permutation indices.

    Args:
        sequence: The sequence to summarize.

    Returns:
        The merge candidate key, and the concatenated per-qubit indices of the sequence's partial
        permutations, which is empty if the sequence contains none.

    Raises:
        TypeError: If the sequence contains an instruction that is neither a gate application nor a
            partial Pauli permutation.
    """
    key = [sequence.fragment_depth]
    indices = []
    for fragment in (
        sequence.start_fragment,
        sequence.repeatable_fragment,
        sequence.end_fragment,
    ):
        tokens = []
        for instr in fragment:
            if isinstance(instr, ApplyGate):
                tokens.append(("gate", instr.gate_name))
            elif isinstance(instr, PartialPauliPermutation):
                tokens.append(("permutation", instr.num_qubits))
                indices.append(instr.partial_permutation_indices)
            else:
                raise TypeError(
                    f"Cannot partition instruction sequences containing "
                    f"{type(instr).__name__} instructions: only gate applications and partial "
                    f"Pauli permutations are supported."
                )
        key.append(tuple(tokens))

    return tuple(key), np.concatenate(indices) if indices else np.zeros(0, dtype=np.int8)


def _completion_masks() -> np.ndarray:
    """Return a bitmask of the complete permutations each partial permutation is consistent with.

    Returns:
        A 1d ``uint8`` array whose ``[idx]`` entry has bit ``c`` set if and only if the partial
        permutation ``idx`` is consistent with the complete permutation ``c``.
    """
    consistency = consistency_matrix()[:NUM_COMPLETE_PERMUTATIONS]
    bits = (1 << np.arange(NUM_COMPLETE_PERMUTATIONS, dtype=np.uint8))[:, None]
    return (consistency * bits).sum(axis=0).astype(np.uint8)


def _row_order(masks: np.ndarray, order: InstructionSequenceOrder) -> np.ndarray:
    """Return the order in which rows are offered to the groups.

    Args:
        masks: A ``uint8`` array whose ``[i, j]`` entry is a bitmask of the complete permutations
            that row ``i`` is consistent with in column ``j``.
        order: The ordering to apply, as documented in ``merge_groups``.

    Returns:
        The row indices, in the order they are to be visited.
    """
    if order == "input":
        return np.arange(len(masks))

    if order == "qubitwise-lexicographic":
        # lexsort needs at least one key, and with no columns the rows are interchangeable anyway
        return np.lexsort(masks.T[::-1]) if masks.shape[1] else np.arange(len(masks))

    # the total number of complete permutations each row admits, negated to sort descending; ties
    # are broken by row index so that every ordering is a deterministic function of the masks
    num_admissible = _POPCOUNT[masks].sum(axis=1)
    if order == "least-constrained-first":
        num_admissible = -num_admissible
    return np.argsort(num_admissible, kind="stable")


def _group_by_witness(
    masks: np.ndarray, order: np.ndarray, strategy: MergingStrategy
) -> list[list[int]]:
    """Greedily group rows that admit a common complete permutation in every column.

    Rows are visited in ``order`` and each is placed in one of the groups built so far, or opens a
    new group if it fits in none of them.

    Args:
        masks: A ``uint8`` array whose ``[i, j]`` entry is a bitmask of the complete permutations
            that row ``i`` is consistent with in column ``j``.
        order: The order in which to visit the rows, as returned by ``_row_order``.
        strategy: How to choose among the groups a row can join, as documented in
            ``merge_groups``.

    Returns:
        The groups of row indices.
    """
    # the bitmask of complete permutations still admissible in each column, one row per group
    admissible_completions = np.zeros((0, masks.shape[1]), dtype=np.uint8)
    groups: list[list[int]] = []
    for row_idx in order:
        row = masks[row_idx]
        # a row may join a group only if every column retains a complete permutation common to both
        joined = admissible_completions & row
        feasible = np.flatnonzero(joined.all(axis=1))

        if not len(feasible):
            admissible_completions = np.vstack([admissible_completions, row])
            groups.append([int(row_idx)])
            continue

        if strategy == "first":
            pick = feasible[0]
        else:
            before = _POPCOUNT[admissible_completions[feasible]].sum(axis=1, dtype=np.int32)
            key = before
            if strategy == "least-impacted":
                key = before - _POPCOUNT[joined[feasible]].sum(axis=1, dtype=np.int32)
            pick = feasible[np.argmin(key)]

        admissible_completions[pick] &= row
        groups[pick].append(int(row_idx))

    return groups


def merge_groups(
    sequences: Sequence[InstructionSequence],
    instruction_sequence_order: InstructionSequenceOrder = "most-constrained-first",
    merging_strategy: MergingStrategy = "least-impacted",
) -> list[list[int]]:
    r"""Group the positions of instruction sequences that can be merged with each other.

    The returned groups partition the positions of ``sequences``: every position appears in exactly
    one group, and the sequences within a group can all be merged together, in any order, into a
    single sequence via :meth:`~.InstructionSequence.merge`.

    In terms of strategy, the list of sequences is first partitioned into sets such that members
    from disjoint sets cannot be merged. This is done based on the fragment structure: if
    corresponding fragments in two instruction sequences do not have the same sequence of gate
    applications and partial permutations, then they cannot be merged.

    Each set is then further partitioned into the returned groups via a family of greedy algorithms.
    The instruction sequences are ordered according to ``instruction_sequence_order``, and the
    algorithm iterates over them, merging them into the group according to the strategy in
    ``merging_strategy``.

    A sequence is called more constrained if it specifies more Pauli mappings.
    ``instruction_sequence_order`` accepts:

    * ``"most-constrained-first"``: ordered from most-constrained to least-constrained.
    * ``"least-constrained-first"``: ordered from least-constrained to most-constrained.
    * ``"qubitwise-lexicographic"``: ordered lexicographically by the bit string indicating which
      complete permutations implement each partial permutation. E.g. a bit string ``001010``
      indicates that the complete permutations at indices ``2`` and ``4`` are valid completions.
      This is a technical condition based on the specific ordering used in
      ``COMPLETE_TO_C1_TABLEAU``, but it is one of many possible choices for a linear ordering of
      the instruction sequences based on similarity of completions.
    * ``"input"``: the order in which they were given.

    ``merging_strategy`` selects which of the groups a sequence can join it is merged into:

    * ``"first"``: the group created earliest.
    * ``"most-constrained"``: the group that already admits the fewest complete permutations, which
      leaves the more flexible groups intact for later sequences.
    * ``"least-impacted"``: the group for which the instruction sequence will reduce the number of
      completions the least.

    Args:
        sequences: The instruction sequences to group.
        instruction_sequence_order: The order in which to consider the sequences.
        merging_strategy: How to choose among the groups a sequence can join.

    Returns:
        The groups of positions of mergeable sequences.

    Raises:
        TypeError: If any sequence contains an instruction that is neither a gate application nor a
            partial Pauli permutation.
        ValueError: If ``instruction_sequence_order`` or ``merging_strategy`` is not one of the
            documented values.
    """
    if instruction_sequence_order not in get_args(InstructionSequenceOrder):
        raise ValueError(
            f"Unknown instruction_sequence_order {instruction_sequence_order!r}: expected one of "
            f"{', '.join(map(repr, get_args(InstructionSequenceOrder)))}."
        )
    if merging_strategy not in get_args(MergingStrategy):
        raise ValueError(
            f"Unknown merging_strategy {merging_strategy!r}: expected one of "
            f"{', '.join(map(repr, get_args(MergingStrategy)))}."
        )

    candidates: dict[Hashable, list[int]] = defaultdict(list)
    permutation_indices = []
    for idx, sequence in enumerate(sequences):
        key, indices = _merge_candidate_data(sequence)
        candidates[key].append(idx)
        permutation_indices.append(indices)

    masks = _completion_masks()

    groups = []
    for candidate_indices in candidates.values():
        columns = masks[np.stack([permutation_indices[idx] for idx in candidate_indices])]
        order = _row_order(columns, instruction_sequence_order)
        groups.extend(
            [candidate_indices[row] for row in rows]
            for rows in _group_by_witness(columns, order, merging_strategy)
        )

    return groups
