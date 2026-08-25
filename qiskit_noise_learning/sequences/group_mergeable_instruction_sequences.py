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

"""Function for partitioning a set of instruction sequences into mergeable subsets.

The public facing function is :func:`group_mergeable_instruction_sequences`; implementation details
are described here.

Two instruction sequences with the same instruction structure (same gate applications and
location of partial pauli permutations), are mergeable exactly when, for each partial permutation
on every qubit, there exists a shared completion. While instruction sequences have a public
interface for pairwise checks of mergeability, this module optimizes performing these checks across
a collection.

The general strategy is to track "shared completions" across a set of instruction sequences as a
bit-packed bit string. There are 6 complete permutations, and therefore the set of completions is
represented as 6 bits, packed into a single ``uint8``. ``_completion_masks`` tabulates this, indexed
by partial permutation index. This representation has the following properties:

* the completions shared by a group are the columnwise bitwise AND of its rows;
* the group is mergeable exactly when that AND is nonzero in every column;
* ``_POPCOUNT`` turns a mask back into a count, so summing it along a row measures how constrained
  a sequence, or a group, is.
"""

from collections import defaultdict
from collections.abc import Hashable, Sequence
from typing import Literal, NamedTuple, get_args

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
"""The order in which :func:`group_mergeable_instruction_sequences` considers the sequences."""

MergingStrategy = Literal["first", "most-constrained", "least-impacted"]
"""Which of the groups a sequence can join :func:`group_mergeable_instruction_sequences` picks."""


class GroupingStrategy(NamedTuple):
    """An instruction sequence order paired with a merging strategy, specifying a full strategy.

    Plain two-tuples are accepted anywhere an instance of this class is, so
    ``("input", "first")`` and ``GroupingStrategy("input", "first")`` are interchangeable.

    Args:
        order: The order in which the instruction sequences are considered.
        merging_strategy: Which of the groups a sequence can join it is merged into.
    """

    order: InstructionSequenceOrder
    merging_strategy: MergingStrategy


DEFAULT_GROUPING_STRATEGIES: tuple[GroupingStrategy, ...] = (
    GroupingStrategy("most-constrained-first", "least-impacted"),
    GroupingStrategy("least-constrained-first", "first"),
    GroupingStrategy("qubitwise-lexicographic", "most-constrained"),
)
"""Default strategies for :func:`group_mergeable_instruction_sequences`, empirically determined."""


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
        masks: A ``uint8`` array whose ``[i, j]`` entry is a bitmask of the completions
            available to row ``i`` in column ``j``.
        order: The ordering to apply, as documented in ``group_mergeable_instruction_sequences``.

    Returns:
        The row indices, in the order they are to be visited.
    """
    if order == "input":
        return np.arange(len(masks))

    if order == "qubitwise-lexicographic":
        # lexsort needs at least one key, and with no columns the rows are interchangeable anyway
        return np.lexsort(masks.T[::-1]) if masks.shape[1] else np.arange(len(masks))

    # the total number of completions each row has available, negated to sort descending; ties
    # are broken by row index so that every ordering is a deterministic function of the masks
    num_admissible = _POPCOUNT[masks].sum(axis=1)
    if order == "least-constrained-first":
        num_admissible = -num_admissible
    return np.argsort(num_admissible, kind="stable")


def _validate_grouping_strategies(grouping_strategies: Sequence[GroupingStrategy]) -> None:
    """Check that each grouping strategy pairs a known order with a known merging strategy.

    Args:
        grouping_strategies: The grouping strategies to check.

    Raises:
        ValueError: If no grouping strategies are given, or if any of them is not a pairing of a
            documented instruction sequence order with a documented merging strategy.
    """
    if not grouping_strategies:
        raise ValueError("At least one grouping strategy is required, but none were given.")

    for grouping_strategy in grouping_strategies:
        if len(grouping_strategy) != 2:
            raise ValueError(
                f"Invalid grouping strategy {grouping_strategy!r}: expected an instruction "
                f"sequence order paired with a merging strategy."
            )

        order, merging_strategy = grouping_strategy
        if order not in get_args(InstructionSequenceOrder):
            raise ValueError(
                f"Unknown instruction sequence order {order!r} in grouping strategy "
                f"{grouping_strategy!r}: expected one of "
                f"{', '.join(map(repr, get_args(InstructionSequenceOrder)))}."
            )
        if merging_strategy not in get_args(MergingStrategy):
            raise ValueError(
                f"Unknown merging strategy {merging_strategy!r} in grouping strategy "
                f"{grouping_strategy!r}: expected one of "
                f"{', '.join(map(repr, get_args(MergingStrategy)))}."
            )


def _group_by_shared_completion(
    masks: np.ndarray, order: np.ndarray, strategy: MergingStrategy
) -> list[list[int]]:
    """Greedily group rows sharing a completion in every column.

    Rows are visited in ``order`` and each is placed in one of the groups built so far, or opens a
    new group if it fits in none of them.

    Args:
        masks: A ``uint8`` array whose ``[i, j]`` entry is a bitmask of the completions
            available to row ``i`` in column ``j``.
        order: The order in which to visit the rows, as returned by ``_row_order``.
        strategy: How to choose among the groups a row can join, as documented in
            ``group_mergeable_instruction_sequences``.

    Returns:
        The groups of row indices.
    """
    # the bitmask of completions still available in each column, one row per group
    admissible_completions = np.zeros((0, masks.shape[1]), dtype=np.uint8)
    groups: list[list[int]] = []
    for row_idx in order:
        row = masks[row_idx]
        # a row may join a group only if every column retains a completion common to both
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


def group_mergeable_instruction_sequences(
    sequences: Sequence[InstructionSequence],
    grouping_strategies: Sequence[GroupingStrategy] | None = None,
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
    A single member of this family is specified by a grouping strategy, which pairs an order to
    visit the instruction sequences in with a strategy for choosing which group to merge each one
    into. Every grouping strategy in ``grouping_strategies`` is applied to every set, and the fewest
    groups found for a set are the ones returned for it, ties going to the earlier strategy. Because
    the sets are treated independently, supplying an additional grouping strategy can only decrease
    the total number of groups returned.

    A sequence is called more constrained if it specifies more Pauli mappings. The first entry of a
    grouping strategy, the order to visit the instruction sequences in, is one of:

    * ``"most-constrained-first"``: ordered from most-constrained to least-constrained.
    * ``"least-constrained-first"``: ordered from least-constrained to most-constrained.
    * ``"qubitwise-lexicographic"``: ordered so that sequences agreeing on the Pauli mappings they
      specify, qubit by qubit from the first onwards, are considered consecutively. Which of two
      differing sequences comes first follows a fixed but arbitrary convention, making this one of
      many possible orderings that place sequences specifying similar mappings near each other.
    * ``"input"``: the order in which they were given.

    The second entry selects which of the groups a sequence can join it is merged into:

    * ``"first"``: the group created earliest.
    * ``"most-constrained"``: the group that already admits the fewest complete permutations, which
      leaves the more flexible groups intact for later sequences.
    * ``"least-impacted"``: the group whose admissible complete permutations the instruction
      sequence rules out the fewest of.

    Args:
        sequences: The instruction sequences to group.
        grouping_strategies: The grouping strategies to take the fewest groups found by any of. If
            ``None``, an empirically determined default set of strategies is used.

    Returns:
        The groups of positions of mergeable sequences.

    Raises:
        TypeError: If any sequence contains an instruction that is neither a gate application nor a
            partial Pauli permutation.
        ValueError: If ``grouping_strategies`` is empty, or if any of its entries is not a pairing
            of a documented instruction sequence order with a documented merging strategy.
    """
    if grouping_strategies is None:
        grouping_strategies = DEFAULT_GROUPING_STRATEGIES
    _validate_grouping_strategies(grouping_strategies)

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
        # every strategy is applied to each set of merge candidates separately
        fewest = min(
            (
                _group_by_shared_completion(columns, _row_order(columns, order), merging_strategy)
                for order, merging_strategy in grouping_strategies
            ),
            key=len,
        )
        groups.extend([candidate_indices[row] for row in rows] for rows in fewest)

    return groups
