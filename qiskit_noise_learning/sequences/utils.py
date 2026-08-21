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


def _merge_candidate_data(sequence: InstructionSequence) -> tuple[Hashable, np.ndarray]:
    """Return a key and a concatenation of the sequences permutation indices.

    The returned key comes with the following guarantees:
    - If the keys for two instruction sequences are different, then they are not mergeable.
    - If the keys for two instruction sequences are equal, the concatenation of the permutation
      indices has the same length, and whether or not they are mergeable is determined completely by
      the consistency of the permutation indices.

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


def _group_by_witness(masks: np.ndarray) -> list[list[int]]:
    """Greedily group rows that admit a common complete permutation in every column.

    Args:
        masks: A ``uint8`` array whose ``[i, j]`` entry is a bitmask of the complete permutations
            that row ``i`` is consistent with in column ``j``.

    Returns:
        The groups of row indices.
    """
    # seeding each group with the least constrained row available yields substantially fewer groups
    # than taking the rows in order
    freedom = _POPCOUNT[masks].sum(axis=1)
    remaining = np.argsort(-freedom, kind="stable")

    covered = np.zeros(len(masks), dtype=np.bool_)
    groups = []
    while len(remaining):
        group = [int(remaining[0])]
        admissible = masks[remaining[0]].copy()
        candidates = remaining[1:]
        while len(candidates):
            # a row may join only if every column retains a permutation common to the whole group,
            # and since ``admissible`` only ever shrinks, the rejected rows can never return
            candidates = candidates[(admissible & masks[candidates]).all(axis=1)]
            if not len(candidates):
                break

            admissible &= masks[candidates[0]]
            group.append(int(candidates[0]))
            candidates = candidates[1:]

        covered[group] = True
        remaining = remaining[~covered[remaining]]
        groups.append(group)

    return groups


def merge_groups(sequences: Sequence[InstructionSequence]) -> list[list[int]]:
    r"""Group the positions of instruction sequences that can be merged with each other.

    The returned groups partition the positions of ``sequences``: every position appears in exactly
    one group, and the sequences within a group can all be merged together, in any order, into a
    single sequence via :meth:`~.InstructionSequence.merge`.

    Sequences are first split by the structure that merging cannot change, such as their gate
    applications and fragment depths, since sequences differing in it are never mergeable. Within
    each of those sets, a group is grown by tracking which complete Pauli permutations remain
    available on every qubit: a sequence may join a group only if some complete permutation is
    consistent with the whole group, which makes the merge of the entire group well defined.

    The grouping is greedy, so it is not guaranteed to return the fewest possible groups.

    Args:
        sequences: The instruction sequences to group.

    Returns:
        The groups of positions of mergeable sequences.

    Raises:
        TypeError: If any sequence contains an instruction that is neither a gate application nor a
            partial Pauli permutation.
    """
    candidates: dict[Hashable, list[int]] = defaultdict(list)
    permutation_indices = []
    for idx, sequence in enumerate(sequences):
        key, indices = _merge_candidate_data(sequence)
        candidates[key].append(idx)
        permutation_indices.append(indices)

    masks = _completion_masks()

    groups = []
    for candidate_indices in candidates.values():
        columns = np.stack([permutation_indices[idx] for idx in candidate_indices])
        groups.extend(
            [candidate_indices[row] for row in rows] for rows in _group_by_witness(masks[columns])
        )

    return groups
