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
from .partial_pauli_permutation import PartialPauliPermutation, consistency_matrix


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


def partition_instruction_sequences(
    sequences: Sequence[InstructionSequence],
) -> list[tuple[list[int], np.ndarray]]:
    r"""Partition instruction sequences such that elements of disjoint groups are not mergeable.

    The returned groups partition the positions of ``sequences``: every position appears in exactly
    one group, and any two sequences drawn from *different* groups are guaranteed to be
    non-mergeable. Each group also carries the pairwise mergeability of its own members.

    A group is returned as a pair ``(indices, mergeable)``, where ``indices`` are the positions in
    ``sequences`` that belong to the group, and ``mergeable`` is a boolean array of shape
    ``(len(indices),) * 2`` whose ``[i, j]`` entry is whether ``sequences[indices[i]]`` is mergeable
    with ``sequences[indices[j]]``.

    Args:
        sequences: The instruction sequences to partition.

    Returns:
        The groups of the partition, each as a pair of indices and their mergeability array.

    Raises:
        TypeError: If any sequence contains an instruction that is neither a gate application nor a
            partial Pauli permutation.
    """
    groups: dict[Hashable, list[int]] = defaultdict(list)
    permutation_indices = []
    for idx, sequence in enumerate(sequences):
        key, indices = _merge_candidate_data(sequence)
        groups[key].append(idx)
        permutation_indices.append(indices)

    consistency = consistency_matrix()

    partition = []
    for group_indices in groups.values():
        columns = np.stack([permutation_indices[idx] for idx in group_indices])

        mergeable = np.ones((len(group_indices),) * 2, dtype=np.bool_)
        for column in columns.T:
            # a column where every sequence agrees imposes no constraint
            if (column == column[0]).all():
                continue
            mergeable &= consistency[np.ix_(column, column)]

        partition.append((group_indices, mergeable))

    return partition
