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

import numpy as np
import pytest

from qiskit_noise_learning.sequences import (
    ApplyGate,
    InstructionSequence,
    PartialPauliPermutation,
    partition_instruction_sequences,
)


def _sequence(gate_names, depth, permutation_indices):
    """Build a sequence whose fragments each hold a partial permutation followed by gates.

    Args:
        gate_names: Three collections of gate names, one per fragment.
        depth: The fragment depth.
        permutation_indices: Three collections of partial permutation indices, one per fragment.
    """
    fragments = [
        [PartialPauliPermutation(np.array(indices))] + [ApplyGate(name) for name in names]
        for names, indices in zip(gate_names, permutation_indices)
    ]

    return InstructionSequence(
        start_fragment=fragments[0],
        repeatable_fragment=fragments[1],
        end_fragment=fragments[2],
        fragment_depth=depth,
    )


def _assorted_sequences():
    """A collection of sequences spanning several merge candidates and permutation values."""
    # indices 0-5 are the complete permutations, 6-14 the single mappings, 15 the unconstrained one
    two_qubit_values = [[0, 0], [0, 15], [15, 15], [6, 7], [6, 15], [7, 6], [1, 6], [0, 6]]

    sequences = []
    for values in two_qubit_values:
        for depth in (1, 2):
            for names in (("P",), ("Q",)):
                sequences.append(_sequence([names] * 3, depth, [values] * 3))

    # a differing number of qubits, and a layout with a differing number of gates
    sequences.append(_sequence([("P",)] * 3, 1, [[0, 0, 15]] * 3))
    sequences.append(_sequence([("P",)] * 3, 1, [[6, 15, 15]] * 3))
    sequences.append(_sequence([("P", "P")] * 3, 1, [[0, 0]] * 3))

    # sequences holding no partial permutations at all
    for names in (("P",), ("Q",)):
        sequences.append(
            InstructionSequence(
                start_fragment=[ApplyGate(name) for name in names],
                repeatable_fragment=[ApplyGate("L")],
                end_fragment=[ApplyGate("M")],
                fragment_depth=1,
            )
        )

    return sequences


def test_partition_covers_every_position_once():
    """Test that the returned groups cover every position of the input exactly once."""
    sequences = _assorted_sequences()

    partition = partition_instruction_sequences(sequences)

    covered = [idx for indices, _ in partition for idx in indices]
    assert sorted(covered) == list(range(len(sequences)))


def test_partition_matches_pairwise_is_mergeable_with():
    """Test that the partition and its mergeability arrays agree with pairwise mergeability.

    Sequences in different groups must never be mergeable, and within a group the mergeability
    array must reproduce ``is_mergeable_with`` for every pair.
    """
    sequences = _assorted_sequences()

    partition = partition_instruction_sequences(sequences)

    for group_idx, (indices, mergeable) in enumerate(partition):
        assert mergeable.shape == (len(indices),) * 2

        # within a group, the array reproduces pairwise mergeability
        for i, idx0 in enumerate(indices):
            for j, idx1 in enumerate(indices):
                assert bool(mergeable[i, j]) == sequences[idx0].is_mergeable_with(sequences[idx1])

        # across groups, no pair is ever mergeable
        for other_indices, _ in partition[group_idx + 1 :]:
            for idx0 in indices:
                for idx1 in other_indices:
                    assert not sequences[idx0].is_mergeable_with(sequences[idx1])


def test_partition_keeps_mergeable_candidates_together():
    """Test that sequences differing only in mergeable permutations land in a single group."""
    sequences = [
        _sequence([("P",)] * 3, 1, [[6, 15]] * 3),
        _sequence([("P",)] * 3, 1, [[15, 7]] * 3),
        _sequence([("P",)] * 3, 1, [[6, 7]] * 3),
    ]

    partition = partition_instruction_sequences(sequences)

    assert len(partition) == 1
    indices, mergeable = partition[0]
    assert sorted(indices) == [0, 1, 2]
    assert mergeable.all()


@pytest.mark.parametrize(
    ("difference", "other"),
    [
        ("fragment depth", _sequence([("P",)] * 3, 2, [[0, 0]] * 3)),
        ("gate name", _sequence([("Q",)] * 3, 1, [[0, 0]] * 3)),
        ("gate count", _sequence([("P", "P")] * 3, 1, [[0, 0]] * 3)),
        ("qubit count", _sequence([("P",)] * 3, 1, [[0, 0, 0]] * 3)),
    ],
)
def test_partition_splits_on_differences_that_rule_out_merging(difference, other):
    """Test that a difference ruling out any merge places sequences in separate groups."""
    sequence = _sequence([("P",)] * 3, 1, [[0, 0]] * 3)

    partition = partition_instruction_sequences([sequence, other])

    assert not sequence.is_mergeable_with(
        other
    ), f"expected differing {difference} to block merging"
    assert len(partition) == 2


def test_partition_of_no_sequences():
    """Test that partitioning no sequences gives no groups."""
    assert partition_instruction_sequences([]) == []


def test_partition_raises_on_unsupported_instruction():
    """Test that an instruction that is neither a gate nor a partial permutation is rejected."""

    class UnsupportedInstruction:
        """A stand-in for an instruction type the partitioning does not know about."""

    sequence = InstructionSequence(
        start_fragment=[UnsupportedInstruction()],
        repeatable_fragment=[ApplyGate("L")],
        end_fragment=[ApplyGate("M")],
        fragment_depth=1,
    )

    with pytest.raises(TypeError, match="UnsupportedInstruction"):
        partition_instruction_sequences([sequence])
