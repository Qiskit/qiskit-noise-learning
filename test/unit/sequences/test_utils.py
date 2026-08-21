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

from itertools import combinations, islice, permutations

import numpy as np
import pytest

from qiskit_noise_learning.sequences import (
    ApplyGate,
    InstructionSequence,
    PartialPauliPermutation,
    merge_groups,
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


def _assert_group_merges(group, sequences):
    """Assert that a group of positions merges into a single sequence, in any order.

    Only a bounded number of orders is tried, which exhausts them for the small groups used here.
    """
    for i, j in combinations(group, 2):
        assert sequences[i].is_mergeable_with(sequences[j]), f"{i} and {j} are not mergeable"

    for order in islice(permutations(group), 24):
        merged = sequences[order[0]]
        for idx in order[1:]:
            assert merged.is_mergeable_with(sequences[idx]), f"order {order} fails at {idx}"
            merged = merged.merge(sequences[idx])


def test_merge_groups_cover_every_position_once():
    """Test that the returned groups cover every position of the input exactly once."""
    sequences = _assorted_sequences()

    groups = merge_groups(sequences)

    covered = [idx for group in groups for idx in group]
    assert sorted(covered) == list(range(len(sequences)))


def test_merge_groups_are_mergeable():
    """Test that the sequences within each group merge into a single sequence, in any order."""
    sequences = _assorted_sequences()

    for group in merge_groups(sequences):
        _assert_group_merges(group, sequences)


def test_merge_groups_keeps_mergeable_candidates_together():
    """Test that sequences differing only in mergeable permutations land in a single group."""
    sequences = [
        _sequence([("P",)] * 3, 1, [[6, 15]] * 3),
        _sequence([("P",)] * 3, 1, [[15, 7]] * 3),
        _sequence([("P",)] * 3, 1, [[6, 7]] * 3),
    ]

    groups = merge_groups(sequences)

    assert len(groups) == 1
    assert sorted(groups[0]) == [0, 1, 2]


def test_merge_groups_separates_inconsistent_permutations():
    """Test that sequences ruled out only by their permutation values are placed in separate groups.

    The two sequences share every gate application and fragment depth, so nothing but the
    inconsistency of two distinct complete permutations keeps them apart.
    """
    sequences = [
        _sequence([("P",)] * 3, 1, [[0, 0]] * 3),
        _sequence([("P",)] * 3, 1, [[1, 1]] * 3),
    ]

    assert not sequences[0].is_mergeable_with(sequences[1])
    assert len(merge_groups(sequences)) == 2


def test_merge_groups_does_not_merge_a_mergeable_pair_into_a_conflicting_group():
    """Test that mutual, not just pairwise, mergeability decides a group.

    The three single mappings ``Y -> Y``, ``Z -> Z``, and ``Z -> X`` are pairwise mergeable except
    for the last two, which disagree on the image of ``Z``. No group can hold all three, so one
    mergeable pair must be split across two groups.
    """
    sequences = [_sequence([("P",)] * 3, 1, [[value]] * 3) for value in (14, 6, 7)]

    assert sequences[0].is_mergeable_with(sequences[1])
    assert sequences[0].is_mergeable_with(sequences[2])
    assert not sequences[1].is_mergeable_with(sequences[2])

    groups = merge_groups(sequences)

    assert len(groups) == 2
    for group in groups:
        _assert_group_merges(group, sequences)


@pytest.mark.parametrize(
    ("difference", "other"),
    [
        ("fragment depth", _sequence([("P",)] * 3, 2, [[0, 0]] * 3)),
        ("gate name", _sequence([("Q",)] * 3, 1, [[0, 0]] * 3)),
        ("gate count", _sequence([("P", "P")] * 3, 1, [[0, 0]] * 3)),
        ("qubit count", _sequence([("P",)] * 3, 1, [[0, 0, 0]] * 3)),
    ],
)
def test_merge_groups_splits_on_differences_that_rule_out_merging(difference, other):
    """Test that a difference ruling out any merge places sequences in separate groups."""
    sequence = _sequence([("P",)] * 3, 1, [[0, 0]] * 3)

    groups = merge_groups([sequence, other])

    assert not sequence.is_mergeable_with(
        other
    ), f"expected differing {difference} to block merging"
    assert len(groups) == 2


def test_merge_groups_of_sequences_without_permutations():
    """Test that sequences holding no partial permutations merge on their gates alone."""
    sequences = [
        InstructionSequence(
            start_fragment=[ApplyGate("P")],
            repeatable_fragment=[ApplyGate("L")],
            end_fragment=[ApplyGate("M")],
            fragment_depth=1,
        )
    ] * 2

    assert merge_groups(sequences) == [[0, 1]]


def test_merge_groups_of_one_sequence():
    """Test that a lone sequence gives a single group holding it."""
    assert merge_groups([_sequence([("P",)] * 3, 1, [[0, 0]] * 3)]) == [[0]]


def test_merge_groups_of_no_sequences():
    """Test that grouping no sequences gives no groups."""
    assert merge_groups([]) == []


def test_merge_groups_raises_on_unsupported_instruction():
    """Test that an instruction that is neither a gate nor a partial permutation is rejected."""

    class UnsupportedInstruction:
        """A stand-in for an instruction type the grouping does not know about."""

    sequence = InstructionSequence(
        start_fragment=[UnsupportedInstruction()],
        repeatable_fragment=[ApplyGate("L")],
        end_fragment=[ApplyGate("M")],
        fragment_depth=1,
    )

    with pytest.raises(TypeError, match="UnsupportedInstruction"):
        merge_groups([sequence])
