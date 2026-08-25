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


from itertools import product
from typing import get_args

import pytest

from qiskit_noise_learning.sequences import (
    ApplyGate,
    InstructionSequence,
    PartialPauliPermutation,
    group_mergeable_instruction_sequences,
)
from qiskit_noise_learning.sequences.group_mergeable_instruction_sequences import (
    DEFAULT_GROUPING_STRATEGIES,
    InstructionSequenceOrder,
    MergingStrategy,
)

_ALL_GROUPING_STRATEGIES = list(
    product(get_args(InstructionSequenceOrder), get_args(MergingStrategy))
)
"""Every pairing of a documented instruction sequence order with a documented merging strategy."""

_IDENTITY = {("X", "X"), ("Y", "Y"), ("Z", "Z")}
"""The fully specified permutation leaving every Pauli alone."""

_X_Z_SWAP = {("X", "Z"), ("Y", "Y"), ("Z", "X")}
"""A fully specified permutation exchanging ``X`` and ``Z``, inconsistent with the identity."""

_UNSPECIFIED = set()
"""The permutation specifying no mappings at all, which is consistent with every other."""


def _sequence(gate_names, depth, permutation_sets):
    """Build a sequence whose fragments each hold a partial permutation followed by gates.

    Args:
        gate_names: Three collections of gate names, one per fragment.
        depth: The fragment depth.
        permutation_sets: Three collections of partial permutations, one per fragment, each
            given as one set of ``(from, to)`` Pauli mappings per qubit.
    """
    fragments = [
        [PartialPauliPermutation.from_sets(sets)] + [ApplyGate(name) for name in names]
        for names, sets in zip(gate_names, permutation_sets)
    ]

    return InstructionSequence(
        start_fragment=fragments[0],
        repeatable_fragment=fragments[1],
        end_fragment=fragments[2],
        fragment_depth=depth,
    )


def _assorted_sequences():
    """A collection of sequences spanning several merge candidates and permutation values."""
    two_qubit_permutations = [
        [_IDENTITY, _IDENTITY],
        [_IDENTITY, _UNSPECIFIED],
        [_UNSPECIFIED, _UNSPECIFIED],
        [{("Z", "Z")}, {("Z", "X")}],
        [{("Z", "Z")}, _UNSPECIFIED],
        [{("Z", "X")}, {("Z", "Z")}],
        [_X_Z_SWAP, {("Z", "Z")}],
        [_IDENTITY, {("Z", "Z")}],
    ]

    sequences = []
    for permutation in two_qubit_permutations:
        for depth in (1, 2):
            for names in (("P",), ("Q",)):
                sequences.append(_sequence([names] * 3, depth, [permutation] * 3))

    # a differing number of qubits, and a layout with a differing number of gates
    sequences.append(_sequence([("P",)] * 3, 1, [[_IDENTITY, _IDENTITY, _UNSPECIFIED]] * 3))
    sequences.append(_sequence([("P",)] * 3, 1, [[{("Z", "Z")}, _UNSPECIFIED, _UNSPECIFIED]] * 3))
    sequences.append(_sequence([("P", "P")] * 3, 1, [[_IDENTITY, _IDENTITY]] * 3))

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


def _indexed_sequences(rows):
    """Build one sequence per row from raw partial permutation indices.

    Args:
        rows: One list of partial permutation indices per sequence, indexing the fixed ordering of
            ``partial_permutation_sets()``. An index below ``NUM_COMPLETE_PERMUTATIONS`` is a
            complete permutation, admitting a single completion; the others specify a single Pauli
            mapping, admitting two, except the last, which specifies nothing and admits all six.
    """
    return [
        InstructionSequence(
            start_fragment=[PartialPauliPermutation(row)],
            repeatable_fragment=[ApplyGate("L")],
            end_fragment=[],
            fragment_depth=1,
        )
        for row in rows
    ]


def _assert_group_merges(group, sequences):
    """Assert that a group of positions merges into a single sequence.

    One pass in the order given settles the group, because merging cannot destroy mergeability: the
    members of a mergeable group share a completion, and a merged subgroup specifies the union of
    its members' mappings, which is still a subset of that completion.
    """
    merged = sequences[group[0]]
    for idx in group[1:]:
        assert merged.is_mergeable_with(sequences[idx]), f"{idx} does not merge into group {group}"
        merged = merged.merge(sequences[idx])


def test_group_mergeable_covers_every_position_once():
    """Test that the returned groups cover every position of the input exactly once."""
    sequences = _assorted_sequences()

    groups = group_mergeable_instruction_sequences(sequences)

    covered = [idx for group in groups for idx in group]
    assert sorted(covered) == list(range(len(sequences)))


def test_group_mergeable_groups_are_mergeable():
    """Test that the sequences within each group merge into a single sequence."""
    sequences = _assorted_sequences()

    for group in group_mergeable_instruction_sequences(sequences):
        _assert_group_merges(group, sequences)


@pytest.mark.parametrize("grouping_strategy", _ALL_GROUPING_STRATEGIES)
def test_group_mergeable_with_a_single_grouping_strategy(grouping_strategy):
    """Test that each documented grouping strategy alone partitions into mergeable groups."""
    sequences = _assorted_sequences()

    groups = group_mergeable_instruction_sequences(sequences, [grouping_strategy])

    assert sorted(idx for group in groups for idx in group) == list(range(len(sequences)))
    for group in groups:
        _assert_group_merges(group, sequences)


def test_group_mergeable_takes_the_fewest_groups_of_its_strategies():
    """Test that several strategies give no more groups than the best of them does alone."""
    sequences = _assorted_sequences()

    combined = group_mergeable_instruction_sequences(sequences, _ALL_GROUPING_STRATEGIES)

    for grouping_strategy in _ALL_GROUPING_STRATEGIES:
        alone = group_mergeable_instruction_sequences(sequences, [grouping_strategy])
        assert len(combined) <= len(alone)


def test_group_mergeable_distinguishes_the_constrainedness_orders():
    """Test that the two constrainedness orders group a witness instance differently.

    The complete permutations at indices 3 and 5 admit one completion each and are mutually
    inconsistent, so each opens a group of its own. Visiting them first lets each of the two single
    mappings join one of them; visiting them last pairs the two single mappings into a group whose
    only completion is a third permutation, which neither complete permutation can then join.
    """
    sequences = _indexed_sequences([[3], [5], [12], [11]])

    counts = {
        order: len(group_mergeable_instruction_sequences(sequences, [(order, "first")]))
        for order in ("most-constrained-first", "least-constrained-first")
    }

    assert counts == {"most-constrained-first": 2, "least-constrained-first": 3}


def test_group_mergeable_distinguishes_the_merging_strategies():
    """Test that the two constrainedness-based merging strategies differ on a witness instance.

    Both pick the group admitting the fewest completions, but ``"least-impacted"`` scores a group by
    how many completions the sequence would rule out rather than by how many it already admits. On
    this instance that leaves a group flexible enough for a later sequence to join.
    """
    sequences = _indexed_sequences([[4, 6], [4, 4], [15, 14], [11, 14], [15, 11]])

    counts = {
        merging_strategy: len(
            group_mergeable_instruction_sequences(
                sequences, [("most-constrained-first", merging_strategy)]
            )
        )
        for merging_strategy in ("least-impacted", "most-constrained")
    }

    assert counts == {"least-impacted": 3, "most-constrained": 4}


def test_group_mergeable_first_merging_strategy_picks_the_earliest_group():
    """Test that ``"first"`` merges into the group created earliest among the feasible ones.

    The last sequence here can join either of the first two groups, and the two choices give the
    same number of groups, so only the membership distinguishes them: ``"first"`` must place it with
    the sequence at position 0, whereas ``"most-constrained"`` places it with the one at position 1.
    """
    sequences = _indexed_sequences([[13], [3], [14], [8]])

    groups = group_mergeable_instruction_sequences(sequences, [("input", "first")])

    assert groups == [[0, 3], [1], [2]]


def test_group_mergeable_qubitwise_lexicographic_order_differs_from_the_input_order():
    """Test that visiting sequences in qubitwise-lexicographic order is not the input order.

    Sorting by the completions each sequence admits places sequences specifying similar mappings
    next to each other, which groups this instance more tightly than the order it was given in.
    """
    sequences = _indexed_sequences([[11], [7], [6], [9]])

    counts = {
        order: len(group_mergeable_instruction_sequences(sequences, [(order, "first")]))
        for order in ("qubitwise-lexicographic", "input")
    }

    assert counts == {"qubitwise-lexicographic": 2, "input": 3}


def test_group_mergeable_default_strategies_are_a_documented_pairing():
    """Test that every default grouping strategy is one of the documented pairings."""
    assert set(DEFAULT_GROUPING_STRATEGIES) <= set(_ALL_GROUPING_STRATEGIES)


@pytest.mark.parametrize(
    ("grouping_strategies", "message"),
    [
        ([], "At least one grouping strategy is required"),
        ([("nonsense", "first")], "Unknown instruction sequence order 'nonsense'"),
        ([("input", "nonsense")], "Unknown merging strategy 'nonsense'"),
        ([("input", "first", "surplus")], "Invalid grouping strategy"),
    ],
)
def test_group_mergeable_raises_on_invalid_grouping_strategies(grouping_strategies, message):
    """Test that grouping strategies that are not documented pairings are rejected."""
    with pytest.raises(ValueError, match=message):
        group_mergeable_instruction_sequences([], grouping_strategies)


def test_group_mergeable_keeps_mergeable_candidates_together():
    """Test that sequences differing only in mergeable permutations land in a single group."""
    sequences = [
        _sequence([("P",)] * 3, 1, [[{("Z", "Z")}, _UNSPECIFIED]] * 3),
        _sequence([("P",)] * 3, 1, [[_UNSPECIFIED, {("Z", "X")}]] * 3),
        _sequence([("P",)] * 3, 1, [[{("Z", "Z")}, {("Z", "X")}]] * 3),
    ]

    groups = group_mergeable_instruction_sequences(sequences)

    assert len(groups) == 1
    assert sorted(groups[0]) == [0, 1, 2]


def test_group_mergeable_does_not_merge_a_mergeable_pair_into_a_conflicting_group():
    """Test that mutual, not just pairwise, mergeability decides a group."""
    sequences = [
        _sequence([("P",)] * 3, 1, [[{mapping}]] * 3)
        for mapping in [("Y", "Y"), ("Z", "Z"), ("Z", "X")]
    ]

    assert sequences[0].is_mergeable_with(sequences[1])
    assert sequences[0].is_mergeable_with(sequences[2])
    assert not sequences[1].is_mergeable_with(sequences[2])

    groups = group_mergeable_instruction_sequences(sequences)

    assert len(groups) == 2
    for group in groups:
        _assert_group_merges(group, sequences)


@pytest.mark.parametrize(
    ("difference", "other"),
    [
        ("permutation values", _sequence([("P",)] * 3, 1, [[_X_Z_SWAP, _X_Z_SWAP]] * 3)),
        ("fragment depth", _sequence([("P",)] * 3, 2, [[_IDENTITY, _IDENTITY]] * 3)),
        ("gate name", _sequence([("Q",)] * 3, 1, [[_IDENTITY, _IDENTITY]] * 3)),
        ("gate count", _sequence([("P", "P")] * 3, 1, [[_IDENTITY, _IDENTITY]] * 3)),
        ("qubit count", _sequence([("P",)] * 3, 1, [[_IDENTITY] * 3] * 3)),
    ],
)
def test_group_mergeable_splits_on_differences_that_rule_out_merging(difference, other):
    """Test that a difference ruling out any merge places sequences in separate groups."""
    sequence = _sequence([("P",)] * 3, 1, [[_IDENTITY, _IDENTITY]] * 3)

    assert not sequence.is_mergeable_with(other), f"differing {difference} must block merging"
    assert len(group_mergeable_instruction_sequences([sequence, other])) == 2


def test_group_mergeable_of_sequences_without_permutations(make_instruction_sequence):
    """Test that sequences holding no partial permutations merge on their gates alone."""
    sequences = [make_instruction_sequence(), make_instruction_sequence()]

    assert group_mergeable_instruction_sequences(sequences) == [[0, 1]]


def test_group_mergeable_of_one_sequence():
    """Test that a lone sequence gives a single group holding it."""
    sequence = _sequence([("P",)] * 3, 1, [[_IDENTITY, _IDENTITY]] * 3)

    assert group_mergeable_instruction_sequences([sequence]) == [[0]]


def test_group_mergeable_of_no_sequences():
    """Test that grouping no sequences gives no groups."""
    assert group_mergeable_instruction_sequences([]) == []


def test_group_mergeable_raises_on_non_instruction():
    """Test that an object that is not an instruction at all is rejected."""

    class NotAnInstruction:
        """A stand-in for an object that carries no structure token."""

    sequence = InstructionSequence(
        start_fragment=[NotAnInstruction()],
        repeatable_fragment=[ApplyGate("L")],
        end_fragment=[ApplyGate("M")],
        fragment_depth=1,
    )

    with pytest.raises(TypeError, match="NotAnInstruction"):
        group_mergeable_instruction_sequences([sequence])
