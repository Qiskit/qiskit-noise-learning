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

import pytest

from qiskit_noise_learning.sequences import (
    ApplyGate,
    InstructionSequence,
    PartialPauliPermutation,
)


def test_construction():
    """Test construction and attributes."""

    start_fragment = [ApplyGate("P")]
    repeatable_fragment = [ApplyGate("L0"), ApplyGate("L1")]
    end_fragment = [ApplyGate("M")]

    seq = InstructionSequence(
        start_fragment=start_fragment,
        repeatable_fragment=repeatable_fragment,
        end_fragment=end_fragment,
    )

    assert seq.start_fragment == start_fragment
    assert seq.repeatable_fragment == repeatable_fragment
    assert seq.end_fragment == end_fragment
    assert seq.depth is None


def test_construction_with_depth():
    """Test construction with a specified depth."""

    start_fragment = [ApplyGate("P")]
    repeatable_fragment = [ApplyGate("L0"), ApplyGate("L1")]
    end_fragment = [ApplyGate("M")]

    seq = InstructionSequence(
        start_fragment=start_fragment,
        repeatable_fragment=repeatable_fragment,
        end_fragment=end_fragment,
        depth=3,
    )

    assert seq.start_fragment == start_fragment
    assert seq.repeatable_fragment == repeatable_fragment
    assert seq.end_fragment == end_fragment
    assert seq.depth == 3
    assert len(seq) == 8


def test_is_mergeable_with():
    """Test mergeability checking for InstructionSequence (variable-depth)."""

    seq0 = InstructionSequence(
        start_fragment=[
            ApplyGate("P"),
            PartialPauliPermutation.from_sets([{("X", "Y"), ("Y", "Z")}, set(), {("Z", "Y")}]),
        ],
        repeatable_fragment=[ApplyGate("L0")],
        end_fragment=[
            PartialPauliPermutation.from_sets([{("Y", "X"), ("Z", "Y")}, set(), {("Y", "Z")}]),
            ApplyGate("M"),
        ],
    )

    assert not seq0.is_mergeable_with(InstructionSequence([], [], []))
    # different gate labels
    assert not seq0.is_mergeable_with(
        InstructionSequence(
            start_fragment=[
                ApplyGate("P"),
                PartialPauliPermutation.from_sets([{("X", "Y"), ("Y", "Z")}, set(), {("Z", "Y")}]),
            ],
            repeatable_fragment=[ApplyGate("L1")],
            end_fragment=[
                PartialPauliPermutation.from_sets([{("Y", "X"), ("Z", "Y")}, set(), {("Y", "Z")}]),
                ApplyGate("M"),
            ],
        )
    )
    # incompatible permutations
    assert not seq0.is_mergeable_with(
        InstructionSequence(
            start_fragment=[
                ApplyGate("P"),
                PartialPauliPermutation.from_sets([{("Y", "X"), ("Z", "Y")}, set(), {("Z", "Y")}]),
            ],
            repeatable_fragment=[ApplyGate("L0")],
            end_fragment=[
                PartialPauliPermutation.from_sets([{("Y", "X"), ("Z", "Y")}, set(), {("Y", "Z")}]),
                ApplyGate("M"),
            ],
        )
    )
    # compatible permutations
    assert seq0.is_mergeable_with(
        InstructionSequence(
            start_fragment=[
                ApplyGate("P"),
                PartialPauliPermutation.from_sets([{("X", "Y")}, {("Y", "Z")}, {("Z", "Y")}]),
            ],
            repeatable_fragment=[ApplyGate("L0")],
            end_fragment=[
                PartialPauliPermutation.from_sets([{("X", "Z")}, set(), {("Y", "Z")}]),
                ApplyGate("M"),
            ],
        )
    )


def test_is_mergeable_with_depth_mismatch():
    """Test that sequences with different depths are not mergeable."""

    seq0 = InstructionSequence(
        start_fragment=[ApplyGate("P")],
        repeatable_fragment=[ApplyGate("L0")],
        end_fragment=[ApplyGate("M")],
        depth=5,
    )

    seq1 = InstructionSequence(
        start_fragment=[ApplyGate("P")],
        repeatable_fragment=[ApplyGate("L0")],
        end_fragment=[ApplyGate("M")],
        depth=4,
    )

    assert not seq0.is_mergeable_with(seq1)

    # None vs int also not mergeable
    seq2 = InstructionSequence(
        start_fragment=[ApplyGate("P")],
        repeatable_fragment=[ApplyGate("L0")],
        end_fragment=[ApplyGate("M")],
    )

    assert not seq0.is_mergeable_with(seq2)


def test_merge():
    """Test merging of instruction sequences."""

    seq0 = InstructionSequence(
        start_fragment=[
            ApplyGate("P"),
            PartialPauliPermutation.from_sets([{("Y", "Z")}, set(), {("Z", "Y")}]),
        ],
        repeatable_fragment=[ApplyGate("L0")],
        end_fragment=[
            PartialPauliPermutation.from_sets([{("Y", "X")}, set(), {("Y", "Z")}]),
            ApplyGate("M"),
        ],
    )
    seq1 = InstructionSequence(
        start_fragment=[
            ApplyGate("P"),
            PartialPauliPermutation.from_sets([{("X", "Y")}, {("Y", "Z")}, {("Z", "Y")}]),
        ],
        repeatable_fragment=[ApplyGate("L0")],
        end_fragment=[
            PartialPauliPermutation.from_sets([{("X", "Z")}, set(), {("Y", "Z")}]),
            ApplyGate("M"),
        ],
    )

    seq2 = seq0.merge(seq1)
    expected = InstructionSequence(
        start_fragment=[
            ApplyGate("P"),
            PartialPauliPermutation.from_sets(
                [{("X", "Y"), ("Y", "Z")}, {("Y", "Z")}, {("Z", "Y")}]
            ),
        ],
        repeatable_fragment=[ApplyGate("L0")],
        end_fragment=[
            PartialPauliPermutation.from_sets([{("Y", "X"), ("Z", "Y")}, set(), {("Y", "Z")}]),
            ApplyGate("M"),
        ],
    )
    assert seq2 == expected


def test_merge_failures():
    """Test merging of instruction sequence failures."""

    # inconsistent lengths
    seq0 = InstructionSequence(
        start_fragment=[ApplyGate("P")],
        repeatable_fragment=[],
        end_fragment=[ApplyGate("M")],
    )
    seq1 = InstructionSequence(
        start_fragment=[ApplyGate("P"), ApplyGate("L0")],
        repeatable_fragment=[],
        end_fragment=[ApplyGate("M")],
    )
    with pytest.raises(ValueError, match="start fragments of different lengths"):
        seq0.merge(seq1)

    # inconsistent gate labels
    seq0 = InstructionSequence(
        start_fragment=[ApplyGate("P"), ApplyGate("L0")],
        repeatable_fragment=[],
        end_fragment=[ApplyGate("M")],
    )
    seq1 = InstructionSequence(
        start_fragment=[ApplyGate("P"), ApplyGate("L1")],
        repeatable_fragment=[],
        end_fragment=[ApplyGate("M")],
    )
    with pytest.raises(ValueError, match="Cannot merge ApplyGate instructions"):
        seq0.merge(seq1)

    # inconsistent partial permutations
    seq0 = InstructionSequence(
        start_fragment=[
            ApplyGate("P"),
            PartialPauliPermutation.from_sets([{("X", "X")}]),
        ],
        repeatable_fragment=[],
        end_fragment=[ApplyGate("M")],
    )
    seq1 = InstructionSequence(
        start_fragment=[
            ApplyGate("P"),
            PartialPauliPermutation.from_sets([{("X", "Y")}]),
        ],
        repeatable_fragment=[],
        end_fragment=[ApplyGate("M")],
    )
    with pytest.raises(ValueError, match="Cannot merge inconsistent partial permutations"):
        seq0.merge(seq1)

    # depth mismatch
    seq0 = InstructionSequence(
        start_fragment=[ApplyGate("P")],
        repeatable_fragment=[],
        end_fragment=[ApplyGate("M")],
        depth=3,
    )
    seq1 = InstructionSequence(
        start_fragment=[ApplyGate("P")],
        repeatable_fragment=[],
        end_fragment=[ApplyGate("M")],
        depth=4,
    )
    with pytest.raises(ValueError, match="different depths"):
        seq0.merge(seq1)


def test_complete():
    """Test InstructionSequence.complete."""

    start_permutation = PartialPauliPermutation.from_sets([{("Z", "X")}, {("X", "Y")}])
    repeatable_permutation = PartialPauliPermutation.from_sets([{("Y", "Z")}, set()])
    end_permutation = PartialPauliPermutation.from_sets([{("X", "X")}, {("X", "Y"), ("Y", "Z")}])

    seq = InstructionSequence(
        start_fragment=[ApplyGate("P"), start_permutation],
        repeatable_fragment=[ApplyGate("L0"), repeatable_permutation],
        end_fragment=[end_permutation, ApplyGate("M")],
    )

    expected = InstructionSequence(
        start_fragment=[
            ApplyGate("P"),
            start_permutation.complete(),
        ],
        repeatable_fragment=[
            ApplyGate("L0"),
            repeatable_permutation.complete(),
        ],
        end_fragment=[
            end_permutation.complete(),
            ApplyGate("M"),
        ],
    )

    assert expected == seq.complete()


def test_complete_preserves_depth():
    """Test that complete() preserves the depth."""

    seq = InstructionSequence(
        start_fragment=[ApplyGate("P")],
        repeatable_fragment=[ApplyGate("L0")],
        end_fragment=[ApplyGate("M")],
        depth=7,
    )

    assert seq.complete().depth == 7


def test_has_same_structure_as():
    """Test has_same_structure_as for InstructionSequence."""

    seq0 = InstructionSequence(
        start_fragment=[
            ApplyGate("P"),
            PartialPauliPermutation.from_sets([{("X", "Y")}]),
        ],
        repeatable_fragment=[
            ApplyGate("L0"),
            ApplyGate("L1"),
        ],
        end_fragment=[
            PartialPauliPermutation.from_sets([{("Y", "Z")}]),
            ApplyGate("M"),
        ],
    )

    # same structure with different PartialPauliPermutations
    seq1 = InstructionSequence(
        start_fragment=[
            ApplyGate("P"),
            PartialPauliPermutation.from_sets([{("Z", "X")}]),
        ],
        repeatable_fragment=[
            ApplyGate("L0"),
            ApplyGate("L1"),
        ],
        end_fragment=[
            PartialPauliPermutation.from_sets([{("X", "Y")}]),
            ApplyGate("M"),
        ],
    )
    assert seq0.has_same_structure_as(seq1)

    # different gate label in repeatable_fragment
    seq2 = InstructionSequence(
        start_fragment=[
            ApplyGate("P"),
            PartialPauliPermutation.from_sets([{("X", "Y")}]),
        ],
        repeatable_fragment=[
            ApplyGate("L0"),
            ApplyGate("L0"),
        ],
        end_fragment=[
            PartialPauliPermutation.from_sets([{("Y", "Z")}]),
            ApplyGate("M"),
        ],
    )
    assert not seq0.has_same_structure_as(seq2)

    # different gate label in start_fragment
    seq3 = InstructionSequence(
        start_fragment=[
            ApplyGate("L0"),
            PartialPauliPermutation.from_sets([{("X", "Y")}]),
        ],
        repeatable_fragment=[
            ApplyGate("L0"),
            ApplyGate("L1"),
        ],
        end_fragment=[
            PartialPauliPermutation.from_sets([{("Y", "Z")}]),
            ApplyGate("M"),
        ],
    )
    assert not seq0.has_same_structure_as(seq3)

    # different number of gates in repeatable_fragment
    seq4 = InstructionSequence(
        start_fragment=[
            ApplyGate("P"),
            PartialPauliPermutation.from_sets([{("X", "Y")}]),
        ],
        repeatable_fragment=[ApplyGate("L0")],
        end_fragment=[
            PartialPauliPermutation.from_sets([{("Y", "Z")}]),
            ApplyGate("M"),
        ],
    )
    assert not seq0.has_same_structure_as(seq4)


def test_has_same_structure_as_depth():
    """Test that has_same_structure_as requires matching depths."""

    seq0 = InstructionSequence(
        start_fragment=[ApplyGate("P")],
        repeatable_fragment=[ApplyGate("L0")],
        end_fragment=[ApplyGate("M")],
        depth=3,
    )
    seq1 = InstructionSequence(
        start_fragment=[ApplyGate("P")],
        repeatable_fragment=[ApplyGate("L0")],
        end_fragment=[ApplyGate("M")],
        depth=4,
    )
    assert not seq0.has_same_structure_as(seq1)

    seq2 = InstructionSequence(
        start_fragment=[ApplyGate("P")],
        repeatable_fragment=[ApplyGate("L0")],
        end_fragment=[ApplyGate("M")],
        depth=3,
    )
    assert seq0.has_same_structure_as(seq2)


def test_bind_at():
    """Test bind_at returns a new instance with the specified depth."""

    start_fragment = [ApplyGate("P")]
    repeatable_fragment = [ApplyGate("L0"), ApplyGate("L1")]
    end_fragment = [ApplyGate("M")]

    seq = InstructionSequence(
        start_fragment=start_fragment,
        repeatable_fragment=repeatable_fragment,
        end_fragment=end_fragment,
    )
    assert seq.depth is None

    bound = seq.bind_at(5)
    assert bound.depth == 5
    assert bound.start_fragment == start_fragment
    assert bound.repeatable_fragment == repeatable_fragment
    assert bound.end_fragment == end_fragment
    assert isinstance(bound, InstructionSequence)

    # bind_at with None gives variable-depth
    unbound = bound.without_depth()
    assert unbound.depth is None
    assert unbound == seq
