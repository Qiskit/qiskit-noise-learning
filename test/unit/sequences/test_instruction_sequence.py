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
    assert seq.fragment_depth is None


def test_construction_with_depth():
    """Test construction with a specified fragment_depth."""

    start_fragment = [ApplyGate("P")]
    repeatable_fragment = [ApplyGate("L0"), ApplyGate("L1")]
    end_fragment = [ApplyGate("M")]

    seq = InstructionSequence(
        start_fragment=start_fragment,
        repeatable_fragment=repeatable_fragment,
        end_fragment=end_fragment,
        fragment_depth=3,
    )

    assert seq.start_fragment == start_fragment
    assert seq.repeatable_fragment == repeatable_fragment
    assert seq.end_fragment == end_fragment
    assert seq.fragment_depth == 3
    assert len(seq) == 8


def test_is_mergeable_with():
    """Test mergeability checking for InstructionSequence (variable-fragment_depth)."""

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
    """Test that sequences with different fragment_depths are not mergeable."""

    seq0 = InstructionSequence(
        start_fragment=[ApplyGate("P")],
        repeatable_fragment=[ApplyGate("L0")],
        end_fragment=[ApplyGate("M")],
        fragment_depth=5,
    )

    seq1 = InstructionSequence(
        start_fragment=[ApplyGate("P")],
        repeatable_fragment=[ApplyGate("L0")],
        end_fragment=[ApplyGate("M")],
        fragment_depth=4,
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

    # fragment_depth mismatch
    seq0 = InstructionSequence(
        start_fragment=[ApplyGate("P")],
        repeatable_fragment=[],
        end_fragment=[ApplyGate("M")],
        fragment_depth=3,
    )
    seq1 = InstructionSequence(
        start_fragment=[ApplyGate("P")],
        repeatable_fragment=[],
        end_fragment=[ApplyGate("M")],
        fragment_depth=4,
    )
    with pytest.raises(ValueError, match="different fragment depths"):
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
    """Test that complete() preserves the fragment_depth."""

    seq = InstructionSequence(
        start_fragment=[ApplyGate("P")],
        repeatable_fragment=[ApplyGate("L0")],
        end_fragment=[ApplyGate("M")],
        fragment_depth=7,
    )

    assert seq.complete().fragment_depth == 7


_PERMUTATION = PartialPauliPermutation.from_sets([{("X", "Y")}])
"""A one-qubit partial permutation."""

_OTHER_PERMUTATION = PartialPauliPermutation.from_sets([{("Y", "Z")}])
"""A one-qubit partial permutation specifying a different mapping than ``_PERMUTATION``."""

_WIDE_PERMUTATION = PartialPauliPermutation.from_sets([{("X", "Y")}, {("Y", "Z")}])
"""A two-qubit partial permutation."""

_REPEATABLE_FRAGMENT = (ApplyGate("L0"), ApplyGate("L1"))
"""The repeatable fragment of every variant that does not vary it."""

_DIFFERENT_GATES = {"different_repeatable_gate", "different_start_gate", "fewer_repeatable_gates"}
"""The variants whose gate applications differ from the reference's."""

_SAME_STRUCTURE = {"reference", "different_permutation"}
"""The variants whose whole instruction structure matches the reference's."""


def _sequence(start_fragment, repeatable_fragment=_REPEATABLE_FRAGMENT):
    """Build an unbound sequence with a fixed end fragment, varying only what is passed."""
    return InstructionSequence(
        start_fragment=start_fragment,
        repeatable_fragment=repeatable_fragment,
        end_fragment=[_OTHER_PERMUTATION, ApplyGate("M")],
    )


@pytest.fixture
def variants():
    """A reference sequence together with variants of it, keyed by how each one differs from it."""
    return {
        "reference": _sequence([ApplyGate("P"), _PERMUTATION]),
        # the permutation sits in the same place but specifies a different Pauli mapping
        "different_permutation": _sequence([ApplyGate("P"), _OTHER_PERMUTATION]),
        # the permutation precedes the gate rather than following it
        "moved_permutation": _sequence([_PERMUTATION, ApplyGate("P")]),
        # the permutation covers two qubits rather than one
        "wider_permutation": _sequence([ApplyGate("P"), _WIDE_PERMUTATION]),
        # a different gate label in the repeatable fragment
        "different_repeatable_gate": _sequence(
            [ApplyGate("P"), _PERMUTATION], [ApplyGate("L0"), ApplyGate("L0")]
        ),
        # a different gate label in the start fragment
        "different_start_gate": _sequence([ApplyGate("L0"), _PERMUTATION]),
        # fewer gates in the repeatable fragment
        "fewer_repeatable_gates": _sequence([ApplyGate("P"), _PERMUTATION], [ApplyGate("L0")]),
    }


def test_gate_key(variants):
    """Test that gate_key sees nothing but the gate applications."""

    reference = variants["reference"]

    for name, variant in variants.items():
        assert (reference.gate_key == variant.gate_key) is (name not in _DIFFERENT_GATES), name


def test_structure_key(variants):
    """Test that structure_key refines gate_key by the placement and width of the permutations."""

    reference = variants["reference"]

    # which mapping a permutation specifies is invisible to the key; where it sits and how many
    # qubits it covers are not
    for name, variant in variants.items():
        assert (reference.structure_key == variant.structure_key) is (name in _SAME_STRUCTURE), name


def test_keys_depend_on_depth(variants):
    """Test that both keys distinguish sequences differing only in fragment depth."""

    reference = variants["reference"]
    at_three, at_four = reference.bind_at(3), reference.bind_at(4)

    assert at_three.gate_key == reference.bind_at(3).gate_key
    assert at_three.structure_key == reference.bind_at(3).structure_key

    assert at_three.gate_key != at_four.gate_key
    assert at_three.structure_key != at_four.structure_key


def test_keys_are_hashable(variants):
    """Test that both keys can be used as dictionary keys."""

    # every variant carries a distinct pair of keys except different_permutation, which shares both
    # of the reference's
    assert len({(v.gate_key, v.structure_key) for v in variants.values()}) == len(variants) - 1


def test_structure_key_raises_on_unsupported_instruction():
    """Test that an instruction that is neither a gate nor a partial permutation is rejected."""

    class UnsupportedInstruction:
        """A stand-in for an instruction type the keys do not know about."""

    sequence = _sequence([ApplyGate("P"), UnsupportedInstruction()])

    with pytest.raises(TypeError, match="UnsupportedInstruction"):
        _ = sequence.structure_key

    # gate_key ignores instructions it does not recognize
    assert sequence.gate_key == (None, ("P",), ("L0", "L1"), ("M",))


def test_bind_at():
    """Test bind_at returns a new instance with the specified fragment_depth."""

    start_fragment = [ApplyGate("P")]
    repeatable_fragment = [ApplyGate("L0"), ApplyGate("L1")]
    end_fragment = [ApplyGate("M")]

    seq = InstructionSequence(
        start_fragment=start_fragment,
        repeatable_fragment=repeatable_fragment,
        end_fragment=end_fragment,
    )
    assert seq.fragment_depth is None

    bound = seq.bind_at(5)
    assert bound.fragment_depth == 5
    assert bound.start_fragment == start_fragment
    assert bound.repeatable_fragment == repeatable_fragment
    assert bound.end_fragment == end_fragment
    assert isinstance(bound, InstructionSequence)

    # bind_at with None gives variable-fragment_depth
    unbound = bound.unbind()
    assert unbound.fragment_depth is None
    assert unbound == seq
