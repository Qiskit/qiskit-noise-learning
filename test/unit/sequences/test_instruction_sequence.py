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
from qiskit import QuantumCircuit
from qiskit.quantum_info import Clifford

from qiskit_noise_learning.gate_sets import ModelGate, ModelGateSet
from qiskit_noise_learning.sequences import (
    ApplyGate,
    InstructionSequence,
    PartialPauliPermutation,
)


@pytest.fixture()
def gate_set():
    model_gate_set = ModelGateSet(3)
    ident = [((0, 1, 2), Clifford(QuantumCircuit(3)))]
    model_gate_set.add_gate(ModelGate("P", ident, prep_idxs=range(3)))
    model_gate_set.add_gate(ModelGate("M", ident, meas_idxs=range(3)))
    model_gate_set.add_gate(ModelGate("L0", ident))
    model_gate_set.add_gate(ModelGate("L1", ident))
    return model_gate_set


@pytest.fixture
def model_gate_set_1q() -> ModelGateSet:
    model_gate_set = ModelGateSet(1)
    ident = [((0,), Clifford(QuantumCircuit(1)))]
    model_gate_set.add_gate(ModelGate("P", ident, prep_idxs=range(1)))
    model_gate_set.add_gate(ModelGate("M", ident, meas_idxs=range(1)))
    model_gate_set.add_gate(ModelGate("L0", ident))
    model_gate_set.add_gate(ModelGate("L1", ident))
    return model_gate_set


def test_construction(gate_set):
    """Test construction and attributes."""

    start_fragment = [ApplyGate(gate_set["P"])]
    repeatable_fragment = [ApplyGate(gate_set["L0"]), ApplyGate(gate_set["L1"])]
    end_fragment = [ApplyGate(gate_set["M"])]

    seq = InstructionSequence(
        start_fragment=start_fragment,
        repeatable_fragment=repeatable_fragment,
        end_fragment=end_fragment,
    )

    assert seq.start_fragment == start_fragment
    assert seq.repeatable_fragment == repeatable_fragment
    assert seq.end_fragment == end_fragment
    assert seq.depth is None


def test_construction_with_depth(gate_set):
    """Test construction with a specified depth."""

    start_fragment = [ApplyGate(gate_set["P"])]
    repeatable_fragment = [ApplyGate(gate_set["L0"]), ApplyGate(gate_set["L1"])]
    end_fragment = [ApplyGate(gate_set["M"])]

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


def test_is_mergeable_with(gate_set):
    """Test mergeability checking for InstructionSequence (variable-depth)."""

    seq0 = InstructionSequence(
        start_fragment=[
            ApplyGate(gate_set["P"]),
            PartialPauliPermutation.from_sets([{("X", "Y"), ("Y", "Z")}, set(), {("Z", "Y")}]),
        ],
        repeatable_fragment=[ApplyGate(gate_set["L0"])],
        end_fragment=[
            PartialPauliPermutation.from_sets([{("Y", "X"), ("Z", "Y")}, set(), {("Y", "Z")}]),
            ApplyGate(gate_set["M"]),
        ],
    )

    assert not seq0.is_mergeable_with(InstructionSequence([], [], []))
    # different gate labels
    assert not seq0.is_mergeable_with(
        InstructionSequence(
            start_fragment=[
                ApplyGate(gate_set["P"]),
                PartialPauliPermutation.from_sets([{("X", "Y"), ("Y", "Z")}, set(), {("Z", "Y")}]),
            ],
            repeatable_fragment=[ApplyGate(gate_set["L1"])],
            end_fragment=[
                PartialPauliPermutation.from_sets([{("Y", "X"), ("Z", "Y")}, set(), {("Y", "Z")}]),
                ApplyGate(gate_set["M"]),
            ],
        )
    )
    # incompatible permutations
    assert not seq0.is_mergeable_with(
        InstructionSequence(
            start_fragment=[
                ApplyGate(gate_set["P"]),
                PartialPauliPermutation.from_sets([{("Y", "X"), ("Z", "Y")}, set(), {("Z", "Y")}]),
            ],
            repeatable_fragment=[ApplyGate(gate_set["L0"])],
            end_fragment=[
                PartialPauliPermutation.from_sets([{("Y", "X"), ("Z", "Y")}, set(), {("Y", "Z")}]),
                ApplyGate(gate_set["M"]),
            ],
        )
    )
    # compatible permutations
    assert seq0.is_mergeable_with(
        InstructionSequence(
            start_fragment=[
                ApplyGate(gate_set["P"]),
                PartialPauliPermutation.from_sets([{("X", "Y")}, {("Y", "Z")}, {("Z", "Y")}]),
            ],
            repeatable_fragment=[ApplyGate(gate_set["L0"])],
            end_fragment=[
                PartialPauliPermutation.from_sets([{("X", "Z")}, set(), {("Y", "Z")}]),
                ApplyGate(gate_set["M"]),
            ],
        )
    )


def test_is_mergeable_with_depth_mismatch(gate_set):
    """Test that sequences with different depths are not mergeable."""

    seq0 = InstructionSequence(
        start_fragment=[ApplyGate(gate_set["P"])],
        repeatable_fragment=[ApplyGate(gate_set["L0"])],
        end_fragment=[ApplyGate(gate_set["M"])],
        depth=5,
    )

    seq1 = InstructionSequence(
        start_fragment=[ApplyGate(gate_set["P"])],
        repeatable_fragment=[ApplyGate(gate_set["L0"])],
        end_fragment=[ApplyGate(gate_set["M"])],
        depth=4,
    )

    assert not seq0.is_mergeable_with(seq1)

    # None vs int also not mergeable
    seq2 = InstructionSequence(
        start_fragment=[ApplyGate(gate_set["P"])],
        repeatable_fragment=[ApplyGate(gate_set["L0"])],
        end_fragment=[ApplyGate(gate_set["M"])],
    )

    assert not seq0.is_mergeable_with(seq2)


def test_merge(gate_set):
    """Test merging of instruction sequences."""

    seq0 = InstructionSequence(
        start_fragment=[
            ApplyGate(gate_set["P"]),
            PartialPauliPermutation.from_sets([{("Y", "Z")}, set(), {("Z", "Y")}]),
        ],
        repeatable_fragment=[ApplyGate(gate_set["L0"])],
        end_fragment=[
            PartialPauliPermutation.from_sets([{("Y", "X")}, set(), {("Y", "Z")}]),
            ApplyGate(gate_set["M"]),
        ],
    )
    seq1 = InstructionSequence(
        start_fragment=[
            ApplyGate(gate_set["P"]),
            PartialPauliPermutation.from_sets([{("X", "Y")}, {("Y", "Z")}, {("Z", "Y")}]),
        ],
        repeatable_fragment=[ApplyGate(gate_set["L0"])],
        end_fragment=[
            PartialPauliPermutation.from_sets([{("X", "Z")}, set(), {("Y", "Z")}]),
            ApplyGate(gate_set["M"]),
        ],
    )

    seq2 = seq0.merge(seq1)
    expected = InstructionSequence(
        start_fragment=[
            ApplyGate(gate_set["P"]),
            PartialPauliPermutation.from_sets(
                [{("X", "Y"), ("Y", "Z")}, {("Y", "Z")}, {("Z", "Y")}]
            ),
        ],
        repeatable_fragment=[ApplyGate(gate_set["L0"])],
        end_fragment=[
            PartialPauliPermutation.from_sets([{("Y", "X"), ("Z", "Y")}, set(), {("Y", "Z")}]),
            ApplyGate(gate_set["M"]),
        ],
    )
    assert seq2 == expected


def test_merge_failures(gate_set):
    """Test merging of instruction sequence failures."""

    # inconsistent lengths
    seq0 = InstructionSequence(
        start_fragment=[ApplyGate(gate_set["P"])],
        repeatable_fragment=[],
        end_fragment=[ApplyGate(gate_set["M"])],
    )
    seq1 = InstructionSequence(
        start_fragment=[ApplyGate(gate_set["P"]), ApplyGate(gate_set["L0"])],
        repeatable_fragment=[],
        end_fragment=[ApplyGate(gate_set["M"])],
    )
    with pytest.raises(ValueError, match="start fragments of different lengths"):
        seq0.merge(seq1)

    # inconsistent gate labels
    seq0 = InstructionSequence(
        start_fragment=[ApplyGate(gate_set["P"]), ApplyGate(gate_set["L0"])],
        repeatable_fragment=[],
        end_fragment=[ApplyGate(gate_set["M"])],
    )
    seq1 = InstructionSequence(
        start_fragment=[ApplyGate(gate_set["P"]), ApplyGate(gate_set["L1"])],
        repeatable_fragment=[],
        end_fragment=[ApplyGate(gate_set["M"])],
    )
    with pytest.raises(ValueError, match="Cannot merge ApplyGate instructions"):
        seq0.merge(seq1)

    # inconsistent partial permutations
    seq0 = InstructionSequence(
        start_fragment=[
            ApplyGate(gate_set["P"]),
            PartialPauliPermutation.from_sets([{("X", "X")}]),
        ],
        repeatable_fragment=[],
        end_fragment=[ApplyGate(gate_set["M"])],
    )
    seq1 = InstructionSequence(
        start_fragment=[
            ApplyGate(gate_set["P"]),
            PartialPauliPermutation.from_sets([{("X", "Y")}]),
        ],
        repeatable_fragment=[],
        end_fragment=[ApplyGate(gate_set["M"])],
    )
    with pytest.raises(ValueError, match="Cannot merge inconsistent partial permutations"):
        seq0.merge(seq1)

    # depth mismatch
    seq0 = InstructionSequence(
        start_fragment=[ApplyGate(gate_set["P"])],
        repeatable_fragment=[],
        end_fragment=[ApplyGate(gate_set["M"])],
        depth=3,
    )
    seq1 = InstructionSequence(
        start_fragment=[ApplyGate(gate_set["P"])],
        repeatable_fragment=[],
        end_fragment=[ApplyGate(gate_set["M"])],
        depth=4,
    )
    with pytest.raises(ValueError, match="different depths"):
        seq0.merge(seq1)


def test_complete(gate_set):
    """Test InstructionSequence.complete."""

    start_permutation = PartialPauliPermutation.from_sets([{("Z", "X")}, {("X", "Y")}])
    repeatable_permutation = PartialPauliPermutation.from_sets([{("Y", "Z")}, set()])
    end_permutation = PartialPauliPermutation.from_sets([{("X", "X")}, {("X", "Y"), ("Y", "Z")}])

    seq = InstructionSequence(
        start_fragment=[ApplyGate(gate_set["P"]), start_permutation],
        repeatable_fragment=[ApplyGate(gate_set["L0"]), repeatable_permutation],
        end_fragment=[end_permutation, ApplyGate(gate_set["M"])],
    )

    expected = InstructionSequence(
        start_fragment=[
            ApplyGate(gate_set["P"]),
            start_permutation.complete(),
        ],
        repeatable_fragment=[
            ApplyGate(gate_set["L0"]),
            repeatable_permutation.complete(),
        ],
        end_fragment=[
            end_permutation.complete(),
            ApplyGate(gate_set["M"]),
        ],
    )

    assert expected == seq.complete()


def test_complete_preserves_depth(gate_set):
    """Test that complete() preserves the depth."""

    seq = InstructionSequence(
        start_fragment=[ApplyGate(gate_set["P"])],
        repeatable_fragment=[ApplyGate(gate_set["L0"])],
        end_fragment=[ApplyGate(gate_set["M"])],
        depth=7,
    )

    assert seq.complete().depth == 7


def test_has_same_structure_as(model_gate_set_1q):
    """Test has_same_structure_as for InstructionSequence."""

    seq0 = InstructionSequence(
        start_fragment=[
            ApplyGate(model_gate_set_1q["P"]),
            PartialPauliPermutation.from_sets([{("X", "Y")}]),
        ],
        repeatable_fragment=[
            ApplyGate(model_gate_set_1q["L0"]),
            ApplyGate(model_gate_set_1q["L1"]),
        ],
        end_fragment=[
            PartialPauliPermutation.from_sets([{("Y", "Z")}]),
            ApplyGate(model_gate_set_1q["M"]),
        ],
    )

    # same structure with different PartialPauliPermutations
    seq1 = InstructionSequence(
        start_fragment=[
            ApplyGate(model_gate_set_1q["P"]),
            PartialPauliPermutation.from_sets([{("Z", "X")}]),
        ],
        repeatable_fragment=[
            ApplyGate(model_gate_set_1q["L0"]),
            ApplyGate(model_gate_set_1q["L1"]),
        ],
        end_fragment=[
            PartialPauliPermutation.from_sets([{("X", "Y")}]),
            ApplyGate(model_gate_set_1q["M"]),
        ],
    )
    assert seq0.has_same_structure_as(seq1)

    # different gate label in repeatable_fragment
    seq2 = InstructionSequence(
        start_fragment=[
            ApplyGate(model_gate_set_1q["P"]),
            PartialPauliPermutation.from_sets([{("X", "Y")}]),
        ],
        repeatable_fragment=[
            ApplyGate(model_gate_set_1q["L0"]),
            ApplyGate(model_gate_set_1q["L0"]),
        ],
        end_fragment=[
            PartialPauliPermutation.from_sets([{("Y", "Z")}]),
            ApplyGate(model_gate_set_1q["M"]),
        ],
    )
    assert not seq0.has_same_structure_as(seq2)

    # different gate label in start_fragment
    seq3 = InstructionSequence(
        start_fragment=[
            ApplyGate(model_gate_set_1q["L0"]),
            PartialPauliPermutation.from_sets([{("X", "Y")}]),
        ],
        repeatable_fragment=[
            ApplyGate(model_gate_set_1q["L0"]),
            ApplyGate(model_gate_set_1q["L1"]),
        ],
        end_fragment=[
            PartialPauliPermutation.from_sets([{("Y", "Z")}]),
            ApplyGate(model_gate_set_1q["M"]),
        ],
    )
    assert not seq0.has_same_structure_as(seq3)

    # different number of gates in repeatable_fragment
    seq4 = InstructionSequence(
        start_fragment=[
            ApplyGate(model_gate_set_1q["P"]),
            PartialPauliPermutation.from_sets([{("X", "Y")}]),
        ],
        repeatable_fragment=[model_gate_set_1q["L0"]],
        end_fragment=[
            PartialPauliPermutation.from_sets([{("Y", "Z")}]),
            ApplyGate(model_gate_set_1q["M"]),
        ],
    )
    assert not seq0.has_same_structure_as(seq4)


def test_has_same_structure_as_depth(model_gate_set_1q):
    """Test that has_same_structure_as requires matching depths."""

    seq0 = InstructionSequence(
        start_fragment=[ApplyGate(model_gate_set_1q["P"])],
        repeatable_fragment=[ApplyGate(model_gate_set_1q["L0"])],
        end_fragment=[ApplyGate(model_gate_set_1q["M"])],
        depth=3,
    )
    seq1 = InstructionSequence(
        start_fragment=[ApplyGate(model_gate_set_1q["P"])],
        repeatable_fragment=[ApplyGate(model_gate_set_1q["L0"])],
        end_fragment=[ApplyGate(model_gate_set_1q["M"])],
        depth=4,
    )
    assert not seq0.has_same_structure_as(seq1)

    seq2 = InstructionSequence(
        start_fragment=[ApplyGate(model_gate_set_1q["P"])],
        repeatable_fragment=[ApplyGate(model_gate_set_1q["L0"])],
        end_fragment=[ApplyGate(model_gate_set_1q["M"])],
        depth=3,
    )
    assert seq0.has_same_structure_as(seq2)


def test_bind_at(gate_set):
    """Test bind_at returns a new instance with the specified depth."""

    start_fragment = [ApplyGate(gate_set["P"])]
    repeatable_fragment = [ApplyGate(gate_set["L0"]), ApplyGate(gate_set["L1"])]
    end_fragment = [ApplyGate(gate_set["M"])]

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
