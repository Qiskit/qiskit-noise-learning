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
from qiskit_noise_learning.sequences import ApplyGate, InstructionPattern, PartialPauliPermutation


@pytest.fixture()
def gate_set():
    model_gate_set = ModelGateSet(3)
    ident = [((0, 1, 2), Clifford(QuantumCircuit(3)))]
    model_gate_set.add_gate(ModelGate("P", ident, prep_idxs=range(3)))
    model_gate_set.add_gate(ModelGate("M", ident, meas_idxs=range(3)))
    model_gate_set.add_gate(ModelGate("L0", ident))
    model_gate_set.add_gate(ModelGate("L1", ident))
    return model_gate_set


def test_construction(gate_set):
    """Test construction and attributes."""

    start_fragment = [ApplyGate(gate_set["P"])]
    repeatable_fragment = [ApplyGate(gate_set["L0"]), ApplyGate(gate_set["L1"])]
    end_fragment = [ApplyGate(gate_set["M"])]

    path_pattern = InstructionPattern(
        start_fragment=start_fragment,
        repeatable_fragment=repeatable_fragment,
        end_fragment=end_fragment,
    )

    assert path_pattern.start_fragment == start_fragment
    assert path_pattern.repeatable_fragment == repeatable_fragment
    assert path_pattern.end_fragment == end_fragment


def test_is_mergeable_with(gate_set):
    """Test mergeability checking for InstructionPattern."""

    pattern0 = InstructionPattern(
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

    assert not pattern0.is_mergeable_with(InstructionPattern([], [], []))
    # different gate labels
    assert not pattern0.is_mergeable_with(
        InstructionPattern(
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
    assert not pattern0.is_mergeable_with(
        InstructionPattern(
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
    assert pattern0.is_mergeable_with(
        InstructionPattern(
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


def test_merge(gate_set):
    """Test merging of instruction patterns."""

    pattern0 = InstructionPattern(
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
    pattern1 = InstructionPattern(
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

    pattern2 = pattern0.merge(pattern1)
    expected = InstructionPattern(
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
    print(pattern2)
    print(expected)
    assert pattern2 == expected


def test_merge_failures(gate_set):
    """Test merging of instruction pattern failures."""

    # inconsistent lengths
    pattern0 = InstructionPattern(
        start_fragment=[ApplyGate(gate_set["P"])],
        repeatable_fragment=[],
        end_fragment=[ApplyGate(gate_set["M"])],
    )
    pattern1 = InstructionPattern(
        start_fragment=[ApplyGate(gate_set["P"]), ApplyGate(gate_set["L0"])],
        repeatable_fragment=[],
        end_fragment=[ApplyGate(gate_set["M"])],
    )
    with pytest.raises(ValueError, match="start_fragments of different lengths"):
        pattern0.merge(pattern1)

    # inconsistent gate labels
    pattern0 = InstructionPattern(
        start_fragment=[ApplyGate(gate_set["P"]), ApplyGate(gate_set["L0"])],
        repeatable_fragment=[],
        end_fragment=[ApplyGate(gate_set["M"])],
    )
    pattern1 = InstructionPattern(
        start_fragment=[ApplyGate(gate_set["P"]), ApplyGate(gate_set["L1"])],
        repeatable_fragment=[],
        end_fragment=[ApplyGate(gate_set["M"])],
    )
    with pytest.raises(ValueError, match="Cannot merge ApplyGate instructions"):
        pattern0.merge(pattern1)

    # inconsistent partial permutations
    pattern0 = InstructionPattern(
        start_fragment=[
            ApplyGate(gate_set["P"]),
            PartialPauliPermutation.from_sets([{("X", "X")}]),
        ],
        repeatable_fragment=[],
        end_fragment=[ApplyGate(gate_set["M"])],
    )
    pattern1 = InstructionPattern(
        start_fragment=[
            ApplyGate(gate_set["P"]),
            PartialPauliPermutation.from_sets([{("X", "Y")}]),
        ],
        repeatable_fragment=[],
        end_fragment=[ApplyGate(gate_set["M"])],
    )
    with pytest.raises(ValueError, match="Cannot merge inconsistent partial permutations"):
        pattern0.merge(pattern1)


def test_complete(gate_set):
    """Test InstructionPattern.complete."""

    start_permutation = PartialPauliPermutation.from_sets([{("Z", "X")}, {("X", "Y")}])
    repeatable_permutation = PartialPauliPermutation.from_sets([{("Y", "Z")}, set()])
    end_permutation = PartialPauliPermutation.from_sets([{("X", "X")}, {("X", "Y"), ("Y", "Z")}])

    pattern = InstructionPattern(
        start_fragment=[ApplyGate(gate_set["P"]), start_permutation],
        repeatable_fragment=[ApplyGate(gate_set["L0"]), repeatable_permutation],
        end_fragment=[end_permutation, ApplyGate(gate_set["M"])],
    )

    expected = InstructionPattern(
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

    assert expected == pattern.complete()


def test_has_same_structure_as():
    """Test has_same_structure_as for InstructionPattern."""

    model_gate_set = ModelGateSet(1)
    ident = [((0,), Clifford(QuantumCircuit(1)))]
    model_gate_set.add_gate(ModelGate("P", ident, prep_idxs=range(1)))
    model_gate_set.add_gate(ModelGate("M", ident, meas_idxs=range(1)))
    model_gate_set.add_gate(ModelGate("L0", ident))
    model_gate_set.add_gate(ModelGate("L1", ident))

    pattern0 = InstructionPattern(
        start_fragment=[
            ApplyGate(model_gate_set["P"]),
            PartialPauliPermutation.from_sets([{("X", "Y")}]),
        ],
        repeatable_fragment=[ApplyGate(model_gate_set["L0"]), ApplyGate(model_gate_set["L1"])],
        end_fragment=[
            PartialPauliPermutation.from_sets([{("Y", "Z")}]),
            ApplyGate(model_gate_set["M"]),
        ],
    )

    # same structure with different PartialPauliPermutations
    pattern1 = InstructionPattern(
        start_fragment=[
            ApplyGate(model_gate_set["P"]),
            PartialPauliPermutation.from_sets([{("Z", "X")}]),
        ],
        repeatable_fragment=[ApplyGate(model_gate_set["L0"]), ApplyGate(model_gate_set["L1"])],
        end_fragment=[
            PartialPauliPermutation.from_sets([{("X", "Y")}]),
            ApplyGate(model_gate_set["M"]),
        ],
    )
    assert pattern0.has_same_structure_as(pattern1)

    # different gate label in repeatable_fragment
    pattern2 = InstructionPattern(
        start_fragment=[
            ApplyGate(model_gate_set["P"]),
            PartialPauliPermutation.from_sets([{("X", "Y")}]),
        ],
        repeatable_fragment=[ApplyGate(model_gate_set["L0"]), ApplyGate(model_gate_set["L0"])],
        end_fragment=[
            PartialPauliPermutation.from_sets([{("Y", "Z")}]),
            ApplyGate(model_gate_set["M"]),
        ],
    )
    assert not pattern0.has_same_structure_as(pattern2)

    # different gate label in start_fragment
    pattern3 = InstructionPattern(
        start_fragment=[
            ApplyGate(model_gate_set["L0"]),
            PartialPauliPermutation.from_sets([{("X", "Y")}]),
        ],
        repeatable_fragment=[ApplyGate(model_gate_set["L0"]), ApplyGate(model_gate_set["L1"])],
        end_fragment=[
            PartialPauliPermutation.from_sets([{("Y", "Z")}]),
            ApplyGate(model_gate_set["M"]),
        ],
    )
    assert not pattern0.has_same_structure_as(pattern3)

    # different number of gates in repeatable_fragment
    pattern4 = InstructionPattern(
        start_fragment=[
            ApplyGate(model_gate_set["P"]),
            PartialPauliPermutation.from_sets([{("X", "Y")}]),
        ],
        repeatable_fragment=[model_gate_set["L0"]],
        end_fragment=[
            PartialPauliPermutation.from_sets([{("Y", "Z")}]),
            ApplyGate(model_gate_set["M"]),
        ],
    )
    assert not pattern0.has_same_structure_as(pattern4)
