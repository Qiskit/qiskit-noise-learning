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
    InstructionPattern,
    InstructionSequence,
    PartialPauliPermutation,
)


@pytest.fixture
def model_gate_set() -> ModelGateSet:
    model_gate_set = ModelGateSet(1)
    ident = [((0,), Clifford(QuantumCircuit(1)))]
    model_gate_set.add_gate(ModelGate("P", ident, prep_idxs=range(1)))
    model_gate_set.add_gate(ModelGate("M", ident, meas_idxs=range(1)))
    model_gate_set.add_gate(ModelGate("L0", ident))
    model_gate_set.add_gate(ModelGate("L1", ident))
    return model_gate_set


def test_construction(model_gate_set):
    """Test construction and attributes."""

    start_fragment = [ApplyGate(model_gate_set["P"])]
    repeatable_fragment = [ApplyGate(model_gate_set["L0"]), ApplyGate(model_gate_set["L1"])]
    end_fragment = [ApplyGate(model_gate_set["M"])]

    pattern = InstructionPattern(
        start_fragment=start_fragment,
        repeatable_fragment=repeatable_fragment,
        end_fragment=end_fragment,
    )

    sequence = InstructionSequence(pattern=pattern, depth=3)

    assert sequence.pattern == pattern
    assert sequence.depth == 3
    assert len(sequence) == 8


def test_is_mergeable_with(model_gate_set):
    """Test mergeability checking for InstructionSequence."""
    pattern0 = InstructionPattern(
        start_fragment=[
            ApplyGate(model_gate_set["P"]),
            PartialPauliPermutation.from_sets([{("X", "Y"), ("Y", "Z")}, set(), {("Z", "Y")}]),
        ],
        repeatable_fragment=[ApplyGate(model_gate_set["L0"])],
        end_fragment=[
            PartialPauliPermutation.from_sets([{("Y", "X"), ("Z", "Y")}, set(), {("Y", "Z")}]),
            ApplyGate(model_gate_set["M"]),
        ],
    )
    sequence0 = InstructionSequence(pattern=pattern0, depth=5)

    assert not sequence0.is_mergeable_with(
        InstructionSequence(pattern=InstructionPattern([], [], []), depth=5)
    )
    # different gate labels
    assert not sequence0.is_mergeable_with(
        InstructionSequence(
            pattern=InstructionPattern(
                start_fragment=[
                    ApplyGate(model_gate_set["P"]),
                    PartialPauliPermutation.from_sets(
                        [{("X", "Y"), ("Y", "Z")}, set(), {("Z", "Y")}]
                    ),
                ],
                repeatable_fragment=[ApplyGate(model_gate_set["L1"])],
                end_fragment=[
                    PartialPauliPermutation.from_sets(
                        [{("Y", "X"), ("Z", "Y")}, set(), {("Y", "Z")}]
                    ),
                    ApplyGate(model_gate_set["M"]),
                ],
            ),
            depth=5,
        )
    )
    # incompatible permutations
    assert not sequence0.is_mergeable_with(
        InstructionSequence(
            pattern=InstructionPattern(
                start_fragment=[
                    ApplyGate(model_gate_set["P"]),
                    PartialPauliPermutation.from_sets(
                        [{("Y", "X"), ("Z", "Y")}, set(), {("Z", "Y")}]
                    ),
                ],
                repeatable_fragment=[ApplyGate(model_gate_set["L0"])],
                end_fragment=[
                    PartialPauliPermutation.from_sets(
                        [{("Y", "X"), ("Z", "Y")}, set(), {("Y", "Z")}]
                    ),
                    ApplyGate(model_gate_set["M"]),
                ],
            ),
            depth=5,
        )
    )
    # compatible permutations
    assert sequence0.is_mergeable_with(
        InstructionSequence(
            pattern=InstructionPattern(
                start_fragment=[
                    ApplyGate(model_gate_set["P"]),
                    PartialPauliPermutation.from_sets([{("X", "Y")}, {("Y", "Z")}, {("Z", "Y")}]),
                ],
                repeatable_fragment=[ApplyGate(model_gate_set["L0"])],
                end_fragment=[
                    PartialPauliPermutation.from_sets([{("X", "Z")}, set(), {("Y", "Z")}]),
                    ApplyGate(model_gate_set["M"]),
                ],
            ),
            depth=5,
        )
    )

    # incompatible depths
    assert not sequence0.is_mergeable_with(
        InstructionSequence(
            pattern=InstructionPattern(
                start_fragment=[
                    ApplyGate(model_gate_set["P"]),
                    PartialPauliPermutation.from_sets([{("X", "Y")}, {("Y", "Z")}, {("Z", "Y")}]),
                ],
                repeatable_fragment=[ApplyGate(model_gate_set["L0"])],
                end_fragment=[
                    PartialPauliPermutation.from_sets([{("X", "Z")}, set(), {("Y", "Z")}]),
                    ApplyGate(model_gate_set["M"]),
                ],
            ),
            depth=4,
        )
    )
