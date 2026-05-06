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
from qiskit.quantum_info import Clifford, QubitSparsePauli

from qiskit_noise_learning.gate_sets import ModelGate, ModelGateSet
from qiskit_noise_learning.sequences import FidelityIndex, InstructionSequence, Path, PathPattern


def test_construction():
    """Test construction and attributes."""

    ident = [((0, 1), Clifford(QuantumCircuit(2)))]
    prep = ModelGate("P", ident, qubit_idxs=range(2), prep_idxs=range(2))
    gate = ModelGate("L0", ident, qubit_idxs=range(2))
    meas = ModelGate("M", ident, qubit_idxs=range(2), meas_idxs=range(2))

    start_fragment = [
        FidelityIndex(prep, pauli=QubitSparsePauli("II"), out_bit_indices=frozenset(range(2)))
    ]
    repeatable_fragment = [FidelityIndex(gate, pauli=QubitSparsePauli("IX"))] * 2
    end_fragment = [
        FidelityIndex(meas, pauli=QubitSparsePauli("II"), in_bit_indices=frozenset(range(2)))
    ]

    path_pattern = PathPattern(
        start_fragment=start_fragment,
        repeatable_fragment=repeatable_fragment,
        end_fragment=end_fragment,
    )

    path = Path(pattern=path_pattern, depth=3)

    assert path.pattern == path_pattern
    assert path.depth == 3
    assert len(path) == 8


def test_iter():
    """Test __iter__ yields elements in the correct order."""

    ident = [((0, 1), Clifford(QuantumCircuit(2)))]
    prep = ModelGate("S", ident, qubit_idxs=range(2), prep_idxs=range(2))
    gate0 = ModelGate("G0", ident, qubit_idxs=range(2))
    gate1 = ModelGate("G1", ident, qubit_idxs=range(2))
    meas = ModelGate("M", ident, qubit_idxs=range(2), meas_idxs=range(2))

    start_fragment = [
        FidelityIndex(prep, pauli=QubitSparsePauli("II"), out_bit_indices=frozenset(range(2)))
    ]
    repeatable_fragment = [
        FidelityIndex(gate0, pauli=QubitSparsePauli("IX")),
        FidelityIndex(gate1, pauli=QubitSparsePauli("IY")),
    ]
    end_fragment = [
        FidelityIndex(meas, pauli=QubitSparsePauli("II"), in_bit_indices=frozenset(range(2)))
    ]

    path_pattern = PathPattern(
        start_fragment=start_fragment,
        repeatable_fragment=repeatable_fragment,
        end_fragment=end_fragment,
    )

    path = Path(pattern=path_pattern, depth=2)
    items = list(path)

    assert len(items) == 6
    assert items[0] == start_fragment[0]
    assert items[1] == repeatable_fragment[0]
    assert items[2] == repeatable_fragment[1]
    assert items[3] == repeatable_fragment[0]
    assert items[4] == repeatable_fragment[1]
    assert items[5] == end_fragment[0]


def test_getitem():
    """Test __getitem__ returns correct elements by index."""

    ident = [((0, 1), Clifford(QuantumCircuit(2)))]
    prep0 = ModelGate("S0", ident, qubit_idxs=range(2), prep_idxs=range(2))
    prep1 = ModelGate("S1", ident, qubit_idxs=range(2), prep_idxs=range(2))
    gate0 = ModelGate("G0", ident, qubit_idxs=range(2))
    gate1 = ModelGate("G1", ident, qubit_idxs=range(2))
    meas = ModelGate("M", ident, qubit_idxs=range(2), meas_idxs=range(2))

    start_fragment = [
        FidelityIndex(prep0, pauli=QubitSparsePauli("II"), out_bit_indices=frozenset(range(2))),
        FidelityIndex(prep1, pauli=QubitSparsePauli("II"), out_bit_indices=frozenset(range(2))),
    ]
    repeatable_fragment = [
        FidelityIndex(gate0, pauli=QubitSparsePauli("IX")),
        FidelityIndex(gate1, pauli=QubitSparsePauli("IY")),
    ]
    end_fragment = [
        FidelityIndex(meas, pauli=QubitSparsePauli("II"), in_bit_indices=frozenset(range(2)))
    ]

    path_pattern = PathPattern(
        start_fragment=start_fragment,
        repeatable_fragment=repeatable_fragment,
        end_fragment=end_fragment,
    )

    path = Path(pattern=path_pattern, depth=3)

    # start fragment
    assert path[0] == start_fragment[0]
    assert path[1] == start_fragment[1]
    # repeatable fragment (3 repetitions)
    assert path[2] == repeatable_fragment[0]
    assert path[3] == repeatable_fragment[1]
    assert path[4] == repeatable_fragment[0]
    assert path[5] == repeatable_fragment[1]
    assert path[6] == repeatable_fragment[0]
    assert path[7] == repeatable_fragment[1]
    # end fragment
    assert path[8] == end_fragment[0]


def test_getitem_out_of_bounds():
    """Test __getitem__ raises IndexError for out of bounds indices."""

    ident = [((0, 1), Clifford(QuantumCircuit(2)))]
    prep = ModelGate("S", ident, qubit_idxs=range(2), prep_idxs=range(2))
    gate = ModelGate("G", ident, qubit_idxs=range(2))
    meas = ModelGate("M", ident, qubit_idxs=range(2), meas_idxs=range(2))

    start_fragment = [
        FidelityIndex(prep, pauli=QubitSparsePauli("II"), out_bit_indices=frozenset(range(2)))
    ]
    repeatable_fragment = [FidelityIndex(gate, pauli=QubitSparsePauli("IX"))]
    end_fragment = [
        FidelityIndex(meas, pauli=QubitSparsePauli("II"), in_bit_indices=frozenset(range(2)))
    ]

    path_pattern = PathPattern(
        start_fragment=start_fragment,
        repeatable_fragment=repeatable_fragment,
        end_fragment=end_fragment,
    )

    path = Path(pattern=path_pattern, depth=2)
    assert len(path) == 4

    with pytest.raises(IndexError, match="exceeds length"):
        path[4]

    with pytest.raises(IndexError, match="negative"):
        path[-1]


def test_to_instruction_sequence():
    gate_set_1q = ModelGateSet(1)
    ident = [((0,), Clifford(QuantumCircuit(1)))]
    gate_set_1q.add_gate(ModelGate("P", ident, prep_idxs=range(1)))
    gate_set_1q.add_gate(ModelGate("M", ident, meas_idxs=range(1)))
    # Clifford maps X -> -Y, Y -> Z, Z -> -X
    gate_set_1q.add_gate(
        ModelGate("L0", [((0,), Clifford([[True, True, True], [True, False, True]]))])
    )

    pattern = PathPattern(
        start_fragment=[
            FidelityIndex.from_transition(
                gate_set_1q["P"], QubitSparsePauli("I"), QubitSparsePauli("Z")
            )
        ],
        repeatable_fragment=[
            FidelityIndex.from_transition(
                gate_set_1q["L0"], QubitSparsePauli("X"), QubitSparsePauli("Y")
            )
        ],
        end_fragment=[
            FidelityIndex.from_transition(
                gate_set_1q["M"], QubitSparsePauli("Z"), QubitSparsePauli("I")
            )
        ],
    )

    path = Path(pattern=pattern, depth=5)

    assert path.to_instruction_sequence() == InstructionSequence(
        pattern=pattern.to_instruction_pattern(), depth=5
    )


def test_hash():
    prep = ModelGate("P", [], qubit_idxs=range(2), prep_idxs=range(2))
    gate = ModelGate("L0", [], qubit_idxs=range(2))
    meas = ModelGate("M", [], qubit_idxs=range(2), meas_idxs=range(2))

    start_fragment = [
        FidelityIndex(prep, pauli=QubitSparsePauli("II"), out_bit_indices=frozenset(range(2)))
    ]
    repeatable_fragment = [FidelityIndex(gate, pauli=QubitSparsePauli("IX"))] * 2
    end_fragment = [
        FidelityIndex(meas, pauli=QubitSparsePauli("II"), in_bit_indices=frozenset(range(2)))
    ]

    path = Path(
        PathPattern(
            start_fragment=start_fragment,
            repeatable_fragment=repeatable_fragment,
            end_fragment=end_fragment,
        ),
        5,
    )
    assert isinstance(hash(path), int)
