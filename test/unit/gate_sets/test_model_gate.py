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

import pytest
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Clifford, Pauli, PhasedQubitSparsePauli, QubitSparsePauli

from qiskit_noise_learning.gate_sets import ModelGate


def test_construction_empty():
    gate = ModelGate("L0", [], [0, 1, 2, 5])

    assert gate.name == "L0"
    assert gate.clifford == Clifford(QuantumCircuit(6))
    assert gate.num_qubits == 4
    assert gate.meas_idxs == set()
    assert gate.prep_idxs == set()
    assert gate.qubit_idxs == (0, 1, 2, 5)
    assert gate.idling_idxs == {0, 1, 2, 5}
    assert gate.gate_idxs == set()
    assert list(gate.constituent_gate_idxs) == []


def test_construction():
    clifford = [((0, 1, 2, 5), Clifford(QuantumCircuit(4)))]
    gate = ModelGate("L0", clifford, [0, 1, 2, 5])

    assert gate.name == "L0"
    assert gate.clifford == Clifford(QuantumCircuit(6))
    assert gate.num_qubits == 4
    assert gate.meas_idxs == set()
    assert gate.prep_idxs == set()
    assert gate.qubit_idxs == (0, 1, 2, 5)
    assert gate.idling_idxs == set()
    assert gate.gate_idxs == {0, 1, 2, 5}
    assert list(gate.constituent_gate_idxs) == [(0, 1, 2, 5)]


def test_constructor_raises():
    with pytest.raises(ValueError, match="At least"):
        ModelGate("L0")

    with pytest.raises(ValueError, match="all the Clifford indices."):
        ModelGate("L0", [((0, 1), Clifford(QuantumCircuit(2)))], [0])

    with pytest.raises(ValueError, match="`prep_idxs` must be a subset of `qubit_idxs`"):
        ModelGate("L0", [((0, 1), Clifford(QuantumCircuit(2)))], [0, 1], prep_idxs=[1, 2])


def test_equality():
    identity3 = [((0, 1, 2), Clifford(QuantumCircuit(3)))]
    gate1 = ModelGate("L0", identity3, [0, 1, 2])
    gate2 = ModelGate("L0", identity3, [0, 1, 2], meas_idxs=[0])
    gate3 = ModelGate("L0", [((0, 1), Clifford(QuantumCircuit(2)))])
    c4 = QuantumCircuit(3)
    c4.cx(0, 1)
    c4 = [((0, 1, 2), Clifford(c4))]
    gate4 = ModelGate("L0", c4)

    assert gate2 == ModelGate("L0", identity3, [0, 1, 2], meas_idxs=[0])
    assert gate2 != ModelGate("L1", identity3, [0, 1, 2], meas_idxs=[0])
    assert gate4 == ModelGate("L0", c4, [0, 1, 2])
    assert gate1 != gate2

    # all of the gates are not equal, so equality should be equivalent to identifier equality
    for g1, g2 in product([gate1, gate2, gate3, gate4], repeat=2):
        assert (g1 == g2) == (g1 is g2), f"{g1} vs {g2} checking failed"


def test_clifford_under_permutations():
    circuit1 = QuantumCircuit(4)
    circuit1.cx(0, 1)
    circuit1.cx(2, 3)
    circuit1.cx(1, 2)
    clifford1 = Clifford(circuit1)
    gate1 = ModelGate("L0", [((3, 9, 8, 2), clifford1)])

    circuit2 = QuantumCircuit(4)
    circuit2.cx(2, 1)
    circuit2.cx(3, 0)
    circuit2.cx(1, 3)
    clifford2 = Clifford(circuit2)
    gate2 = ModelGate("L0", [((2, 9, 3, 8), clifford2)])

    assert gate1 != gate2
    assert gate1.clifford == gate2.clifford


def test_clifford_propagate():
    # test for phaseless qubit sparse paulis
    # Clifford maps X -> -Y, Y -> Z, Z -> -X
    gate = ModelGate("L0", [((0,), Clifford([[True, True, True], [True, False, True]]))])
    assert QubitSparsePauli("X") == gate.clifford_propagate(QubitSparsePauli("Z"))
    assert QubitSparsePauli("X") == gate.clifford_propagate(QubitSparsePauli("Y"), inverse=True)

    # test phased propagation of qubit sparse paulis
    assert PhasedQubitSparsePauli(Pauli("-X")) == gate.clifford_propagate(
        PhasedQubitSparsePauli("Z")
    )
    assert PhasedQubitSparsePauli(Pauli("-X")) == gate.clifford_propagate(
        PhasedQubitSparsePauli("Y"), inverse=True
    )

    # test on subset
    assert PhasedQubitSparsePauli(Pauli("-IX")) == gate.clifford_propagate(
        PhasedQubitSparsePauli("IZ")
    )

    gate = ModelGate("L0", [((1,), Clifford([[True, True, True], [True, False, True]]))])
    assert PhasedQubitSparsePauli(Pauli("-XI")) == gate.clifford_propagate(
        PhasedQubitSparsePauli("ZI")
    )


def test_hash():
    """Ensure hash call works."""
    assert isinstance(hash(ModelGate(name="gate_name", qubit_idxs=[1, 2, 3])), int)


def test_latex_str():
    # falls back to the name when not provided
    assert ModelGate("L0", qubit_idxs=[0]).latex_str == "L0"

    # uses the explicit value when provided
    assert ModelGate("L0", qubit_idxs=[0], latex_str=r"\Lambda_0").latex_str == r"\Lambda_0"
