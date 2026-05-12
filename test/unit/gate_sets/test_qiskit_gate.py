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
from qiskit.circuit import Annotation, QuantumCircuit
from qiskit.quantum_info import Clifford
from samplomatic import Tag

from qiskit_noise_learning.gate_sets import ModelGate, QiskitGate


def test_construction():
    circuit = QuantumCircuit(4)
    gate = QiskitGate("L0", circuit, [0, 1, 2, 5])

    assert gate.name == "L0"
    assert gate.circuit == circuit
    assert gate.num_qubits == 4
    assert gate.meas_idxs == set()
    assert gate.prep_idxs == set()
    assert gate.qubit_idxs == (0, 1, 2, 5)
    assert gate.idling_idxs == {0, 1, 2, 5}
    assert gate.gate_idxs == set()
    assert list(gate.constituent_gate_idxs) == []
    assert Tag("L0") in gate.annotations


def test_construction_with_prep():
    circuit = QuantumCircuit(4)
    circuit.reset(3)
    gate = QiskitGate("L0", circuit, [0, 1, 2, 5], [1, 2])

    assert gate.name == "L0"
    assert gate.circuit == circuit
    assert gate.meas_idxs == set()
    assert gate.prep_idxs == {1, 2, 5}
    assert gate.qubit_idxs == (0, 1, 2, 5)
    assert gate.idling_idxs == {0}
    assert gate.gate_idxs == set()
    assert list(gate.constituent_gate_idxs) == []


def test_construction_with_meas():
    circuit = QuantumCircuit(4, 2)
    circuit.measure([3, 1], range(2))
    gate = QiskitGate("L0", circuit, [0, 1, 2, 5])

    assert gate.name == "L0"
    assert gate.circuit == circuit
    assert gate.meas_idxs == {1, 5}
    assert gate.prep_idxs == set()
    assert gate.qubit_idxs == (0, 1, 2, 5)
    assert gate.idling_idxs == {0, 2}
    assert gate.gate_idxs == set()
    assert list(gate.constituent_gate_idxs) == []


def test_construction_with_nonempty_circuit():
    circuit = QuantumCircuit(4)
    circuit.cx(2, 3)
    circuit.cx(1, 2)
    gate = QiskitGate("L", circuit, [3, 9, 8, 2], [9])

    assert gate.name == "L"
    assert gate.circuit == circuit
    assert gate.meas_idxs == set()
    assert gate.prep_idxs == {9}
    assert gate.qubit_idxs == (3, 9, 8, 2)
    assert gate.idling_idxs == {3}
    assert gate.gate_idxs == {9, 8, 2}
    assert sorted(gate.constituent_gate_idxs) == [(8, 2), (9, 8)]


def test_constructor_raises():
    with pytest.raises(ValueError, match="`qubit_idxs` .* `circuit.num_qubits`"):
        QiskitGate("L0", QuantumCircuit(2), [0])

    with pytest.raises(ValueError, match="`prep_idxs` must be a subset of `qubit_idxs`"):
        QiskitGate("L0", QuantumCircuit(2), [0, 1], [1, 2])

    with pytest.raises(ValueError, match="must include a ''Twirl''"):
        QiskitGate("L0", QuantumCircuit(2), [0, 1], annotations=[Annotation()])


def test_equality():
    gate1 = QiskitGate("L0", QuantumCircuit(3), [0, 1, 2])
    gate2 = QiskitGate("L0", QuantumCircuit(3), [0, 1, 2], [0])
    gate3 = QiskitGate("L0", QuantumCircuit(2), [0, 1])
    c4 = QuantumCircuit(3)
    c4.cx(0, 1)
    gate4 = QiskitGate("L0", c4, [0, 1, 2])

    assert gate2 == QiskitGate("L0", QuantumCircuit(3), [0, 1, 2], [0])
    assert gate2 != QiskitGate("L1", QuantumCircuit(3), [0, 1, 2], [0])
    assert gate4 == QiskitGate("L0", c4, [0, 1, 2])

    # all of the gates are not equal, so equality should be equivalent to idenifier equality
    for g1, g2 in product([gate1, gate2, gate3, gate4], repeat=2):
        assert (g1 == g2) == (g1 is g2), f"{g1} vs {g2} checking failed"


def test_equality_under_permutations():
    assert QiskitGate("L0", QuantumCircuit(3), [0, 1, 2]) == QiskitGate(
        "L0", QuantumCircuit(3), [2, 1, 0]
    )

    circuit1 = QuantumCircuit(4)
    circuit1.cx(0, 1)
    circuit1.cx(2, 3)
    circuit1.cx(1, 2)
    gate1 = QiskitGate("L0", circuit1, [3, 9, 8, 2])

    circuit2 = QuantumCircuit(4)
    circuit2.cx(2, 1)
    circuit2.cx(3, 0)
    circuit2.cx(1, 3)
    gate2 = QiskitGate("L0", circuit2, [2, 9, 3, 8])

    assert gate1 == gate2


def test_model_gate():
    qc = QuantumCircuit(4, 4)
    qc.cx(0, 1)
    qc.cx(2, 3)
    qc.cx(1, 2)
    gate = QiskitGate("L0", qc, qubit_idxs=[3, 9, 8, 2])

    clifford = [((3, 9, 8, 2), Clifford(qc))]
    assert gate.model_gate.clifford == ModelGate("L0", clifford).clifford

    # add measurements
    qc.measure([0, 1], [1, 2])
    gate = QiskitGate("L0", qc, qubit_idxs=[3, 9, 8, 2])
    expected_model_gate = ModelGate("L0", clifford, meas_idxs=[3, 9])
    assert gate.model_gate.clifford == expected_model_gate.clifford
    assert gate.model_gate.meas_idxs == expected_model_gate.meas_idxs

    # add resets
    qc.reset(3)
    gate = QiskitGate("L0", qc, qubit_idxs=[3, 9, 8, 2])
    expected_model_gate = ModelGate("L0", clifford, meas_idxs=[3, 9], prep_idxs=[2])
    assert gate.model_gate.clifford == expected_model_gate.clifford
    assert gate.model_gate.meas_idxs == expected_model_gate.meas_idxs
    assert gate.model_gate.prep_idxs == expected_model_gate.prep_idxs

    # intermixing operation types while still following ordering rules within the circuit
    qc = QuantumCircuit(4, 4)
    qc.cx(0, 1)
    qc.measure(0, 0)
    qc.cx(2, 3)
    qc.reset(3)
    qc.cx(1, 2)
    qc.reset(2)
    gate = QiskitGate("L0", qc, qubit_idxs=[3, 9, 8, 2])
    expected_model_gate = ModelGate("L0", clifford, meas_idxs=[3], prep_idxs=[2, 8])
    assert gate.model_gate.clifford == expected_model_gate.clifford
    assert gate.model_gate.meas_idxs == expected_model_gate.meas_idxs
    assert gate.model_gate.prep_idxs == expected_model_gate.prep_idxs


def test_draw():
    circuit = QuantumCircuit(2)
    circuit.cx(0, 1)
    gate = QiskitGate("cx", circuit, [12, 5])

    text = gate.draw(output="text")
    lines = str(text).split("\n")

    # Wire labels should show the virtual-to-physical mapping
    assert any("v_0 -> 12" in line for line in lines)
    assert any("v_1 -> 5" in line for line in lines)

    # Physical ordering: qubit 5 should appear above qubit 12
    line_5 = next(idx for idx, line in enumerate(lines) if "v_1 -> 5" in line)
    line_12 = next(idx for idx, line in enumerate(lines) if "v_0 -> 12" in line)
    assert line_5 < line_12

    # Idle wires should be hidden by default
    assert not any("idle" in line for line in lines)


def test_draw_with_measurements():
    circuit = QuantumCircuit(2, 2)
    circuit.cx(0, 1)
    circuit.measure(0, 0)
    circuit.measure(1, 1)
    gate = QiskitGate("meas_cx", circuit, [3, 7])

    text = gate.draw(output="text")
    lines = str(text).split("\n")

    assert any("v_0 -> 3" in line for line in lines)
    assert any("v_1 -> 7" in line for line in lines)
    # Classical register should be present
    assert any("c:" in line for line in lines)


def test_model_gate_raises():
    with pytest.raises(ValueError, match="two resets occur"):
        qc = QuantumCircuit(4, 4)
        qc.reset(2)
        qc.reset(2)
        QiskitGate("L0", circuit=qc, qubit_idxs=range(4)).model_gate

    with pytest.raises(ValueError, match="two measurements occur"):
        qc = QuantumCircuit(4, 4)
        qc.measure(2, 0)
        qc.measure(2, 0)
        QiskitGate("L0", circuit=qc, qubit_idxs=range(4)).model_gate

    with pytest.raises(ValueError, match="measurement occurs after reset"):
        qc = QuantumCircuit(4, 4)
        qc.reset(2)
        qc.measure(2, 0)
        QiskitGate("L0", circuit=qc, qubit_idxs=range(4)).model_gate

    with pytest.raises(ValueError, match="non-reset instruction occurs after a measurement"):
        qc = QuantumCircuit(4, 4)
        qc.measure(2, 0)
        qc.x(2)
        QiskitGate("L0", circuit=qc, qubit_idxs=range(4)).model_gate

    with pytest.raises(ValueError, match="instruction occurs after a reset"):
        qc = QuantumCircuit(4, 4)
        qc.reset(2)
        qc.x(2)
        QiskitGate("L0", circuit=qc, qubit_idxs=range(4)).model_gate
