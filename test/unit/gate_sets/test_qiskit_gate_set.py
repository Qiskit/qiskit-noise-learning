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
from qiskit.circuit import ClassicalRegister, Measure, QuantumCircuit, QuantumRegister
from qiskit.circuit.library import CZGate
from qiskit.transpiler import CouplingMap, Target
from samplomatic import Twirl

from qiskit_noise_learning.gate_sets import ModelGate, QiskitGate, QiskitGateSet


@pytest.fixture()
def target_4q():
    target = Target()
    cz_properties = {(0, 1): None, (1, 2): None, (2, 3): None}
    target.add_instruction(CZGate(), cz_properties)
    target.add_instruction(Measure(), {(idx,): None for idx in range(4)})
    return target


def test_construction_no_target():
    gs = QiskitGateSet(127)
    assert gs.num_qubits == 127
    assert gs.target is None
    assert gs.qubit_subset == set(range(127))
    assert set(gs) == {"M", "P"}

    meas = QuantumCircuit(127, 127)
    meas.measure(range(127), range(127))
    assert gs["M"] == QiskitGate("M", meas, range(127))
    assert gs["P"] == QiskitGate("P", QuantumCircuit(127), range(127), range(127))


def test_construction_with_qubit_subset():
    gs = QiskitGateSet(127, qubit_subset=range(4, 90))
    assert gs.num_qubits == 127
    assert gs.target is None
    assert gs.qubit_subset == set(range(4, 90))
    assert set(gs) == {"M", "P"}

    num_in_subset = 90 - 4
    meas = QuantumCircuit(num_in_subset, num_in_subset)
    meas.measure(range(num_in_subset), range(num_in_subset))
    assert gs["M"] == QiskitGate("M", meas, range(4, 90))
    assert gs["P"] == QiskitGate("P", QuantumCircuit(num_in_subset), range(4, 90), range(4, 90))


def test_construction_with_qubit_subset_ordering():
    gs = QiskitGateSet(10, qubit_subset=[7, 2, 5])
    assert gs["M"].qubit_idxs == (7, 2, 5)
    assert gs["P"].qubit_idxs == (7, 2, 5)


@pytest.mark.parametrize("num_qubits", [None, 4])
def test_construction_with_target(num_qubits, target_4q):
    gs = QiskitGateSet(num_qubits, target=target_4q)
    assert gs.num_qubits == 4
    assert gs.target == target_4q
    assert gs.qubit_subset == set(range(4))
    assert set(gs) == {"M", "P"}


def test_constructor_raises(target_4q):
    with pytest.raises(ValueError, match="At least one of `num_qubits` or `target`"):
        QiskitGateSet()

    with pytest.raises(ValueError, match=r"must be a subset of range\(20\)"):
        QiskitGateSet(20, qubit_subset=range(25))

    with pytest.raises(ValueError, match="`num_qubits` must match `target.num_qubits`"):
        QiskitGateSet(3, target=target_4q)


def test_add_gate(target_4q):
    gs = QiskitGateSet(10)
    gate = QiskitGate("L0", QuantumCircuit(3), [4, 8, 9])
    gs.add_gate(gate)

    assert gs["L0"] == gate

    gs = QiskitGateSet(target=target_4q)
    circuit = QuantumCircuit(3)
    circuit.cz(0, 1)
    gate = QiskitGate("L0", circuit, [0, 1, 2])
    gs.add_gate(gate)

    assert gs["L0"] == gate


def test_add_gate_raises(target_4q):
    gs = QiskitGateSet(10)
    with pytest.raises(ValueError, match="outside of the valid range"):
        gs.add_gate(QiskitGate("L0", QuantumCircuit(2), [5, 15]))
    assert len(gs) == 2

    gs = QiskitGateSet(target=target_4q)
    circuit = QuantumCircuit(2)
    circuit.cz(0, 1)
    with pytest.raises(ValueError, match=r"Operation .*cz.* on \(0, 3\) is not supported"):
        gs.add_gate(QiskitGate("L0", circuit, [0, 3]))
    assert len(gs) == 2

    with pytest.raises(ValueError, match="The gate name"):
        gs.add_gate(QiskitGate("M", circuit, [0, 1]))


@pytest.mark.parametrize("name", [None, "X"])
def test_add_box_as_gate(name):
    gs = QiskitGateSet(10)
    circuit = QuantumCircuit(10)
    with circuit.box():
        circuit.cx(3, 4)
        circuit.noop([1, 8])
    returned_name = gs.add_box_as_gate(circuit[0], name=name)
    assert name is None or name == returned_name

    body = QuantumCircuit(4)
    body.cx(0, 1)
    assert gs[returned_name] == QiskitGate(returned_name, body, [3, 4, 1, 8])


def test_add_box_as_gate_raises():
    gs = QiskitGateSet(10)

    circuit = QuantumCircuit(QuantumRegister(10, "not_q"))
    with circuit.box():
        circuit.cx(3, 4)
        circuit.noop([1, 8])
    with pytest.raises(ValueError, match="does not act on a single quantum register named 'q'"):
        gs.add_box_as_gate(circuit[0])
    assert len(gs) == 2

    circuit = QuantumCircuit(QuantumRegister(4, "q"))
    with circuit.box():
        circuit.cx(1, 0)
    with pytest.raises(ValueError, match="single quantum register named 'q' of size 10"):
        gs.add_box_as_gate(circuit[0])
    assert len(gs) == 2


@pytest.mark.parametrize("name", [None, "X"])
def test_add_circuit_as_gate(name):
    circuit = QuantumCircuit(5)
    circuit.cx(1, 2)

    gs = QiskitGateSet(10)
    returned_name = gs.add_circuit_as_gate(circuit, name=name)
    assert name is None or name == returned_name
    assert gs[returned_name] == QiskitGate(returned_name, circuit, [0, 1, 2, 3, 4])

    gs = QiskitGateSet(10)
    returned_name = gs.add_circuit_as_gate(circuit, [4, 5, 0, 8, 9], name=name)
    assert name is None or name == returned_name
    assert gs[returned_name] == QiskitGate(returned_name, circuit, [4, 5, 0, 8, 9])


@pytest.mark.parametrize("name", [None, "X"])
def test_build_new_gate(name):
    gs = QiskitGateSet(10)

    with gs.build_new_gate(name, idle_unused=False) as builder:
        builder.circuit.cx(4, 5)
        builder.circuit.cx(9, 1)
        builder.circuit.noop([3, 8])

    assert name is None or name == builder.name

    body = QuantumCircuit(6)
    body.cx(0, 1)
    body.cx(2, 3)
    assert gs[builder.name] == QiskitGate(builder.name, body, [4, 5, 9, 1, 3, 8])


def test_build_new_gate_idle_unused_default():
    gs = QiskitGateSet(10)

    with gs.build_new_gate() as builder:
        builder.circuit.cx(4, 5)
        builder.circuit.cx(9, 1)

    gate = gs[builder.name]
    assert frozenset(gate.qubit_idxs) == frozenset(range(10))
    assert gate.gate_idxs == frozenset({4, 5, 9, 1})
    assert gate.idling_idxs == frozenset({0, 2, 3, 6, 7, 8})


def test_build_new_gate_idle_unused_with_subset():
    gs = QiskitGateSet(10, qubit_subset=range(2, 8))

    with gs.build_new_gate() as builder:
        builder.circuit.cx(4, 5)
        builder.circuit.add_bits(ClassicalRegister(1))
        builder.circuit.measure([6], [0])

    gate = gs[builder.name]
    assert frozenset(gate.qubit_idxs) == frozenset(range(2, 8))
    assert gate.gate_idxs == frozenset({4, 5})
    assert gate.idling_idxs == frozenset({2, 3, 7})
    assert gate.meas_idxs == frozenset({6})


@pytest.mark.parametrize("name", [None, "M2"])
def test_add_measurement(name):
    gs = QiskitGateSet(10)
    returned_name = gs.add_measurement(name=name)
    assert name is None or name == returned_name
    meas = QuantumCircuit(10, 10)
    meas.measure(range(10), range(10))
    assert gs[returned_name] == QiskitGate(returned_name, meas, range(10))

    other_returned_name = gs.add_measurement(annotations=[Twirl(decomposition="rzrx")])
    assert gs[other_returned_name] == QiskitGate(
        other_returned_name, meas, range(10), annotations=[Twirl(decomposition="rzrx")]
    )
    assert gs[other_returned_name] != gs[returned_name]

    gs = QiskitGateSet(10)
    returned_name = gs.add_measurement(qubit_idxs=[6, 7, 8], name=name)
    assert name is None or name == returned_name
    meas = QuantumCircuit(3, 3)
    meas.measure(range(3), range(3))
    assert gs[returned_name] == QiskitGate(returned_name, meas, [6, 7, 8])

    class Measure2(Measure):
        pass

    gs = QiskitGateSet(10)
    returned_name = gs.add_measurement([6, 7, 8], Measure2, name=name)
    assert name is None or name == returned_name
    meas = QuantumCircuit(3, 3)
    meas.append(Measure2(), [0], [0])
    meas.append(Measure2(), [1], [1])
    meas.append(Measure2(), [2], [2])
    assert gs[returned_name] == QiskitGate(returned_name, meas, [6, 7, 8])


@pytest.mark.parametrize("name", [None, "M2"])
def test_add_preparation(name):
    gs = QiskitGateSet(10)
    returned_name = gs.add_preparation(name=name)
    assert name is None or name == returned_name
    assert gs[returned_name] == QiskitGate(returned_name, QuantumCircuit(10), range(10), range(10))

    other_returned_name = gs.add_preparation(annotations=[Twirl(decomposition="rzrx")])
    assert gs[other_returned_name] == QiskitGate(
        other_returned_name,
        QuantumCircuit(10),
        range(10),
        prep_idxs=range(10),
        annotations=[Twirl(decomposition="rzrx")],
    )
    assert gs[other_returned_name] != gs[returned_name]

    gs = QiskitGateSet(10)
    returned_name = gs.add_preparation(qubit_idxs=[6, 7, 8], name=name)
    assert name is None or name == returned_name
    assert gs[returned_name] == QiskitGate(returned_name, QuantumCircuit(3), [6, 7, 8], [6, 7, 8])


def test_model_gate_set(target_4q):
    # default measurement and prep
    gs = QiskitGateSet(127)
    model_gs = gs.model_gate_set

    assert len(model_gs) == 2
    assert model_gs["M"] == ModelGate("M", [], qubit_idxs=range(127), meas_idxs=range(127))
    assert model_gs["P"] == ModelGate("P", [], qubit_idxs=range(127), prep_idxs=range(127))
    assert model_gs.coupling_map == CouplingMap.from_full(127)

    gs = QiskitGateSet(4, target=target_4q)
    model_gs = gs.model_gate_set
    assert len(model_gs) == 2
    assert model_gs["M"] == ModelGate("M", [], qubit_idxs=range(4), meas_idxs=range(4))
    assert model_gs["P"] == ModelGate("P", [], qubit_idxs=range(4), prep_idxs=range(4))
    assert model_gs.coupling_map == CouplingMap([(0, 1), (1, 2), (2, 3)])


def test_latex_str_propagation():
    """The ``latex_str`` argument is forwarded by every gate-adding method."""
    gs = QiskitGateSet(10)

    circuit = QuantumCircuit(5)
    circuit.cx(1, 2)
    name = gs.add_circuit_as_gate(circuit, latex_str=r"\alpha")
    assert gs[name].latex_str == r"\alpha"

    box_circuit = QuantumCircuit(10)
    with box_circuit.box():
        box_circuit.cx(3, 4)
        box_circuit.noop([1, 8])
    name = gs.add_box_as_gate(box_circuit[0], latex_str=r"\beta")
    assert gs[name].latex_str == r"\beta"

    name = gs.add_measurement(latex_str=r"\gamma")
    assert gs[name].latex_str == r"\gamma"

    name = gs.add_preparation(latex_str=r"\delta")
    assert gs[name].latex_str == r"\delta"

    with gs.build_new_gate(latex_str=r"\epsilon") as builder:
        builder.circuit.cx(4, 5)
    assert gs[builder.name].latex_str == r"\epsilon"

    # gates added without an explicit value fall back to the name
    assert gs["M"].latex_str == "M"


def test_name_and_latex_str():
    # default to the class name, with latex_str falling back to name
    gs = QiskitGateSet(10)
    assert gs.name == "QiskitGateSet"
    assert gs.latex_str == "QiskitGateSet"

    # explicit name, latex_str still falls back to it
    gs = QiskitGateSet(10, name="my_set")
    assert gs.name == "my_set"
    assert gs.latex_str == "my_set"

    # both supplied explicitly
    gs = QiskitGateSet(10, name="my_set", latex_str=r"\mathcal{G}")
    assert gs.name == "my_set"
    assert gs.latex_str == r"\mathcal{G}"


def test_name_and_latex_str_propagate_to_model_gate_set():
    # an explicit name/latex_str carries over to the derived model gate set
    gs = QiskitGateSet(4, name="my_set", latex_str=r"\mathcal{G}")
    model_gs = gs.model_gate_set
    assert model_gs.name == "my_set"
    assert model_gs.latex_str == r"\mathcal{G}"

    # when unset, the model gate set falls back to its own class name rather than inheriting
    # the QiskitGateSet class name
    model_gs = QiskitGateSet(4).model_gate_set
    assert model_gs.name == "ModelGateSet"


def test_repr_html():
    """Test that the HTML repr doesn't fail."""

    gs = QiskitGateSet(10)

    with gs.build_new_gate("x") as builder:
        builder.circuit.cx(4, 5)
        builder.circuit.cx(9, 1)
        builder.circuit.noop([3, 8])

    html = gs._repr_html_()
    assert isinstance(html, str)
    assert html.count("<table") == 1
    assert html.count("<tr") == 4
