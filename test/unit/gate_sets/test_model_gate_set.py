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
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Clifford
from qiskit.transpiler import CouplingMap

from qiskit_noise_learning.gate_sets import ModelGate, ModelGateSet


def test_construction():
    gs = ModelGateSet(127)
    assert gs.num_qubits == 127
    assert gs.qubit_subset == set(range(127))
    assert set(gs) == set()
    assert gs.coupling_map == CouplingMap.from_full(127)

    coupling_map = CouplingMap([[x, x + 1] for x in range(126)])
    gs = ModelGateSet(127, coupling_map=coupling_map)
    assert gs.coupling_map == coupling_map


def test_construction_with_qubit_subset():
    gs = ModelGateSet(127, qubit_subset=range(4, 90))
    assert gs.num_qubits == 127
    assert gs.qubit_subset == set(range(4, 90))
    assert set(gs) == set()


def test_name_and_latex_str():
    # default to the class name, with latex_str falling back to name
    gs = ModelGateSet(4)
    assert gs.name == "ModelGateSet"
    assert gs.latex_str == "ModelGateSet"

    # explicit name, latex_str still falls back to it
    gs = ModelGateSet(4, name="my_set")
    assert gs.name == "my_set"
    assert gs.latex_str == "my_set"

    # both supplied explicitly
    gs = ModelGateSet(4, name="my_set", latex_str=r"\mathcal{G}")
    assert gs.name == "my_set"
    assert gs.latex_str == r"\mathcal{G}"


def test_constructor_raises():
    with pytest.raises(ValueError, match=r"must be a subset of range\(20\)"):
        ModelGateSet(20, qubit_subset=range(25))


def test_add_gate():
    gs = ModelGateSet(10)
    gate = ModelGate("L0", [((4, 8, 9), Clifford(QuantumCircuit(3)))])
    gs.add_gate(gate)

    assert gs["L0"] == gate


def test_add_gate_raises():
    gs = ModelGateSet(10)
    with pytest.raises(ValueError, match="outside of the valid range"):
        gs.add_gate(ModelGate("L0", [((4, 5, 15), Clifford(QuantumCircuit(3)))]))

    with pytest.raises(ValueError, match="The gate name L0 is already used"):
        gate = ModelGate("L0", [((4, 8, 9), Clifford(QuantumCircuit(3)))])
        gs.add_gate(gate)
        gs.add_gate(gate)
