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

import plotly.graph_objects as go
import pytest
from qiskit.circuit import Measure, QuantumCircuit, Reset
from qiskit.circuit.library import CXGate
from qiskit.transpiler import InstructionProperties, Target

from qiskit_noise_learning.gate_sets import QiskitGateSet
from qiskit_noise_learning.visualizations import gate_set_topology


def _make_5q_target() -> Target:
    """Build a minimal 5-qubit Target with CX, Measure, Reset on a star topology."""
    target = Target(num_qubits=5)
    cx_props = {
        (0, 1): InstructionProperties(),
        (1, 0): InstructionProperties(),
        (1, 2): InstructionProperties(),
        (2, 1): InstructionProperties(),
        (1, 3): InstructionProperties(),
        (3, 1): InstructionProperties(),
        (1, 4): InstructionProperties(),
        (4, 1): InstructionProperties(),
    }
    target.add_instruction(CXGate(), cx_props)
    single_props = {(q,): InstructionProperties() for q in range(5)}
    target.add_instruction(Measure(), single_props)
    target.add_instruction(Reset(), single_props)
    return target


@pytest.fixture()
def gate_set_5q():
    target = _make_5q_target()
    gate_set = QiskitGateSet(target=target)

    circ1 = QuantumCircuit(5)
    circ1.cx(0, 1)
    circ1.cx(2, 1)
    gate_set.add_circuit_as_gate(circ1, range(5), name="L0")

    circ2 = QuantumCircuit(5)
    circ2.cx(3, 1)
    circ2.cx(4, 1)
    gate_set.add_circuit_as_gate(circ2, range(5), name="L1")

    return gate_set


def test_returns_figure(gate_set_5q):
    fig = gate_set_topology(gate_set_5q)
    assert isinstance(fig, go.Figure)


def test_draw_method(gate_set_5q):
    fig = gate_set_5q.draw()
    assert isinstance(fig, go.Figure)


def test_has_traces(gate_set_5q):
    fig = gate_set_topology(gate_set_5q)
    # Should have background edges + per-gate colored edges + SPAM markers + node trace
    assert len(fig.data) > 1


def test_no_target_raises():
    gate_set = QiskitGateSet(5)
    with pytest.raises(ValueError, match="target is None"):
        gate_set.draw()
