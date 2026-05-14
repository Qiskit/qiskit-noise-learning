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

"""Unit tests for DecomposeRxPass."""

import numpy as np
import pytest
from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.transpiler import PassManager

from qiskit_noise_learning.aer_executor.decompose_rx_pass import DecomposeRxPass


def run_pass(qc: QuantumCircuit) -> QuantumCircuit:
    return PassManager([DecomposeRxPass()]).run(qc)


def test_no_rx_gates_unchanged():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.rz(np.pi / 2, 1)

    result = run_pass(qc)

    assert result.count_ops() == qc.count_ops()


def test_rx_replaced_by_h_rz_h():
    qc = QuantumCircuit(1)
    qc.rx(np.pi / 2, 0)

    result = run_pass(qc)
    ops = [inst.operation.name for inst in result.data]

    assert "rx" not in result.count_ops()
    assert ops == ["h", "rz", "h"]


def test_rx_angle_preserved():
    angle = np.pi / 3
    qc = QuantumCircuit(1)
    qc.rx(angle, 0)

    result = run_pass(qc)
    rz_inst = next(inst for inst in result.data if inst.operation.name == "rz")

    assert rz_inst.operation.params[0] == angle


@pytest.mark.parametrize("angle", [0, np.pi / 2, np.pi, -np.pi / 4])
def test_rx_decomposition_multiple_angles(angle):
    qc = QuantumCircuit(1)
    qc.rx(angle, 0)

    result = run_pass(qc)

    assert "rx" not in result.count_ops()
    assert result.count_ops().get("h", 0) == 2
    assert result.count_ops().get("rz", 0) == 1


def test_multiple_rx_gates_all_decomposed():
    qc = QuantumCircuit(2)
    qc.rx(np.pi, 0)
    qc.rx(np.pi / 2, 1)

    result = run_pass(qc)

    assert "rx" not in result.count_ops()
    assert result.count_ops()["h"] == 4
    assert result.count_ops()["rz"] == 2


def test_rx_with_other_gates_preserved():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.rx(np.pi, 0)
    qc.cx(0, 1)

    result = run_pass(qc)

    assert "rx" not in result.count_ops()
    assert result.count_ops()["cx"] == 1
    # 1 original H plus 2 from the RX decomposition
    assert result.count_ops()["h"] == 3


def test_parameterized_rx_preserves_parameter():
    theta = Parameter("theta")
    qc = QuantumCircuit(1)
    qc.rx(theta, 0)

    result = run_pass(qc)
    rz_inst = next(inst for inst in result.data if inst.operation.name == "rz")

    assert "rx" not in result.count_ops()
    assert rz_inst.operation.params[0] == theta


def test_dag_interface_directly():
    qc = QuantumCircuit(1)
    qc.rx(np.pi, 0)

    out_dag = DecomposeRxPass().run(circuit_to_dag(qc))
    result = dag_to_circuit(out_dag)

    assert "rx" not in result.count_ops()
    assert [inst.operation.name for inst in result.data] == ["h", "rz", "h"]
