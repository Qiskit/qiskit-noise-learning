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

"""Unit tests for run_quantum_program."""

from unittest.mock import MagicMock

import numpy as np
import pytest
from qiskit.circuit import Parameter, QuantumCircuit
from qiskit_ibm_runtime.quantum_program import QuantumProgram

from qiskit_noise_learning.aer_executor.run_quantum_program import run_quantum_program


def test_unsupported_item_type_raises_type_error(stabilizer_simulator):
    fake_item = MagicMock()
    fake_item.circuit = QuantumCircuit(1)

    program = MagicMock()
    program.items = [fake_item]
    program.shots = 64
    program.passthrough_data = None

    with pytest.raises(TypeError, match="Unsupported QuantumProgramItem type"):
        run_quantum_program(stabilizer_simulator, program)


def test_angle_rounding_snaps_near_clifford(stabilizer_simulator):
    """RZ(π + ε) with a tiny ε should round to π, yielding |1⟩ deterministically.

    H → RZ(π) → H maps |0⟩ → |1⟩. Without rounding, ε would make the angle non-Clifford
    and the stabilizer simulator would error.
    """
    theta = Parameter("theta")
    qc = QuantumCircuit(1, 1)
    qc.h(0)
    qc.rz(theta, 0)
    qc.h(0)
    qc.measure(0, 0)

    circuit_arguments = np.array([[np.pi + 1e-10]])  # shape (sweeps=1, params=1)
    program = QuantumProgram(shots=64)
    program.append_circuit_item(qc, circuit_arguments=circuit_arguments)

    result = run_quantum_program(stabilizer_simulator, program, angle_decimals=5)

    assert (result[0]["c"] == [[True]]).all()
