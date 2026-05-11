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

import numpy as np
import pytest
from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.primitives.containers.bindings_array import BindingsArray
from qiskit.quantum_info import PauliList
from qiskit_aer.noise import PauliLindbladError

from qiskit_noise_learning.aer_executor.run_quantum_program import get_aer_sampler


@pytest.mark.parametrize("noise", [True, False])
@pytest.mark.parametrize("tol", [1e-16])
@pytest.mark.parametrize("angle", [0, -np.pi / 2])
def test_aer(stabilizer_simulator, angle: float, tol: float, noise: bool):
    qc = QuantumCircuit(156)
    par = Parameter("phi")
    qc.rz(phi=par, qubit=qc.qubits)

    if noise:
        pauli_lindblad_error = PauliLindbladError(
            generators=PauliList(["X" * qc.num_qubits]),
            rates=[1e-6],
        )
        qc.append(pauli_lindblad_error, qc.qubits)

    qc.measure_active()

    bindings_array = BindingsArray({par: [angle + tol] * qc.num_qubits})

    aer_sampler = get_aer_sampler(aer_simulator=stabilizer_simulator)

    sampler_job = aer_sampler.run([(qc, bindings_array)], shots=5)
    sampler_job.result()
