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

"""End-to-end test of a gate acting on fewer qubits than the whole gate set."""

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import Executor
from qiskit_ibm_runtime.fake_provider import FakeManilaV2
from samplomatic import Twirl

from qiskit_noise_learning.data import RawData
from qiskit_noise_learning.protocols import prepare_vanilla_program, process_vanilla_result


def _instructions(num_qubits):
    """A two-qubit gate box, and a measurement box spanning every qubit."""
    circuit = QuantumCircuit(num_qubits, num_qubits)
    with circuit.box([Twirl()]):
        circuit.cx(0, 1)
    with circuit.box([Twirl()]):
        for qubit in range(num_qubits):
            circuit.measure(qubit, qubit)
    return [circuit[0], circuit[1]]


def test_narrow_gate_layer_simulates_without_noise():
    """A gate on two of five qubits yields deterministic outcomes on all five.

    Ensures everything can build and is well-formed.
    """
    backend = FakeManilaV2()
    program = prepare_vanilla_program(
        backend,
        _instructions(backend.num_qubits),
        num_randomizations=8,
        shots_per_randomization=32,
        fragment_depths=[2, 4],
    )
    executor = Executor(
        mode=AerSimulator(method="stabilizer"),
        options={"simulator": {"seed_simulator": 11}},
    )
    fit = process_vanilla_result(executor.run(program).result())

    for name, child in fit[RawData].datatree.children.items():
        data = np.asarray(child.dataset["data"])
        flips = np.asarray(child.dataset["measurement_flips"])
        nonzero_per_bit = np.count_nonzero(data ^ flips[:, None, :], axis=(0, 1))
        assert not nonzero_per_bit.any(), f"item {name}: nonzero per bit {nonzero_per_bit}"
