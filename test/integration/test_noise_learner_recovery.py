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

"""End-to-end tests that a learning experiment recovers the noise it was given.

Unlike the tests under ``test/unit``, these do not target a single subpackage: each one drives the
whole stack — experiment construction, circuit generation, execution and analysis — against a
locally simulated executor carrying a known injected noise model, and asserts on the recovered
rates. They are the check that the pieces still compose, so they are the tests most worth keeping
green through any redesign of how executors are supplied.
"""

import pytest
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import PauliLindbladMap
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider.backends.fez import FakeFez
from samplomatic import InjectNoise, Twirl

from qiskit_noise_learning.aer_executor import AerExecutor
from qiskit_noise_learning.noise_learner import LearningOptions, NoiseLearner


def _make_annotated_layer(backend, pair=(17, 27)):
    """Build a circuit holding one twirled, noise-injectable box of a single CZ."""
    circuit = QuantumCircuit(backend.num_qubits)
    with circuit.box([Twirl(), InjectNoise("layer")]):
        circuit.cz(*pair)
    return circuit


def test_noise_learner_run_against_aer_executor():
    """A learner pointed at an Aer executor learns back the noise that was injected."""
    backend = FakeFez()
    circuit = _make_annotated_layer(backend)

    injected_rate = 5e-3
    executor = AerExecutor(
        AerSimulator(method="stabilizer"),
        noise_dict={
            "layer": PauliLindbladMap.from_list([("ZZ", injected_rate)]),
            "P": PauliLindbladMap.from_list([("XI", 1e-3), ("IX", 1e-3)]),
            "M": PauliLindbladMap.from_list([("XI", 1e-3), ("IX", 1e-3)]),
        },
        root_seed=7,
    )
    options = LearningOptions(
        num_randomizations=16, shots_per_randomizations=64, fragment_depths=[2, 8, 32]
    )

    result = NoiseLearner(backend, options, executor=executor).run([circuit[0]]).result()

    learned = result.to_dict()
    assert set(learned) == {"layer"}

    rates = {
        (pauli, tuple(indices)): rate for pauli, indices, rate in learned["layer"].to_sparse_list()
    }
    assert rates.pop(("ZZ", (17, 27))) == pytest.approx(injected_rate, rel=0.1)
    assert max(rates.values()) < 0.075 * injected_rate, "weight leaked onto uninjected generators"
