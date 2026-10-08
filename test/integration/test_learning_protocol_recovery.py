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

"""End-to-end test that the learning protocol recovers the noise it was given.

Covers the ``prepare_learning_program`` / ``process_learning_result`` pair together: the program
is built, run against a simulator injecting a known noise map, and analyzed using nothing but the
results, so the passthrough data is what carries the experiment between the two halves.
"""

import pytest
from qiskit.circuit import ClassicalRegister, QuantumCircuit
from qiskit.circuit.library import Measure, XGate
from qiskit.quantum_info import PauliLindbladMap
from qiskit.transpiler import PassManager, TransformationPass
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider.backends.fez import FakeFez
from samplomatic import InjectNoise, Twirl

from qiskit_noise_learning.aer_executor import AerExecutor
from qiskit_noise_learning.analysis import FlipPostSelect
from qiskit_noise_learning.models import split_pauli_lindblad_model
from qiskit_noise_learning.protocols import prepare_learning_program, process_learning_result

PAIR = (17, 27)
INJECTED_RATE = 5e-3


def _make_annotated_layer(backend, pair=(17, 27)):
    """Build a circuit holding one twirled, noise-injectable box of a single CZ."""
    circuit = QuantumCircuit(backend.num_qubits)
    with circuit.box([Twirl(), InjectNoise("layer")]):
        circuit.cz(*pair)
    return circuit


def test_learning_protocol_recovers_injected_noise():
    """The protocol pair learns back the noise that was injected."""
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

    program = prepare_learning_program(
        backend,
        [circuit[0]],
        num_randomizations=16,
        shots_per_randomization=64,
        fragment_depths=[2, 8, 32],
    )
    fit = process_learning_result(executor.run(program).result())

    learned = split_pauli_lindblad_model(fit.model).model.to_pauli_lindblad_maps(fit.model_data)
    assert set(learned) == {"layer"}

    rates = {
        (pauli, tuple(indices)): rate for pauli, indices, rate in learned["layer"].to_sparse_list()
    }
    assert rates.pop(("ZZ", (17, 27))) == pytest.approx(injected_rate, rel=0.1)
    assert max(rates.values()) < 0.075 * injected_rate, "weight leaked onto uninjected generators"

    with_spam = split_pauli_lindblad_model(fit.model).model.to_pauli_lindblad_maps(
        fit.model_data, include_spam=True
    )
    assert set(with_spam) == {"layer", "M"}
    assert {(pauli, tuple(indices)) for pauli, indices, _ in with_spam["M"].to_sparse_list()} == {
        ("X", (qubit,)) for qubit in PAIR
    }


class _AddFlipCheck(TransformationPass):
    """Re-measure each qubit of the pair after an ``X``, into a ``_ps``-suffixed register.

    This is the scheme FlipPostSelect's two-creg rule expects: a healthy bit flips between the
    two reads, and one that does not is what post-selection discards.
    """

    def run(self, dag):
        creg = ClassicalRegister(len(PAIR), "meas0_ps")
        dag.add_creg(creg)
        for index, qubit in enumerate(PAIR):
            dag.apply_operation_back(XGate(), qargs=[dag.qubits[qubit]])
            dag.apply_operation_back(Measure(), qargs=[dag.qubits[qubit]], cargs=[creg[index]])
        return dag


def test_post_selection_hooks_compose():
    """The pass manager and raw-data stage arguments work together across the pair."""
    backend = FakeFez()
    circuit = _make_annotated_layer(backend)

    executor = AerExecutor(
        AerSimulator(method="stabilizer"),
        noise_dict={
            "layer": PauliLindbladMap.from_list([("ZZ", INJECTED_RATE)]),
            "P": PauliLindbladMap.from_list([("XI", 1e-3), ("IX", 1e-3)]),
            "M": PauliLindbladMap.from_list([("XI", 1e-3), ("IX", 1e-3)]),
        },
        root_seed=7,
    )

    program = prepare_learning_program(
        backend,
        [circuit[0]],
        num_randomizations=16,
        shots_per_randomization=64,
        fragment_depths=[2, 8, 32],
        pass_manager=PassManager([_AddFlipCheck()]),
    )
    fit = process_learning_result(executor.run(program).result(), raw_data_stage=FlipPostSelect())

    learned = split_pauli_lindblad_model(fit.model).model.to_pauli_lindblad_maps(fit.model_data)
    rates = {
        (pauli, tuple(indices)): rate for pauli, indices, rate in learned["layer"].to_sparse_list()
    }
    assert rates[("ZZ", PAIR)] == pytest.approx(INJECTED_RATE, rel=0.1)


def test_recovery_with_a_supplied_measurement_layer():
    """The protocol still recovers the layer when the caller supplies the measurement box."""
    backend = FakeFez()
    layer = _make_annotated_layer(backend)
    measurement = QuantumCircuit(backend.num_qubits, len(PAIR))
    with measurement.box([Twirl(), InjectNoise("readout")]):
        # Reversed, so that the supplied order is what the analysis reads outcomes back in.
        for clbit, qubit in enumerate(reversed(PAIR)):
            measurement.measure(qubit, clbit)

    executor = AerExecutor(
        AerSimulator(method="stabilizer"),
        noise_dict={
            "layer": PauliLindbladMap.from_list([("ZZ", INJECTED_RATE)]),
            "P": PauliLindbladMap.from_list([("XI", 1e-3), ("IX", 1e-3)]),
            "readout": PauliLindbladMap.from_list([("XI", 1e-3), ("IX", 1e-3)]),
        },
        root_seed=7,
    )

    program = prepare_learning_program(
        backend,
        [layer[0], measurement[0]],
        num_randomizations=16,
        shots_per_randomization=64,
        fragment_depths=[2, 8, 32],
    )
    fit = process_learning_result(executor.run(program).result())

    model = split_pauli_lindblad_model(fit.model).model
    learned = model.to_pauli_lindblad_maps(fit.model_data)
    assert set(learned) == {"layer"}

    rates = {
        (pauli, tuple(indices)): rate for pauli, indices, rate in learned["layer"].to_sparse_list()
    }
    assert rates.pop(("ZZ", PAIR)) == pytest.approx(INJECTED_RATE, rel=0.1)
    assert max(rates.values()) < 0.075 * INJECTED_RATE, "weight leaked onto uninjected generators"

    # The SPAM noise is carried by the supplied box, under the name its annotation gave it.
    with_spam = model.to_pauli_lindblad_maps(fit.model_data, include_spam=True)
    assert set(with_spam) == {"layer", "readout"}
    assert {
        (pauli, tuple(indices)) for pauli, indices, _ in with_spam["readout"].to_sparse_list()
    } == {("X", (qubit,)) for qubit in PAIR}
