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

"""End-to-end tests for protocol functions."""

import pytest
from qiskit.circuit import ClassicalRegister, QuantumCircuit
from qiskit.circuit.library import Measure, XGate
from qiskit.quantum_info import PauliLindbladMap
from qiskit.transpiler import PassManager, TransformationPass
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import Executor
from qiskit_ibm_runtime.fake_provider.backends.fez import FakeFez
from samplomatic import InjectNoise, Tag, Twirl

from qiskit_noise_learning.analysis import FlipPostSelect
from qiskit_noise_learning.protocols import prepare_vanilla_program, process_vanilla_result

# backend and qubits
BACKEND = FakeFez()
PAIR = (17, 27)

# rates for CZ layer
ZZ_RATE, Z17_RATE, Z27_RATE = 5e-3, 2e-3, 3e-3

# rates for measurement
X17_RATE, X27_RATE = 3e-2, 2e-2

# noise maps
LAYER_NOISE = PauliLindbladMap.from_sparse_list(
    [("ZZ", (0, 1), ZZ_RATE), ("Z", (0,), Z17_RATE), ("Z", (1,), Z27_RATE)], 2
)
READOUT_NOISE = PauliLindbladMap.from_sparse_list([("X", (0,), X17_RATE), ("X", (1,), X27_RATE)], 2)


@pytest.fixture(scope="module")
def cz_layer():
    """The boxed CZ whose noise is being learned."""
    circuit = QuantumCircuit(BACKEND.num_qubits)
    with circuit.box([Twirl(), Tag("layer"), InjectNoise("layer")]):
        circuit.cz(*PAIR)
    return circuit[0]


def _measurement_layer(qubits):
    """A boxed measurement reading ``qubits`` into classical bits in the order given."""
    circuit = QuantumCircuit(BACKEND.num_qubits, len(qubits))
    with circuit.box([Twirl(), Tag("readout"), InjectNoise("readout")]):
        for clbit, qubit in enumerate(qubits):
            circuit.measure(qubit, clbit)
    return circuit[0]


def _run_protocol(cz_layer, measurement, pass_manager=None, raw_data_stage=None):
    """Build a program for the two layers, simulate it with the injected noise, and fit it."""
    executor = Executor(
        mode=AerSimulator(method="stabilizer"),
        options={
            "simulator": {
                "layer_noise_model": [(cz_layer, LAYER_NOISE), (measurement, READOUT_NOISE)],
                "seed_simulator": 7,
                # Every program carries a preparation box, whose noise cannot be simulated.
                "warn_absent": False,
            }
        },
    )
    # 64 x 256 rather than something smaller because the readout rates need it: at 16 x 64 they
    # swing by as much as 20%, too loose to catch readout noise being dropped on the floor.
    program = prepare_vanilla_program(
        BACKEND,
        [cz_layer, measurement],
        num_randomizations=64,
        shots_per_randomization=256,
        fragment_depths=[2, 8, 32],
        pass_manager=pass_manager,
    )
    return process_vanilla_result(executor.run(program).result(), raw_data_stage=raw_data_stage)


def _assert_recovers_injected_noise(learned):
    """Assert every injected rate comes back on its own generator, and nowhere else.

    The rates are read off the reported noise maps, so this also pins their indexing: each map
    acts on its own instruction's qubits in ascending physical order, making local qubit 0 the
    physical qubit 17 and local qubit 1 the physical qubit 27 -- whichever order the instruction
    itself names them in.
    """
    assert learned.gate_names == ["layer", "readout"]
    layer_rates, readout_rates = (
        {(pauli, tuple(idxs)): rate for pauli, idxs, rate in noise_map.to_sparse_list()}
        for noise_map in learned.noise_maps
    )

    injected = {("ZZ", (0, 1)): ZZ_RATE, ("Z", (0,)): Z17_RATE, ("Z", (1,)): Z27_RATE}
    for generator, rate in injected.items():
        assert layer_rates[generator] == pytest.approx(rate, rel=0.1)
    assert readout_rates[("X", (0,))] == pytest.approx(X17_RATE, rel=0.1)
    assert readout_rates[("X", (1,))] == pytest.approx(X27_RATE, rel=0.1)

    leaked = {gen: rate for gen, rate in layer_rates.items() if gen not in injected}
    assert max(leaked.values()) < 0.1 * min(
        injected.values()
    ), f"weight leaked onto uninjected generators: {leaked}"


def test_learning_protocol_recovers_injected_noise(cz_layer):
    """The protocol pair learns back the layer and readout noise that were injected."""
    learned = _run_protocol(cz_layer, _measurement_layer(PAIR))

    _assert_recovers_injected_noise(learned)


def test_recovery_with_a_reversed_measurement_bit_order(cz_layer):
    """Recovery is unaffected by which classical bit a supplied box measures each qubit into."""
    learned = _run_protocol(cz_layer, _measurement_layer(tuple(reversed(PAIR))))

    _assert_recovers_injected_noise(learned)


class _AddFlipCheck(TransformationPass):
    """Re-measure each qubit of the pair after an ``X``, into a ``_ps``-suffixed register."""

    def run(self, dag):
        creg = ClassicalRegister(len(PAIR), "meas0_ps")
        dag.add_creg(creg)
        for index, qubit in enumerate(PAIR):
            dag.apply_operation_back(XGate(), qargs=[dag.qubits[qubit]])
            dag.apply_operation_back(Measure(), qargs=[dag.qubits[qubit]], cargs=[creg[index]])
        return dag


def test_post_selection_hooks_compose(cz_layer):
    """The pass manager and raw-data stage arguments work together across the pair."""
    learned = _run_protocol(
        cz_layer,
        _measurement_layer(PAIR),
        pass_manager=PassManager([_AddFlipCheck()]),
        raw_data_stage=FlipPostSelect(),
    )

    _assert_recovers_injected_noise(learned)
