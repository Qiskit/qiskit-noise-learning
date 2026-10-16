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

"""Tests for process_vanilla_result."""

import pytest
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import PauliLindbladMap, QubitSparsePauli
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import Executor
from qiskit_ibm_runtime.fake_provider.backends.fez import FakeFez
from samplomatic import InjectNoise, Tag, Twirl
from samplomatic.quantum_program import QuantumProgramResult

from qiskit_noise_learning.analysis import (
    ComputeObservables,
    CurveFitObservables,
    FlipPostSelect,
    LegacySolve,
)
from qiskit_noise_learning.circuit_generator.executor.data_mapper_serialization import (
    PAYLOAD_KEY,
    PROTOCOL_KEY,
)
from qiskit_noise_learning.data import (
    AggregatedObservableData,
    ModelData,
    ObservableData,
    RawData,
)
from qiskit_noise_learning.protocols import prepare_vanilla_program, process_vanilla_result
from qiskit_noise_learning.protocols.vanilla.prepare_vanilla_program import (
    VANILLA_KEY,
    VERSION,
)

PAIR = (17, 27)


@pytest.fixture(scope="module")
def result():
    """Result of a small real learning program, so the pair is exercised end to end."""
    backend = FakeFez()
    circuit = QuantumCircuit(backend.num_qubits)
    with circuit.box([Twirl(), Tag("layer"), InjectNoise("layer")]):
        circuit.cz(*PAIR)

    program = prepare_vanilla_program(
        backend,
        [circuit[0]],
        num_randomizations=4,
        shots_per_randomization=16,
        fragment_depths=[2, 8],
    )
    executor = Executor(
        mode=AerSimulator(method="stabilizer"),
        options={
            "simulator": {
                "layer_noise_model": [(circuit[0], PauliLindbladMap.from_list([("ZZ", 5e-3)]))],
                "seed_simulator": 7,
                "warn_absent": False,
            }
        },
    )
    return executor.run(program).result()


@pytest.fixture(scope="module")
def result_with_a_measurement_box():
    """Result of a program whose submitted instructions include the measurement layer."""
    backend = FakeFez()
    circuit = QuantumCircuit(backend.num_qubits)
    with circuit.box([Twirl(), Tag("layer"), InjectNoise("layer")]):
        circuit.cz(*PAIR)
    measurement = QuantumCircuit(backend.num_qubits, len(PAIR))
    with measurement.box([Twirl(), Tag("readout"), InjectNoise("readout")]):
        for clbit, qubit in enumerate(PAIR):
            measurement.measure(qubit, clbit)

    program = prepare_vanilla_program(
        backend,
        [circuit[0], measurement[0]],
        num_randomizations=4,
        shots_per_randomization=16,
        fragment_depths=[2, 8],
    )
    executor = Executor(
        mode=AerSimulator(method="stabilizer"),
        options={
            "simulator": {
                "layer_noise_model": [
                    (circuit[0], PauliLindbladMap.from_list([("ZZ", 5e-3)])),
                    (measurement[0], PauliLindbladMap.from_list([("XI", 2e-2)])),
                ],
                "seed_simulator": 7,
                "warn_absent": False,
            }
        },
    )
    return executor.run(program).result()


def test_rejects_raw_data_stage_that_is_not_a_stage():
    """Caught here rather than as an AttributeError from inside pipeline construction."""
    with pytest.raises(TypeError, match="must be an AnalysisStage"):
        process_vanilla_result(object(), raw_data_stage=object())


@pytest.mark.parametrize(
    "stage",
    [
        ComputeObservables(),
        LegacySolve(),
        ComputeObservables() + CurveFitObservables(),
    ],
    ids=["consumes-raw-produces-observable", "neither-end-raw", "pipeline-with-wrong-output"],
)
def test_rejects_raw_data_stage_with_wrong_levels(stage):
    with pytest.raises(ValueError, match="must consume and produce RawData"):
        process_vanilla_result(object(), raw_data_stage=stage)


def test_populates_every_analysis_level(result):
    """The prepared program's result carries enough to run the whole analysis."""
    _, _, fit = process_vanilla_result(result)

    assert isinstance(fit.raw_data, RawData)
    assert isinstance(fit.observable_data, ObservableData)
    assert isinstance(fit.aggregated_observable_data, AggregatedObservableData)
    assert isinstance(fit.model_data, ModelData)
    # Recovered from the program's passthrough data, not supplied by the caller.
    assert fit.model is not None
    assert fit.paths
    assert fit.instruction_sequences


def test_raw_data_stage_runs_ahead_of_the_standard_analysis(result):
    """A supplied stage writes RawData once more than the collection alone does."""
    *_, without = process_vanilla_result(result)
    *_, with_stage = process_vanilla_result(result, raw_data_stage=FlipPostSelect())

    assert len(with_stage.history.raw_data) == len(without.history.raw_data) + 1


def test_reports_the_noise_of_the_one_submitted_instruction(result):
    """One instruction in, one noise map out, keyed by the gate it became."""
    noise_maps, gate_names, _ = process_vanilla_result(result)

    assert gate_names == ["layer"]
    assert len(noise_maps) == 1
    assert noise_maps[0].num_qubits == len(PAIR)
    assert noise_maps[0].rates.sum() > 0


def test_reports_a_submitted_measurement_box_at_its_own_position(result_with_a_measurement_box):
    """A measurement box is one of the instructions, so its readout noise is reported like any."""
    noise_maps, gate_names, _ = process_vanilla_result(result_with_a_measurement_box)

    assert gate_names == ["layer", "readout"]
    readout = noise_maps[1]
    # All SPAM noise is attributed to the measurement, as X generators on the measured qubits.
    assert readout.num_qubits == len(PAIR)
    assert {
        (tuple(generator.indices), tuple(generator.paulis)) for generator in readout.generators()
    } == {((idx,), (QubitSparsePauli.Pauli.X,)) for idx in range(len(PAIR))}


def test_the_synthesized_measurement_layer_is_not_reported(result):
    """It is not a submitted instruction, so it is on the fit but not in the lists."""
    _, gate_names, fit = process_vanilla_result(result)
    model_data = fit.model_data

    assert "M" not in gate_names
    assert any(index.gate_name == "M" for index in model_data.dataset["parameter_index"].data)


def _with_protocol_data(result, protocol_data):
    """The given result, with the protocol data of its passthrough data replaced.

    ``None`` leaves the entry out altogether, as the result of a program no protocol built has it.
    """
    payload = {
        key: value
        for key, value in result.passthrough_data[PAYLOAD_KEY].items()
        if key != PROTOCOL_KEY
    }
    if protocol_data is not None:
        payload[PROTOCOL_KEY] = protocol_data
    return QuantumProgramResult(
        list(result), passthrough_data={**result.passthrough_data, PAYLOAD_KEY: payload}
    )


@pytest.mark.parametrize(
    ("protocol_data", "match"),
    [
        pytest.param(None, "prepare_vanilla_program did not build", id="no-protocol-data"),
        pytest.param(
            {"gauge_aware": {"version": 1}}, "built by a different protocol", id="another-protocol"
        ),
        pytest.param(
            {VANILLA_KEY: {"version": VERSION + 1, "gate_names": ["layer"]}},
            f"reads version {VERSION}",
            id="another-version",
        ),
        pytest.param(
            {VANILLA_KEY: {"gate_names": ["layer"]}},
            f"reads version {VERSION}",
            id="no-version",
        ),
    ],
)
def test_rejects_protocol_data_it_cannot_read(result, protocol_data, match):
    """Processing is paired with building, so only this protocol's own record is read."""
    with pytest.raises(ValueError, match=match):
        process_vanilla_result(_with_protocol_data(result, protocol_data))


@pytest.mark.parametrize("carries_passthrough_data", [False, True], ids=["bare", "with-payload"])
def test_rejects_empty_result(result, carries_passthrough_data):
    passthrough_data = result.passthrough_data if carries_passthrough_data else None
    empty = QuantumProgramResult([], passthrough_data=passthrough_data)

    with pytest.raises(ValueError, match="holds no program items"):
        process_vanilla_result(empty)
