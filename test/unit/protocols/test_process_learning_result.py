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

"""Tests for process_learning_result."""

import pytest
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import PauliLindbladMap
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider.backends.fez import FakeFez
from samplomatic import InjectNoise, Twirl

from qiskit_noise_learning.aer_executor import AerExecutor
from qiskit_noise_learning.analysis import (
    ComputeObservables,
    CurveFitObservables,
    FlipPostSelect,
    LegacySolve,
)
from qiskit_noise_learning.data import (
    AggregatedObservableData,
    ModelData,
    ObservableData,
    RawData,
)
from qiskit_noise_learning.protocols import prepare_learning_program, process_learning_result

PAIR = (17, 27)


@pytest.fixture(scope="module")
def result():
    """Result of a small real learning program, so the pair is exercised end to end."""
    backend = FakeFez()
    circuit = QuantumCircuit(backend.num_qubits)
    with circuit.box([Twirl(), InjectNoise("layer")]):
        circuit.cz(*PAIR)

    program = prepare_learning_program(
        backend,
        [circuit[0]],
        num_randomizations=4,
        shots_per_randomization=16,
        fragment_depths=[2, 8],
    )
    executor = AerExecutor(
        AerSimulator(method="stabilizer"),
        noise_dict={
            "layer": PauliLindbladMap.from_list([("ZZ", 5e-3)]),
            "P": PauliLindbladMap.from_list([("XI", 1e-3), ("IX", 1e-3)]),
            "M": PauliLindbladMap.from_list([("XI", 1e-3), ("IX", 1e-3)]),
        },
        root_seed=7,
    )
    return executor.run(program).result()


def test_rejects_raw_data_stage_that_is_not_a_stage():
    """Caught here rather than as an AttributeError from inside pipeline construction."""
    with pytest.raises(TypeError, match="must be an AnalysisStage"):
        process_learning_result(object(), raw_data_stage=object())


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
        process_learning_result(object(), raw_data_stage=stage)


def test_populates_every_analysis_level(result):
    """The prepared program's result carries enough to run the whole analysis."""
    fit = process_learning_result(result)

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
    without = process_learning_result(result)
    with_stage = process_learning_result(result, raw_data_stage=FlipPostSelect())

    assert len(with_stage.history.raw_data) == len(without.history.raw_data) + 1
