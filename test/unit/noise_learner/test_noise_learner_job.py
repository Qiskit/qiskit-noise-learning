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


from types import SimpleNamespace

import numpy as np
import pytest
from qiskit.circuit.library import CZGate
from qiskit.quantum_info import Clifford, QubitSparsePauliList

from qiskit_noise_learning.analysis import AnalysisStage
from qiskit_noise_learning.circuit_generator.executor_circuit_generator import ExecutorDataMapper
from qiskit_noise_learning.data import ModelData, RawData
from qiskit_noise_learning.gate_sets import ModelGate, ModelGateSet
from qiskit_noise_learning.models import PauliLindbladModel
from qiskit_noise_learning.noise_learner import NoiseLearnerResult
from qiskit_noise_learning.noise_learner.noise_learner_job import ExperimentSchema, NoiseLearnerJob


class _StubProgramResult:
    """Minimal stub mimicking QuantumProgramResult with metadata."""

    def __init__(self):
        self.metadata = SimpleNamespace(chunk_timing=[])

    def __len__(self):
        return 0

    def __getitem__(self, idx):
        raise IndexError(idx)


class _StubRuntimeJob:
    def __init__(self):
        self.call_count = 0
        self.last_args = None
        self.last_kwargs = None

    def result(self, *args, **kwargs):
        self.call_count += 1
        self.last_args = args
        self.last_kwargs = kwargs
        return _StubProgramResult()


class _StubAnalysisStage(AnalysisStage):
    @property
    def input_level(self):
        return RawData

    @property
    def output_level(self):
        return ModelData

    def _run(self, fit):
        fit[ModelData] = ModelData.from_arrays(
            parameter_indices=[],
            parameter_values=np.array([]),
            covariance=np.empty((0, 0)),
            time_lbs=np.empty(0, dtype="datetime64[us]"),
            time_ubs=np.empty(0, dtype="datetime64[us]"),
        )


@pytest.fixture()
def gate_set_cz():
    model_gate_set = ModelGateSet(2)
    model_gate_set.add_gate(ModelGate("CZ", [((0, 1), Clifford(CZGate()))]))
    model_gate_set.add_gate(ModelGate("P", qubit_idxs=range(2), prep_idxs=range(2)))
    model_gate_set.add_gate(ModelGate("M", qubit_idxs=range(2), meas_idxs=range(2)))
    return model_gate_set


@pytest.fixture()
def model(gate_set_cz):
    generators = {
        "CZ": QubitSparsePauliList(["ZI"]),
        "P": QubitSparsePauliList(["XI"]),
        "M": QubitSparsePauliList(["IX"]),
    }
    return PauliLindbladModel(gate_set_cz, generators)


@pytest.fixture()
def data_mapper():
    return ExecutorDataMapper(
        item_sequence_indices=[],
        creg_names=[],
        measurement_maps=[],
        instruction_sequences=[],
        num_randomizations=1,
    )


@pytest.fixture()
def experiment_schema(data_mapper, model):
    return ExperimentSchema(data_mapper=data_mapper, paths=[], model=model)


@pytest.fixture()
def stub_runtime_job():
    return _StubRuntimeJob()


@pytest.fixture()
def analysis_stage():
    return _StubAnalysisStage()


@pytest.fixture()
def job(stub_runtime_job, experiment_schema, analysis_stage):
    return NoiseLearnerJob(stub_runtime_job, experiment_schema, analysis_stage)


def test_noise_learner_job_init(job, stub_runtime_job):
    """Test NoiseLearnerJob init."""
    assert job.runtime_job is stub_runtime_job


def test_noise_learner_result_result(job, model):
    """Test NoiseLearnerJob.result returns sensible data."""
    result = job.result()
    assert isinstance(result, NoiseLearnerResult)
    assert result.fit.model is model
    assert isinstance(result.fit.raw_data, RawData)
    assert isinstance(result.fit.model_data, ModelData)


def test_noise_learner_job_result_calls_runtime_job_result(job, stub_runtime_job):
    """Test NoiseLearnerJob.result calls RuntimeJobV2.result."""
    job.result()
    assert stub_runtime_job.call_count == 1


def test_noise_learner_job_result_passes_args_to_runtime_job(job, stub_runtime_job):
    """Test NoiseLearnerJob.result passes args to RuntimeJobV2."""
    job.result("arg1", timeout=10)
    assert stub_runtime_job.last_args == ("arg1",)
    assert stub_runtime_job.last_kwargs == {"timeout": 10}
