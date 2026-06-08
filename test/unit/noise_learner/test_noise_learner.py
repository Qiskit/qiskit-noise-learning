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

from unittest.mock import MagicMock, patch

import pytest
from qiskit.circuit import BoxOp, QuantumCircuit
from qiskit_ibm_runtime.quantum_program import QuantumProgram

from qiskit_noise_learning.circuit_generator import ExecutorDataMapper
from qiskit_noise_learning.noise_learner import LearningOptions, NoiseLearner
from qiskit_noise_learning.noise_learner.noise_learner_job import NoiseLearnerJob


class _MockTarget:
    pass


class _MockBackend:
    @property
    def target(self):
        return _MockTarget()


def _make_box_instruction(num_qubits=2):
    """Create a BoxOp CircuitInstruction."""
    inner = QuantumCircuit(num_qubits)
    inner.cx(0, 1)
    qc = QuantumCircuit(num_qubits)
    qc.append(BoxOp(inner), range(num_qubits))
    return qc.data[0]


def _make_non_box_instruction(num_qubits=2):
    """Create a non-BoxOp CircuitInstruction."""
    qc = QuantumCircuit(num_qubits)
    qc.cx(0, 1)
    return qc.data[0]


@pytest.fixture()
def backend():
    return _MockBackend()


@pytest.fixture()
def options():
    return LearningOptions(num_randomizations=4, shots_per_randomizations=16, depths=[0, 1, 2])


@pytest.fixture()
def learner(backend, options):
    return NoiseLearner(backend, options)


def test_noise_learner_init(backend):
    """Test NoiseLearner construction."""
    learner = NoiseLearner(backend, None)
    assert learner.options == LearningOptions()
    assert learner.backend is backend
    assert isinstance(learner.backend, _MockBackend)
    assert isinstance(learner.options, LearningOptions)


def test_noise_learner_run_rejects_non_box_instruction(learner):
    """Test instruction validation in NoiseLearner.run."""
    instr = _make_non_box_instruction()
    with pytest.raises(ValueError, match="BoxOps"):
        learner.run([instr])

    box_instr = _make_box_instruction()
    with pytest.raises(ValueError, match="BoxOps"):
        learner.run([box_instr, instr])


@patch("qiskit_noise_learning.noise_learner.noise_learner.Executor")
def test_noise_learner_run_orchestration(mock_executor_cls, learner):
    """Test run() orchestration with monkeypatched _generate."""
    generate_calls = []

    fake_program = MagicMock(spec=QuantumProgram)
    fake_data_mapper = MagicMock(spec=ExecutorDataMapper)
    fake_job = MagicMock()
    mock_executor_cls.return_value.run.return_value = fake_job

    def fake_generate(instructions):
        generate_calls.append(instructions)
        return (fake_program, fake_data_mapper)

    learner._generate = fake_generate  # noqa: SLF001

    box_instr = _make_box_instruction()
    result = learner.run([box_instr])

    assert isinstance(result, NoiseLearnerJob)
    assert len(generate_calls) == 1
    mock_executor_cls.return_value.run.assert_called_once_with(fake_program)
    assert result._data_mapper is fake_data_mapper  # noqa: SLF001
    assert result._analysis_stage is learner._analyzer  # noqa: SLF001
