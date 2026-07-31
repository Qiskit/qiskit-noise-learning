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

"""Noise learner job."""

from typing import Protocol, runtime_checkable

from qiskit_ibm_runtime.results import QuantumProgramResult

from ..analysis import AnalysisStage
from ..circuit_generator.executor_circuit_generator import ExecutorCircuitGenerator
from ..circuit_generator.executor_data_mapper import ExecutorDataMapper
from .noise_learner_result import NoiseLearnerResult


@runtime_checkable
class ProgramJob(Protocol):
    """The job role that :class:`NoiseLearnerJob` wraps.

    Only :meth:`result` is required, so both :class:`~qiskit_ibm_runtime.RuntimeJobV2` and the
    job type returned by the Aer executor qualify without either declaring conformance.
    """

    def result(self, *args, **kwargs) -> QuantumProgramResult:
        """Return the result of the executed program.

        Returns:
            The program's :class:`~qiskit_ibm_runtime.results.QuantumProgramResult`.
        """
        ...


class NoiseLearnerJob:
    """A noise learner job.

    This class is a wrapper around the job returned by the executor that ran the experiment,
    which also includes attributes to analyze the outcome of a noise learning experiment.

    Args:
        runtime_job: The job returned by the executor the program was submitted to. This is a
            :class:`~qiskit_ibm_runtime.RuntimeJobV2` unless a different executor was supplied
            to :class:`NoiseLearner`.
        data_mapper: The data mapper describing the experiment layout.
        analysis_stage: The analysis stage to process the data.
    """

    def __init__(
        self,
        runtime_job: ProgramJob,
        data_mapper: ExecutorDataMapper,
        analysis_stage: AnalysisStage,
    ):
        self._runtime_job = runtime_job
        self._data_mapper = data_mapper
        self._analysis_stage = analysis_stage

    @property
    def runtime_job(self) -> ProgramJob:
        """The job returned by the executor the program was submitted to."""
        return self._runtime_job

    def result(self, *args, **kwargs) -> NoiseLearnerResult:
        """Compute the result of the noise learning job.

        This method forwards arguments to :meth:`ProgramJob.result`.
        """
        raw_result = self._runtime_job.result(*args, **kwargs)
        fit = ExecutorCircuitGenerator.collect(raw_result, self._data_mapper)
        return NoiseLearnerResult(self._analysis_stage.run(fit))
