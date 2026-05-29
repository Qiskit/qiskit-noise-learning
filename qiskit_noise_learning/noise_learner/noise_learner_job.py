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

from qiskit_ibm_runtime import RuntimeJobV2

from ..analysis import AnalysisStage
from ..circuit_generator.executor_circuit_generator import ExecutorCircuitGenerator
from ..circuit_generator.executor_data_mapper import ExecutorDataMapper
from .noise_learner_result import NoiseLearnerResult


class NoiseLearnerJob:
    """A noise learner job.

    This class is a wrapper around :class:`~qiskit_ibm_runtime.RuntimeJobV2` that also includes
    attributes to analyze the outcome of a noise learning experiment.

    Args:
        runtime_job: The runtime job.
        data_mapper: The data mapper describing the experiment layout.
        analysis_stage: The analysis stage to process the data.
    """

    def __init__(
        self,
        runtime_job: RuntimeJobV2,
        data_mapper: ExecutorDataMapper,
        analysis_stage: AnalysisStage,
    ):
        self._runtime_job = runtime_job
        self._data_mapper = data_mapper
        self._analysis_stage = analysis_stage

    @property
    def runtime_job(self) -> RuntimeJobV2:
        """The runtime job."""
        return self._runtime_job

    def result(self, *args, **kwargs) -> NoiseLearnerResult:
        """Compute the result of the noise learning job.

        This method forwards arguments to :meth:`~qiskit_ibm_runtime.RuntimeJobV2.result`.
        """
        raw_result = self._runtime_job.result(*args, **kwargs)
        fit = ExecutorCircuitGenerator.collect(raw_result, self._data_mapper)
        return NoiseLearnerResult(self._analysis_stage.run(fit))
