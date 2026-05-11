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

from dataclasses import dataclass

from qiskit_ibm_runtime import RuntimeJobV2

from qiskit_noise_learning.data import RawData

from ..analysis import AnalysisStage, Fit
from ..circuit_generator import ExecutorCircuitGenerator, ExecutorDataMapper
from ..models import PauliLindbladModel
from ..sequences import Path
from .noise_learner_result import NoiseLearnerResult


@dataclass
class ExperimentSchema:
    """A schema of the experiment."""

    data_mapper: ExecutorDataMapper
    paths: list[Path]
    model: PauliLindbladModel


class NoiseLearnerJob:
    """A noise learner job.

    This class is a wrapper around :class:`~qiskit_ibm_runtime.RuntimeJobV2` that also includes
    attributes to analyze the outcome of a noise learning experiment.

    Args:
        runtime_job: The runtime job.
        experiment_schema: A description of the noise learning experiment.
        analysis_stage: The analysis stage to process the data.
    """

    def __init__(
        self,
        runtime_job: RuntimeJobV2,
        experiment_schema: ExperimentSchema,
        analysis_stage: AnalysisStage,
    ):
        self._runtime_job = runtime_job
        self._experiment_schema = experiment_schema
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
        raw_data = ExecutorCircuitGenerator.collect(raw_result, self._experiment_schema.data_mapper)
        fit = Fit(model=self._experiment_schema.model, paths=self._experiment_schema.paths)
        fit[RawData] = raw_data
        return NoiseLearnerResult(self._analysis_stage.run(fit))
