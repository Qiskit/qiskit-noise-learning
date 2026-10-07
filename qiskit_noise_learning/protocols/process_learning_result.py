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

"""Analysis of a noise learning result."""

from qiskit_ibm_runtime.results import QuantumProgramResult

from ..analysis import (
    AnalysisPipeline,
    AnalysisStage,
    ComputeObservables,
    CurveFitObservables,
    Fit,
    LegacySolve,
)
from ..circuit_generator import ExecutorCircuitGenerator
from ..data import RawData


def process_learning_result(
    result: QuantumProgramResult,
    raw_data_stage: AnalysisStage | None = None,
) -> Fit:
    """Analyze the result of a program built by :func:`~.prepare_learning_program`.

    Applies the standard "vanilla" learning program analysis using curve fitting and
    :class:`~.LegacySolve` fit the model.

    Example::

        fit = process_learning_result(job.result())
        noise_maps = split_pauli_lindblad_model(fit.model).model.to_pauli_lindblad_maps(
            fit.model_data
        )

    Args:
        result: The result of a program built by :func:`~.prepare_learning_program`.
        raw_data_stage: An optional analysis stage to run on the raw data ahead of the standard
            analysis, typically to post-select on registers that a pass manager added during
            preparation. It must consume and produce :class:`~.RawData`.

    Returns:
        The fit, holding the data at every analysis level.

    Raises:
        TypeError: If *raw_data_stage* is not an analysis stage.
        ValueError: If *raw_data_stage* does not consume and produce :class:`~.RawData`, if the
            result's passthrough data was written in an incompatible format version, or if the
            result violates the solver's assumptions.
    """
    # Validated before the result is touched, so the cheap error does not require deserializing a
    # payload first.
    if raw_data_stage is not None:
        if not isinstance(raw_data_stage, AnalysisStage):
            raise TypeError(
                f"raw_data_stage must be an AnalysisStage, but got {type(raw_data_stage).__name__}."
            )
        if raw_data_stage.input_level is not RawData or raw_data_stage.output_level is not RawData:
            raise ValueError(
                f"raw_data_stage must consume and produce RawData, but {raw_data_stage!r} maps "
                f"{raw_data_stage.input_level.__name__} to {raw_data_stage.output_level.__name__}."
            )

    fit = ExecutorCircuitGenerator.collect(result)

    stages: list[AnalysisStage] = [ComputeObservables(), CurveFitObservables(), LegacySolve()]
    if raw_data_stage is not None:
        stages.insert(0, raw_data_stage)

    return AnalysisPipeline(*stages).run(fit)
