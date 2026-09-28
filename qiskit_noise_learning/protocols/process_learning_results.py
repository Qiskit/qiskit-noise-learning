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

"""Analysis of noise learning results."""

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


def process_learning_results(
    results: QuantumProgramResult,
    raw_data_stage: AnalysisStage | None = None,
) -> Fit:
    """Analyze the results of a program built by :func:`~.prepare_learning_program`.

    Observables are computed from the raw data, exponentials are fitted to them across the fragment
    depths, and the resulting pair fidelities are solved for the model's generator rates. The
    returned fit holds the data at every level of that chain, together with the model, paths,
    instruction sequences and relations recovered from the results' passthrough data.

    .. note::
        The analysis applied here assumes the experiment that :func:`~.prepare_learning_program`
        builds. In particular the solver requires the linearly independent path set that the
        preparation's rank reduction produces, and rejects results in which a pair fidelity is
        measured more than once. To analyze a hand-built experiment, collect it with
        :meth:`~.ExecutorCircuitGenerator.collect` and compose an :class:`~.AnalysisPipeline`
        directly.

    Example::

        fit = process_learning_results(job.result())
        noise_maps = split_pauli_lindblad_model(fit.model).model.to_pauli_lindblad_maps(
            fit.model_data
        )

    Args:
        results: The results of a program built by :func:`~.prepare_learning_program`.
        raw_data_stage: An optional analysis stage to run on the raw data ahead of the standard
            analysis, typically to post-select on registers that a pass manager added during
            preparation. It must consume and produce :class:`~.RawData`. An
            :class:`~.AnalysisPipeline` is itself a stage, so a chain composed with ``+`` is
            accepted, and is spliced in ahead of the standard stages rather than nested.

    Returns:
        The fit, holding the data at every analysis level.

    Raises:
        TypeError: If *raw_data_stage* is not an analysis stage.
        ValueError: If *raw_data_stage* does not consume and produce :class:`~.RawData`, if the
            results' passthrough data was written in an incompatible format version, or if the
            results violate the solver's assumptions.
    """
    # Validated before the results are touched, so the cheap error does not require deserializing a
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

    fit = ExecutorCircuitGenerator.collect(results)

    stages: list[AnalysisStage] = [ComputeObservables(), CurveFitObservables(), LegacySolve()]
    if raw_data_stage is not None:
        stages.insert(0, raw_data_stage)

    return AnalysisPipeline(*stages).run(fit)
