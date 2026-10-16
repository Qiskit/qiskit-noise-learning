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

from samplomatic.quantum_program import QuantumProgramResult

from ...analysis import (
    AnalysisPipeline,
    AnalysisStage,
    ComputeObservables,
    CurveFitObservables,
    LegacySolve,
)
from ...circuit_generator import ExecutorCircuitGenerator
from ...data import RawData
from ...models import split_pauli_lindblad_model
from .vanilla_data_serialization import load
from .vanilla_fit import VanillaFit


def process_vanilla_result(
    result: QuantumProgramResult,
    raw_data_stage: AnalysisStage | None = None,
) -> VanillaFit:
    """Analyze the result of a program built by :func:`~.prepare_vanilla_program`.

    Applies the standard "vanilla" learning program analysis using curve fitting and
    :class:`~.LegacySolve` fit the model.

    Example::

        learned = process_vanilla_result(job.result())
        learned.noise_maps[0]

    Args:
        result: The result of a program built by :func:`~.prepare_vanilla_program`.
        raw_data_stage: An optional analysis stage to run on the raw data ahead of the standard
            analysis, typically to post-select on registers that a pass manager added during
            preparation. It must consume and produce :class:`~.RawData`.

    Returns:
        The noise learned for each instruction the program was built from, alongside the fit it was
        read out of.

    Raises:
        TypeError: If *raw_data_stage* is not an analysis stage.
        ValueError: If *raw_data_stage* does not consume and produce :class:`~.RawData`, if
            *result* holds no program items, if the result's passthrough data carries no vanilla
            protocol data or was written in an incompatible format version, if the result violates
            the solver's assumptions, or if the fitted model gives no noise for one of the gates
            the program was built from.
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

    if len(result) == 0:
        raise ValueError("The result holds no program items, so there is no data to analyze.")

    gate_names = load(result.passthrough_data).gate_names

    fit = ExecutorCircuitGenerator.collect(result)

    stages: list[AnalysisStage] = [ComputeObservables(), CurveFitObservables(), LegacySolve()]
    if raw_data_stage is not None:
        stages.insert(0, raw_data_stage)
    fit = AnalysisPipeline(*stages).run(fit)

    # The protocol builds the model itself, as a Pauli-Lindblad model, so splitting it out is a
    # formality that keeps the failure legible if that ever stops being true.
    noise_maps = split_pauli_lindblad_model(fit.model).model.to_pauli_lindblad_maps(
        fit.model_data, include_spam=True, restrict_to_qubit_idxs=True
    )
    if missing := [name for name in gate_names if name not in noise_maps]:
        raise ValueError(
            f"The fitted model gives no noise for the gate(s) {missing}, so there is nothing to "
            "report for the instruction(s) they were built from."
        )

    return VanillaFit(
        noise_maps=[noise_maps[name] for name in gate_names],
        gate_names=gate_names,
        fit=fit,
    )
