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

from qiskit.quantum_info import PauliLindbladMap
from samplomatic.quantum_program import QuantumProgramResult

from ...analysis import (
    AnalysisPipeline,
    AnalysisStage,
    ComputeObservables,
    CurveFitObservables,
    Fit,
    LegacySolve,
)
from ...circuit_generator import ExecutorCircuitGenerator
from ...circuit_generator.executor.data_mapper_serialization import PAYLOAD_KEY, PROTOCOL_KEY
from ...data import RawData
from ...models import split_pauli_lindblad_model
from .prepare_vanilla_program import VANILLA_KEY, VERSION


def _recorded_gate_names(passthrough_data: dict) -> list[str]:
    """Retrieve the gate name passthrough data.

    This function also validates the passthrough data.

    Args:
        passthrough_data: The passthrough data from the result of a program it built.

    Returns:
        The list of gate names recorded in the passthrough data.

    Raises:
        ValueError: If the passthrough data carries no protocol data, if the protocol data it
            carries belongs to a different protocol, or if the passthrough data is from an
            incompatible version.
    """
    # Checked rather than assumed: this data has been through execution, and whoever submitted the
    # program chose what else went into it.
    payload = passthrough_data.get(PAYLOAD_KEY) if isinstance(passthrough_data, dict) else None
    protocols = payload.get(PROTOCOL_KEY) if isinstance(payload, dict) else None
    if not isinstance(protocols, dict):
        raise ValueError(
            f"Found no protocol data to read: the passthrough data has no '{PROTOCOL_KEY}' entry "
            f"under '{PAYLOAD_KEY}'. This result may come from a program that "
            "prepare_vanilla_program did not build."
        )

    record = protocols.get(VANILLA_KEY)
    if not isinstance(record, dict):
        raise ValueError(
            f"This program was built by a different protocol: its protocol data holds "
            f"{sorted(protocols)} rather than a '{VANILLA_KEY}' entry, so it has to be processed "
            "by the function pair that built it."
        )

    if (version := record.get("version")) != VERSION:
        raise ValueError(
            f"This protocol data is at version {version!r}, but this release requires "
            f"version {VERSION}."
        )

    return [str(name) for name in record["gate_names"]]


def process_vanilla_result(
    result: QuantumProgramResult,
    raw_data_stage: AnalysisStage | None = None,
) -> tuple[list[PauliLindbladMap], list[str], Fit]:
    """Analyze the result of a program built by :func:`~.prepare_vanilla_program`.

    Applies the standard "vanilla" learning program analysis using curve fitting and
    :class:`~.LegacySolve` fit the model.

    Example::

        noise_maps, gate_names, fit = process_vanilla_result(job.result())
        noise_maps[0]

    Args:
        result: The result of a program built by :func:`~.prepare_vanilla_program`.
        raw_data_stage: An optional analysis stage to run on the raw data ahead of the standard
            analysis, typically to post-select on registers that a pass manager added during
            preparation. It must consume and produce :class:`~.RawData`.

    Returns:
        A ``(noise_maps, gate_names, fit)`` tuple, whose first two entries are parallel to the
        ``instructions`` submitted to :func:`~.prepare_vanilla_program`. I.e. for instruction
        ``idx``, ``noise_maps[idx]`` is the learned noise map, and ``gate_names[idx]`` is the name
        assigned to the instruction used in the returned fit. A noise map acts on the qubits of its
        instruction, its qubit ``j`` being the ``j``-th of the instruction in ascending
        physical-qubit order.

    Raises:
        TypeError: If *raw_data_stage* is not an analysis stage.
        ValueError: If *raw_data_stage* does not consume and produce :class:`~.RawData`, if
            *result* holds no program items, if the result's passthrough data carries no
            vanilla protocol data or carries it at an incompatible version, if the result violates
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

    gate_names = _recorded_gate_names(result.passthrough_data)

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

    return [noise_maps[name] for name in gate_names], gate_names, fit
