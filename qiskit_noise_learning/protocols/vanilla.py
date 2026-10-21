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

"""The vanilla learning protocol."""

from collections import defaultdict
from collections.abc import Callable, Iterator, Sequence

import numpy as np
from qiskit.circuit import CircuitInstruction, QuantumRegister
from qiskit.providers import BackendV2
from qiskit.quantum_info import PauliLindbladMap
from qiskit.transpiler import PassManager
from samplomatic import InjectNoise
from samplomatic.quantum_program import QuantumProgram, QuantumProgramResult
from samplomatic.utils import get_annotation

from ..analysis import (
    AnalysisPipeline,
    AnalysisStage,
    ComputeObservables,
    CurveFitObservables,
    Fit,
    LegacySolve,
    suffix_creg_identifier,
)
from ..circuit_generator import ExecutorCircuitGenerator
from ..circuit_generator.executor.data_mapper_serialization import PAYLOAD_KEY, PROTOCOL_KEY
from ..data import RawData
from ..experiment_builder import (
    BindFragmentDepths,
    CompleteSequences,
    EvenDepthVanillaPaths,
    Experiment,
    GenerateInstructionSequences,
    IdentifyRelations,
    MergeInstructionSequences,
    RankReducePaths,
    SPAMPaths,
    VanillaInstructionSequences,
)
from ..gate_sets import QiskitGateSet
from ..models import PauliLindbladModel, split_pauli_lindblad_model

VANILLA_KEY = "vanilla"
"""Key for protocol passthrough data."""

VERSION = "1"
"""Passthrough data version."""


def _measures(box_instr: CircuitInstruction) -> bool:
    """Whether any operation of a box is a measurement."""
    return any(
        len(instr.qubits) == 1 and instr.operation.name.startswith("meas")
        for instr in box_instr.operation.blocks[0]
    )


def prepare_vanilla_program(
    backend: BackendV2,
    instructions: Sequence[CircuitInstruction],
    num_randomizations: int = 32,
    shots_per_randomization: int = 128,
    fragment_depths: Sequence[int] = (0, 1, 2, 4, 16, 32),
    creg_prefix: str = "meas",
    local_clifford_ref_prefix: str = "c",
    pass_manager: PassManager | None = None,
) -> QuantumProgram:
    """Build the quantum program containing the vanilla learning experiment.

    The experiment is setup to learn a 2-local Pauli-Lindblad model for each layer defined in
    ``instructions`` via the "vanilla" learning protocol, with paths generated via
    :class:`~.EvenDepthVanillaPaths` and instruction sequences generated via
    :class:`~.VanillaInstructionSequences`, as well as a SPAM noise model with perfect preparation
    and a 1-local measurement noise model.

    Each instruction becomes one gate of the model. A box carrying a noise injection annotation
    takes that annotation's reference as its gate name, and any other box is named automatically.
    The learned model comes back keyed by those names, so annotate the boxes to be identified in the
    result.

    If an all-measurement box is supplied in ``instructions`` (i.e. one that measures every qubit
    appearing in all supplied instructions), that will be treated as the measurement layer. If no
    such layer is supplied, a default one will be built with name ``"M"``.

    The program records which gate each instruction became, in the order the instructions were
    given, for :func:`~.process_vanilla_result` to report the learned noise against.

    Example::

        circuit = QuantumCircuit(backend.num_qubits)
        with circuit.box([Twirl(), InjectNoise("layer")]):
            circuit.cz(17, 27)

        program = prepare_vanilla_program(backend, [circuit[0]])

    .. note::

        The arguments ``creg_prefix``, ``local_clifford_ref_prefix``, and ``pass_manager`` are all
        directly passed to :class:`~.ExecutorCircuitGenerator`.

    Args:
        backend: The backend supplying the compilation target: the gate set, coupling map and qubit
            count that generated circuits are built against.
        instructions: The instructions to learn the noise of. Each instruction should contain a
            :class:`~qiskit.circuit.BoxOp` operation, and must be self-inverse, with the exception
            of a box that measures all relevant qubits.
        num_randomizations: The number of randomizations to use per learning circuit.
        shots_per_randomization: The number of shots to use per randomization.
        fragment_depths: The fragment depths to use, that is, the number of repetitions of each
            path's repeatable fragment.
        creg_prefix: The prefix assigned to all creg names used in learning experiment measurements.
            Defaults to ``"meas"``.
        local_clifford_ref_prefix: The prefix assigned to all local Clifford parameter references
            in template circuits. Defaults to ``"c"``.
        pass_manager: An optional pass manager to apply to every template circuit generated.

    Returns:
        The program to submit, whose result :func:`~.process_vanilla_result` consumes.

    Raises:
        ValueError: If *instructions* is empty, if any instruction does not contain a ``BoxOp``, if
            *num_randomizations* or *shots_per_randomization* is less than one, if any entry of
            *fragment_depths* is negative, if more than one instruction measures, if a measuring
            instruction also contains other operations or does not measure every qubit that
            *instructions* act on, if ``backend.target`` does not support an operation of one of
            the boxes, or if a classical register added by *pass_manager* is not measured into
            exactly once.
    """
    if not instructions:
        raise ValueError("instructions must contain at least one instruction, but got none.")

    for instr in instructions:
        if instr.operation.name != "box":
            raise ValueError(f"All instructions must be BoxOps, got '{instr.operation.name}'.")

    if num_randomizations < 1:
        raise ValueError(f"num_randomizations must be at least 1, but got {num_randomizations}.")
    if shots_per_randomization < 1:
        raise ValueError(
            f"shots_per_randomization must be at least 1, but got {shots_per_randomization}."
        )
    fragment_depths = list(fragment_depths)
    if any(depth < 0 for depth in fragment_depths):
        raise ValueError(f"fragment_depths must all be non-negative, but got {fragment_depths}.")

    # The experiment should contain exactly 1 or 0 measuring gates
    meas_instr_idxs = [idx for idx, instr in enumerate(instructions) if _measures(instr)]
    if len(meas_instr_idxs) > 1:
        raise ValueError(
            f"At most one instruction may measure, but instructions {meas_instr_idxs} all do."
        )

    # This register exists only to turn the instructions' qubits into integer indices.
    qreg = QuantumRegister(backend.num_qubits, name="q")
    qubit_subset = {qreg.index(qubit) for instr in instructions for qubit in instr.qubits}

    gate_set = QiskitGateSet(
        target=backend.target, qubit_subset=sorted(qubit_subset), add_default_spam=False
    )
    if not meas_instr_idxs:
        gate_set.add_measurement(name="M")
    gate_set.add_preparation(name="P")

    meas_name = "M"
    gate_names = []
    for idx, instr in enumerate(instructions):
        inject_noise = get_annotation(instr.operation, InjectNoise)
        name = gate_set.add_box_as_gate(
            instr, name=None if inject_noise is None else inject_noise.ref
        )
        gate_names.append(name)
        if idx in meas_instr_idxs:
            meas_name = name

    fidelity_model = PauliLindbladModel.k_local(gate_set, k=2, gate_k={meas_name: 1, "P": 0})
    meas_gate = fidelity_model.gate_set[meas_name]

    builder = (
        EvenDepthVanillaPaths(meas_gate=meas_gate)
        + RankReducePaths()
        + VanillaInstructionSequences(meas_gate=meas_gate)
        + IdentifyRelations()
        + SPAMPaths(meas_gate=meas_gate)
        + GenerateInstructionSequences()
        + MergeInstructionSequences()
        + CompleteSequences()
        + BindFragmentDepths(fragment_depths)
    )
    experiment = builder.run(
        Experiment(
            fidelity_model=fidelity_model,
            shots=shots_per_randomization,
            randomizations=num_randomizations,
        )
    )

    program = ExecutorCircuitGenerator(
        gate_set,
        creg_prefix=creg_prefix,
        local_clifford_ref_prefix=local_clifford_ref_prefix,
        pass_manager=pass_manager,
    ).generate(experiment)
    program.passthrough_data[PAYLOAD_KEY][PROTOCOL_KEY] = {
        VANILLA_KEY: {"version": VERSION, "gate_names": gate_names}
    }
    return program


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
            f"version {VERSION!r}."
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

    noise_maps = split_pauli_lindblad_model(fit.model).model.to_pauli_lindblad_maps(
        fit.model_data, include_spam=True, restrict_to_qubit_idxs=True
    )
    if missing := [name for name in gate_names if name not in noise_maps]:
        raise ValueError(
            f"The fitted model gives no noise for the gate(s) {missing}, so there is nothing to "
            "report for the instruction(s) they were built from."
        )

    return [noise_maps[name] for name in gate_names], gate_names, fit


# ---------------------------------------------------------------------------------------------
# Unsupported helper functions
#
# The functions below are private and unsupported: they exist to be consumed by
# qiskit-ibm-runtime, which reports their values in the post-selection metadata of a noise
# learner V3 result. They are not part of this package's public interface, carry no stability
# guarantee, and may change or be removed without notice or a deprecation period.
# ---------------------------------------------------------------------------------------------


def _fraction_kept(raw_data: RawData) -> dict[int, float]:
    """Compute the fraction of shots surviving post-selection, per fragment depth.

    Mirrors the ``fraction_kept`` field that the noise learner V3 result schema carries in its
    post-selection metadata.

    Each randomization contributes ``1 - data_mask.mean()``, since ``data_mask`` marks the shots to
    exclude, and the randomizations sharing a fragment depth are averaged. The sequences with no
    repeatable fragment, which carry the preparation and measurement noise, are bound at fragment
    depth zero, so they land in the depth-zero entry alongside the depth-zero learning sequences
    rather than being reported separately.

    Data carrying no mask at all counts as fully kept, giving every fragment depth a fraction of
    ``1.0``, so that the depths reported do not depend on whether a post-selection stage has run.

    Args:
        raw_data: The raw data, after any post-selection stage has run. Without one every entry is
            ``1.0``, since nothing has been masked.

    Returns:
        A mapping from fragment depth to the average fraction of shots kept, between 0 and 1.
    """
    fractions = defaultdict(list)

    for _, node in raw_data.datatree.items():
        dataset = node.dataset
        if "fragment_depth" not in dataset:
            continue

        depths = dataset["fragment_depth"].values
        if "data_mask" in dataset:
            kept = 1.0 - dataset["data_mask"].values.mean(axis=-1)
        else:
            kept = np.ones(len(depths))

        for depth, fraction in zip(depths, kept):
            fractions[int(depth)].append(float(fraction))

    return {depth: float(np.mean(values)) for depth, values in sorted(fractions.items())}


def _success_rates(
    raw_data: RawData,
    creg_identifier: Callable[[list[str]], Iterator[tuple[str, ...]]] | None = None,
) -> dict[int, dict[int, float]]:
    """Compute the per-qubit bit-flip success rate, per fragment depth.

    Mirrors the ``success_rates`` field that the noise learner V3 result schema carries in its
    post-selection metadata: the fraction of shots in which post-selection successfully flipped
    each qubit. Where :func:`_fraction_kept` gives one number per fragment depth, this resolves it
    per qubit, so that a single qubit responsible for a low yield is visible.

    A bit counts as successfully flipped under the rule :class:`~.FlipPostSelect` uses to decide
    failure, read the other way around. For a group of two registers ``(base, ps)`` a bit succeeded
    if its outcome differs between the two, and for a group of one register a bit succeeded if its
    outcome is ``False``. Both read the outcome ``data ^ measurement_flips`` on each side of the
    comparison, undoing the flip introduced by measurement twirling so that what is compared is the
    physical measurement outcomes; the rates are therefore independent of the twirl each register
    drew. As in :func:`_fraction_kept`, the sequences with no repeatable fragment are bound at
    fragment depth zero and so are reported in the depth-zero entry.

    Qubits with no post-selection register to compare against get a rate of ``1.0``, as do all
    qubits when the data carries no post-selection registers at all, so that every fragment depth
    reports a rate for every measured qubit either way.

    Args:
        raw_data: The raw data. This reads ``data`` and ``measurement_flips`` rather than
            ``data_mask``, so the rates do not depend on whether a post-selection stage has run,
            nor on the strategy it ran with.
        creg_identifier: A callable that, given the register names present, returns an iterator
            over tuples of one or two register names to post-select on, as the ``creg_identifier``
            argument of :class:`~.FlipPostSelect`. Defaults to the same
            :func:`~.suffix_creg_identifier` default that class uses.

    Returns:
        A mapping from fragment depth to a mapping from physical qubit index to the average
        success rate of that qubit, between 0 and 1.

    Raises:
        ValueError: If the identifier yields a group of two registers that do not measure the same
            qubits in the same classical bit order, or a group of other than one or two registers.
    """
    creg_identifier = creg_identifier or suffix_creg_identifier()

    rates = defaultdict(lambda: defaultdict(list))

    for _, node in raw_data.datatree.items():
        dataset = node.dataset
        if "data" not in dataset:
            continue

        outcomes = dataset["data"].values ^ dataset["measurement_flips"].values[:, np.newaxis, :]
        boundaries = dataset.attrs["creg_bit_boundaries"]
        clbit_qubit_idxs = dataset.attrs["clbit_qubit_idxs"]
        depths = dataset["fragment_depth"].values

        # Every qubit this leaf measures starts out fully successful, so that a qubit without a
        # post-selection register, and every qubit of a leaf without any, still reports a rate.
        checked_qubit_idxs = set()
        groups = []
        for names in creg_identifier(dataset.attrs["creg_names"]):
            if len(names) == 1:
                (name,) = names
                start, end = boundaries[name]
                flipped = ~outcomes[:, :, start:end]
                qubit_idxs = clbit_qubit_idxs[name]
            elif len(names) == 2:
                base_name, ps_name = names
                base_qubit_idxs = clbit_qubit_idxs[base_name]
                if not np.array_equal(base_qubit_idxs, clbit_qubit_idxs[ps_name]):
                    raise ValueError(
                        f"Registers '{base_name}' and '{ps_name}' must measure the same qubits in "
                        "the same classical bit order."
                    )
                base_start, base_end = boundaries[base_name]
                ps_start, ps_end = boundaries[ps_name]
                flipped = outcomes[:, :, base_start:base_end] != outcomes[:, :, ps_start:ps_end]
                qubit_idxs = base_qubit_idxs
            else:
                raise ValueError(
                    "The register identifier must yield tuples of one or two register names, but "
                    f"got {tuple(names)}."
                )

            checked_qubit_idxs.update(int(idx) for idx in qubit_idxs)
            # Average over shots, leaving one rate per randomization per bit.
            groups.append((qubit_idxs, flipped.mean(axis=1)))

        measured_qubit_idxs = {
            int(idx) for idxs in clbit_qubit_idxs.values() for idx in idxs
        } - checked_qubit_idxs

        for depth in {int(depth) for depth in depths}:
            for qubit_idx in measured_qubit_idxs:
                rates[depth][qubit_idx].append(1.0)

        for qubit_idxs, per_randomization in groups:
            for depth, randomization_rates in zip(depths, per_randomization):
                for qubit_idx, rate in zip(qubit_idxs, randomization_rates):
                    rates[int(depth)][int(qubit_idx)].append(float(rate))

    return {
        depth: {
            qubit_idx: float(np.mean(values)) for qubit_idx, values in sorted(per_qubit.items())
        }
        for depth, per_qubit in sorted(rates.items())
    }
