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

"""Preparation of a noise learning program."""

from collections.abc import Sequence

from qiskit.circuit import CircuitInstruction, QuantumRegister
from qiskit.providers import BackendV2
from qiskit.transpiler import PassManager
from qiskit_ibm_runtime.quantum_program import QuantumProgram
from samplomatic import InjectNoise
from samplomatic.utils import get_annotation

from ..circuit_generator import ExecutorCircuitGenerator
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
from ..models import PauliLindbladModel


def _measures(box_instr: CircuitInstruction) -> bool:
    """Whether any operation of a box is a measurement."""
    return any(
        len(instr.qubits) == 1 and instr.operation.name.startswith("meas")
        for instr in box_instr.operation.blocks[0]
    )


def prepare_learning_program(
    backend: BackendV2,
    instructions: Sequence[CircuitInstruction],
    num_randomizations: int = 32,
    shots_per_randomization: int = 128,
    fragment_depths: Sequence[int] = (0, 1, 2, 4, 16, 32),
    creg_prefix: str = "meas",
    local_clifford_ref_prefix: str = "c",
    pass_manager: PassManager | None = None,
) -> QuantumProgram:
    """Build the quantum program of a standard noise learning experiment.

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

    Example::

        circuit = QuantumCircuit(backend.num_qubits)
        with circuit.box([Twirl(), InjectNoise("layer")]):
            circuit.cz(17, 27)

        program = prepare_learning_program(backend, [circuit[0]])

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
        The program to submit, whose result :func:`~.process_learning_result` consumes.

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
    # A negative depth is not merely out of range: -1 is the sentinel that CurveFitObservables
    # writes for a decay row, so it would silently collide with fitted output downstream.
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
    for idx, instr in enumerate(instructions):
        inject_noise = get_annotation(instr.operation, InjectNoise)
        name = gate_set.add_box_as_gate(
            instr, name=None if inject_noise is None else inject_noise.ref
        )
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

    return ExecutorCircuitGenerator(
        gate_set,
        creg_prefix=creg_prefix,
        local_clifford_ref_prefix=local_clifford_ref_prefix,
        pass_manager=pass_manager,
    ).generate(experiment)
