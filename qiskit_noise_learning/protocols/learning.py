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

"""The noise learning protocol."""

from collections.abc import Sequence

from qiskit.circuit import CircuitInstruction, QuantumRegister
from qiskit.providers import BackendV2
from qiskit.transpiler import PassManager
from qiskit_ibm_runtime.quantum_program import QuantumProgram
from qiskit_ibm_runtime.results import QuantumProgramResult
from samplomatic import InjectNoise
from samplomatic.utils import get_annotation

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
from ..experiment_builder import (
    BindFragmentDepths,
    CompleteSequences,
    EvenDepthVanillaPaths,
    Experiment,
    IdentifyRelations,
    MergeInstructionSequences,
    RankReducePaths,
    VanillaInstructionSequences,
)
from ..gate_sets import QiskitGateSet
from ..models import PauliLindbladModel


def prepare_learning_program(
    backend: BackendV2,
    instructions: Sequence[CircuitInstruction],
    num_randomizations: int = 32,
    shots_per_randomization: int = 128,
    fragment_depths: Sequence[int] = (0, 1, 2, 4, 16, 32),
    pass_manager: PassManager | None = None,
) -> QuantumProgram:
    """Build the quantum program of a standard noise learning experiment.

    The experiment learns a 2-local Pauli-Lindblad model of each given layer. Even-depth paths are
    generated for the model's generators, reduced to a maximal linearly independent set, and
    traversed by vanilla instruction sequences bound at each of *fragment_depths*.

    Each instruction becomes one gate of the model. A box carrying a noise injection annotation
    takes that annotation's reference as its gate name, and any other box is named automatically.
    The learned model comes back keyed by those names, so annotate the boxes to be identified in the
    result.

    .. note::
        *pass_manager* runs on each template circuit after that circuit has been built and
        **before** the passthrough data describing its classical registers is written into the
        program. Registers the pass manager adds are therefore recorded, and are recovered during
        processing. Running a pass manager over the returned program's circuits instead cannot
        achieve this: the passthrough data is fixed by then, and the added registers' outcomes would
        be unreadable.

    .. note::
        Which qubit each bit of an added register reads is recovered by scanning the template
        circuit for the measurements writing to it, and that scan only looks at the circuit's top
        level. Every bit of an added register must therefore be measured into exactly once by a
        measurement placed there; one placed inside a box is not found, and the register is rejected
        as unmeasured. This is a limit of that recovery rather than a requirement on the circuit:
        the experiment's own measurements do sit inside boxes, but the qubits they read come from
        the gate definitions instead of from the scan.

    .. note::
        Added registers carry no measurement flips, so their bits are untwirled. And a pass manager
        must not reorder qubits or rename the experiment's own measurement registers.

    Example::

        circuit = QuantumCircuit(backend.num_qubits)
        with circuit.box([Twirl(), InjectNoise("layer")]):
            circuit.cz(17, 27)

        program = prepare_learning_program(backend, [circuit[0]])

    Args:
        backend: The backend supplying the compilation target: the gate set, coupling map and qubit
            count that generated circuits are built against.
        instructions: The instructions to learn the noise of. Each instruction should contain a
            :class:`~qiskit.circuit.BoxOp` operation.
        num_randomizations: The number of randomizations to use per learning circuit.
        shots_per_randomization: The number of shots to use per randomization.
        fragment_depths: The fragment depths to use, that is, the number of repetitions of each
            path's repeatable fragment.
        pass_manager: An optional pass manager to apply to every template circuit generated, for
            example to add post-selection measurements. See the notes above.

    Returns:
        The program to submit, whose results :func:`process_learning_results` consumes.

    Raises:
        ValueError: If any instruction does not contain a ``BoxOp``, if *num_randomizations* or
            *shots_per_randomization* is less than one, if any entry of *fragment_depths* is
            negative, if ``backend.target`` does not support an operation of one of the boxes, or if
            a classical register added by *pass_manager* is not measured into exactly once.
    """
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

    # This register exists only to turn the instructions' qubits into integer indices.
    qreg = QuantumRegister(backend.num_qubits, name="q")
    qubit_subset = {qreg.index(qubit) for instr in instructions for qubit in instr.qubits}

    gate_set = QiskitGateSet(target=backend.target, qubit_subset=sorted(qubit_subset))
    for instr in instructions:
        inject_noise = get_annotation(instr.operation, InjectNoise)
        gate_set.add_box_as_gate(instr, name=None if inject_noise is None else inject_noise.ref)

    # The locality of the gate model is fixed by the protocol rather than exposed as an argument.
    fidelity_model = PauliLindbladModel.k_local(gate_set, k=2)

    builder = (
        EvenDepthVanillaPaths()
        # RankReducePaths is load-bearing, not an optimization: it discards the duplicate row each
        # conjugate Pauli pair contributes, without which LegacySolve in process_learning_results
        # receives two independent estimates of one pair fidelity and rejects them.
        + RankReducePaths()
        + VanillaInstructionSequences()
        # IdentifyRelations must precede MergeInstructionSequences, which requires relations.
        # VanillaInstructionSequences does not populate them, unlike GenerateInstructionSequences.
        + IdentifyRelations()
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

    return ExecutorCircuitGenerator(gate_set, pass_manager=pass_manager).generate(experiment)


def process_learning_results(
    results: QuantumProgramResult,
    raw_data_stage: AnalysisStage | None = None,
) -> Fit:
    """Analyze the results of a program built by :func:`prepare_learning_program`.

    Observables are computed from the raw data, exponentials are fitted to them across the fragment
    depths, and the resulting pair fidelities are solved for the model's generator rates. The
    returned fit holds the data at every level of that chain, together with the model, paths,
    instruction sequences and relations recovered from the results' passthrough data.

    .. note::
        The analysis applied here assumes the experiment that :func:`prepare_learning_program`
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
        results: The results of a program built by :func:`prepare_learning_program`.
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
