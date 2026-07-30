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

"""Noise learner implementation."""

from collections.abc import Sequence
from typing import Protocol, TypeAlias

from qiskit.circuit import CircuitInstruction as _CircuitInstruction
from qiskit.circuit import QuantumRegister
from qiskit.providers import BackendV2
from qiskit_ibm_runtime import Executor
from qiskit_ibm_runtime.quantum_program import QuantumProgram
from samplomatic import InjectNoise
from samplomatic.utils import get_annotation

from ..analysis import (
    AnalysisPipeline,
    ComputeObservables,
    CurveFitObservables,
    NNLSSolve,
)
from ..circuit_generator import ExecutorCircuitGenerator, ExecutorDataMapper
from ..experiment_builder import (
    BindFragmentDepths,
    CompleteSequences,
    EvenDepthVanillaPaths,
    Experiment,
    GenerateInstructionSequences,
    IdentifyRelations,
    MergeInstructionSequences,
)
from ..gate_sets import QiskitGateSet
from ..models import PauliLindbladModel
from .learning_options import LearningOptions
from .noise_learner_job import NoiseLearnerJob, ProgramJob

_ANALYZERS = {
    "standard": AnalysisPipeline(ComputeObservables(), CurveFitObservables(), NNLSSolve())
}

_PATH_GENERATION_STAGES = {
    "even_depth": EvenDepthVanillaPaths,
}

CircuitInstruction: TypeAlias = _CircuitInstruction  # type: ignore


class ProgramExecutor(Protocol):
    """The executor role that :class:`NoiseLearner` submits programs to.

    Only :meth:`run` is required, so both :class:`~qiskit_ibm_runtime.Executor` and the Aer
    executor qualify without either declaring conformance.

    .. note::
        This protocol, and the ``executor`` argument of :class:`NoiseLearner` that consumes it,
        are provisional. They exist so that a learning experiment can be run against a locally
        simulated executor, and this setup is expected to eventually change.

        In particular, the backend and the executor are separate arguments because an executor
        is not an execution mode, and so cannot be folded into the single ``mode`` argument that
        the qiskit-ibm-runtime primitives take. Naming a backend that the supplied executor
        already knows about is therefore redundant on the runtime path. That redundancy is a
        consequence of the current shape rather than a deliberate convention, and the shape
        should not be relied upon as stable.
    """

    def run(self, program: QuantumProgram) -> ProgramJob:
        """Submit a program for execution.

        Args:
            program: The quantum program to submit.

        Returns:
            A job carrying the program's result.
        """
        ...


class NoiseLearner:
    """A noise learner.

    Args:
        backend: The backend supplying the compilation target: the gate set, coupling map and
            qubit count that generated circuits are built against. When ``executor`` is given,
            this need not be the device the programs actually run on.
        options: Learning options. If ``None``, default options are used.
        executor: Where generated programs are submitted, as described by
            :class:`ProgramExecutor`. If ``None`` (default), a
            :class:`~qiskit_ibm_runtime.Executor` in ``backend``'s execution mode is used, so
            that programs run on ``backend`` itself. Supplying an executor built with
            ``Executor(mode=...)`` is what makes session and batch execution reachable.
    """

    def __init__(
        self,
        backend: BackendV2,
        options: LearningOptions | None = None,
        executor: ProgramExecutor | None = None,
    ):
        self._backend = backend
        self._options = options or LearningOptions()
        self._analyzer = _ANALYZERS[self._options.analyzer]
        self._executor = executor

    @property
    def backend(self) -> BackendV2:
        """The backend."""
        return self._backend

    @property
    def options(self) -> LearningOptions:
        """The learning options."""
        return self._options

    def run(self, instructions: Sequence[CircuitInstruction]) -> NoiseLearnerJob:
        """Submit a job to learn the noise of the given instructions.

        Args:
            instructions: The instructions to learn the noise of. Each instruction should
                contain a :class:`~qiskit.circuit.BoxOp` operation.

        Returns:
            The submitted job. The result of the job is a :class:`~NoiseLearnerResult`.

        Raises:
            ValueError: If any instruction does not contain a ``BoxOp``.
        """
        for instr in instructions:
            if instr.operation.name != "box":
                raise ValueError(f"All instructions must be BoxOps, got '{instr.operation.name}'.")

        program, data_mapper = self._generate(instructions)
        executor = self._executor if self._executor is not None else Executor(mode=self._backend)
        job = executor.run(program)
        return NoiseLearnerJob(job, data_mapper, self._analyzer)

    def _generate(
        self, instructions: Sequence[CircuitInstruction]
    ) -> tuple[QuantumProgram, ExecutorDataMapper]:
        """Generate a quantum program from the given instructions.

        Args:
            instructions: The BoxOp instructions to learn.

        Returns:
            A tuple of the quantum program and data mapper.
        """
        # Build gate set from backend target
        qreg = QuantumRegister(self.backend.num_qubits, name="q")
        qubit_subset = set(qreg.index(qubit) for instr in instructions for qubit in instr.qubits)

        gate_set = QiskitGateSet(target=self._backend.target, qubit_subset=sorted(qubit_subset))
        for instr in instructions:
            inject_noise = get_annotation(instr.operation, InjectNoise)
            gate_set.add_box_as_gate(instr, name=None if inject_noise is None else inject_noise.ref)

        # Build fidelity model
        fidelity_model = PauliLindbladModel.k_local(gate_set, k=self._options.k_locality)

        # Build experiment via staged pipeline
        path_stage = _PATH_GENERATION_STAGES[self._options.path_generator]()
        pipeline = (
            path_stage
            + GenerateInstructionSequences()
            + MergeInstructionSequences()
            + IdentifyRelations()
            + CompleteSequences()
            + BindFragmentDepths(self._options.fragment_depths)
        )
        experiment = pipeline.run(
            Experiment(
                fidelity_model=fidelity_model,
                shots=self._options.shots_per_randomizations,
                randomizations=self._options.num_randomizations,
            )
        )

        # Generate circuits
        circuit_gen = ExecutorCircuitGenerator(gate_set)
        return circuit_gen.generate(experiment)
