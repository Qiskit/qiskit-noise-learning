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
from typing import TypeAlias

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
    BindSequenceDepths,
    CompleteSequences,
    EvenDepthVanillaPaths,
    Experiment,
    GenerateInstructionSequences,
    IdentifyRelations,
    MergeInstructionSequences,
)
from ..gate_sets import QiskitGateSet
from ..models._legacy import PauliLindbladModel
from .learning_options import LearningOptions
from .noise_learner_job import NoiseLearnerJob

_ANALYZERS = {
    "standard": AnalysisPipeline(ComputeObservables(), CurveFitObservables(), NNLSSolve())
}

_PATH_GENERATION_STAGES = {
    "even_depth": EvenDepthVanillaPaths,
}

CircuitInstruction: TypeAlias = _CircuitInstruction  # type: ignore


class NoiseLearner:
    """A noise learner.

    Args:
        backend: The backend to learn noise from.
        options: Learning options. If ``None``, default options are used.
    """

    def __init__(
        self,
        backend: BackendV2,
        options: LearningOptions | None = None,
    ):
        self._backend = backend
        self._options = options or LearningOptions()
        self._analyzer = _ANALYZERS[self._options.analyzer]

    @property
    def backend(self) -> BackendV2:
        """The backend."""
        return self._backend

    @property
    def options(self) -> LearningOptions:
        """The learning options."""
        return self._options

    def run(self, instructions: Sequence[CircuitInstruction]):
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
        executor = Executor(mode=self._backend)
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
            + BindSequenceDepths(self._options.depths)
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
