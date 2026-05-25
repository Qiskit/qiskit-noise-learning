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
from itertools import chain
from typing import TypeAlias

from qiskit.circuit import CircuitInstruction as _CircuitInstruction
from qiskit.circuit import QuantumRegister
from qiskit.providers import BackendV2
from qiskit_ibm_runtime import Executor
from qiskit_ibm_runtime.quantum_program import QuantumProgram
from qiskit_ibm_runtime.quantum_program.quantum_program import SamplexItem
from samplomatic import InjectNoise
from samplomatic.utils import get_annotation

from ..analysis import (
    AnalysisPipeline,
    ComputeObservables,
    CurveFitObservables,
    NNLSSolve,
)
from ..circuit_generator import ExecutorCircuitGenerator
from ..experiment_builder import (
    ExperimentBuilder,
    even_depth_vanilla_path_generator,
)
from ..gate_sets import QiskitGateSet
from ..models import PauliLindbladModel
from .learning_options import LearningOptions
from .noise_learner_job import ExperimentSchema, NoiseLearnerJob

_ANALYZERS = {
    "standard": AnalysisPipeline(ComputeObservables(), CurveFitObservables(), NNLSSolve())
}

_PATH_GENERATORS = {
    "even_depth": even_depth_vanilla_path_generator,
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
        self._path_generator = _PATH_GENERATORS[self._options.path_generator]

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

        samplex_items, experiment_schema = self._generate(instructions)
        job = self._execute(samplex_items)
        job.experiment_schema = experiment_schema
        return NoiseLearnerJob(job, experiment_schema, self._analyzer)

    def _generate(
        self, instructions: Sequence[CircuitInstruction]
    ) -> tuple[list[SamplexItem], ExperimentSchema]:
        """Generate samplex items from the given instructions.

        Args:
            instructions: The BoxOp instructions to learn.

        Returns:
            A tuple of samplex items and an experiment schema.
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

        # Get the model gate set for accessing gate properties
        model_gate_set = gate_set.model_gate_set

        # Identify SPAM gates
        prep_gate = None
        meas_gate = None
        for name, gate in model_gate_set.items():
            if gate.prep_idxs:
                prep_gate = gate
            elif gate.meas_idxs:
                meas_gate = gate

        # Generate paths for each non-SPAM gate
        path_iterators = []
        for name, gate in model_gate_set.items():
            if gate.prep_idxs or gate.meas_idxs:
                continue
            input_paulis = fidelity_model.generators[name]
            path_iterators.append(self._path_generator(prep_gate, meas_gate, gate, input_paulis))

        # Build experiments
        builder = ExperimentBuilder(fidelity_model)
        builder.add_paths(chain.from_iterable(path_iterators))
        builder.merge_instruction_sequences()
        builder.complete()

        paths = [p.bind_at(d) for p in builder.paths for d in self._options.depths]
        paths.extend(builder.paths)

        # Generate instruction sequences
        sequences = builder.generate_instruction_sequences(depths=self._options.depths)

        # Generate circuits
        circuit_gen = ExecutorCircuitGenerator(
            gate_set, num_randomizations=self._options.num_randomizations
        )
        samplex_items, executor_data_mapper = circuit_gen.generate(sequences)

        return samplex_items, ExperimentSchema(executor_data_mapper, paths, fidelity_model)

    def _execute(self, samplex_items: list[SamplexItem]):
        """Submit a job to execute samplex items on the backend.

        Args:
            samplex_items: The samplex items to execute.

        Returns:
            The job.
        """
        program = QuantumProgram(shots=self._options.shots_per_randomizations, items=samplex_items)
        executor = Executor(mode=self._backend)
        return executor.run(program)
