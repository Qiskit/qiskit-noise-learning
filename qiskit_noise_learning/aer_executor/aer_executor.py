# This code is a Qiskit project.
#
# (C) Copyright IBM 2025, 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""AerExecutor and AerRuntimeJob: local simulation executor for QuantumProgram objects."""

import uuid

from qiskit.quantum_info import PauliLindbladMap
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QuantumProgram
from qiskit_ibm_runtime.quantum_program.quantum_program_result import QuantumProgramResult

from .run_quantum_program import run_quantum_program


class AerRuntimeJob:
    def __init__(
        self,
        qasm_simulator: AerSimulator,
        program: QuantumProgram,
        noise_dict: dict[str, PauliLindbladMap] | None = None,
        angle_decimals: int = 5,
    ):
        self._qasm_simulator = qasm_simulator
        self._program = program
        self._noise_dict = noise_dict
        self._angle_decimals = angle_decimals
        self._job_id: str = str(uuid.uuid4())
        self.tags: list[str] = []

        self._result = run_quantum_program(
            qasm_simulator=self._qasm_simulator,
            program=self._program,
            noise_dict=self._noise_dict,
            angle_decimals=self._angle_decimals,
        )

    def job_id(self) -> str:
        return self._job_id

    def result(
        self,
        timeout: None = None,
        decoder: None = None,
    ) -> QuantumProgramResult:
        return self._result


class AerExecutor:
    """Local Aer-based executor mimicking the IBM Runtime executor interface.

    Runs a :class:`~qiskit_ibm_runtime.QuantumProgram` eagerly on construction of the
    returned job — the result is available immediately when :meth:`AerRuntimeJob.result`
    is called.
    """

    def __init__(
        self,
        qasm_simulator: AerSimulator,
        noise_dict: dict[str, PauliLindbladMap] | None = None,
        angle_decimals: int = 5,
    ) -> None:
        self._qasm_simulator = qasm_simulator
        self._noise_dict = noise_dict
        self._angle_decimals = angle_decimals

    def run(self, program: QuantumProgram) -> AerRuntimeJob:
        return AerRuntimeJob(
            qasm_simulator=self._qasm_simulator,
            program=program,
            noise_dict=self._noise_dict,
            angle_decimals=self._angle_decimals,
        )
