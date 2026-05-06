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

"""Transpiler pass that inserts Pauli-Lindblad noise after labeled barriers."""

import re

from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGCircuit, DAGOpNode
from qiskit.quantum_info import PauliLindbladMap, QubitSparsePauliList
from qiskit.transpiler import TransformationPass
from qiskit_aer.noise import PauliLindbladError


class InsertNoisePass(TransformationPass):
    def __init__(
        self,
        noise_dict: dict[str, PauliLindbladMap] | None,
        noise_after: bool = True,
        noise_scale: float = 1.0,
    ):
        self._noise_dict = noise_dict
        self._noise_after = noise_after
        self._noise_scale = noise_scale

        self._pattern = re.compile(r"^(?P<pos>[A-Za-z])(?P<idx>\d+)@tag=(?P<tag>.+)$")

        super().__init__()

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        for op_node in reversed(list(dag.topological_op_nodes())):
            if op_node.name != "barrier":
                continue

            _new_subdag = self._new_subdag(op_node=op_node)
            if _new_subdag:
                dag.substitute_node_with_dag(
                    node=op_node,
                    input_dag=_new_subdag,
                )
        return dag

    def _match_key(self, name: str) -> str | None:
        m = self._pattern.match(name)
        if not m:
            return None
        pos = m.group("pos")
        tag = m.group("tag")

        if self._noise_after:
            if pos != "R":
                return None
        else:
            if pos != "M":
                return None

        return tag

    def _new_subdag(self, op_node: DAGOpNode) -> DAGCircuit | None:
        # op._label is a private attribute; no public API exposes barrier labels
        if op_node.op._label is None:  # noqa: SLF001
            return None
        noise_key = self._match_key(op_node.op._label)  # noqa: SLF001
        if noise_key is None:
            return None

        try:
            if self._noise_dict is None:
                return None
            pauli_lindblad_map = self._noise_dict[noise_key]
        except KeyError:
            raise ValueError(
                f"Noise not found for {noise_key}, expecting one of: {self._noise_dict.keys()}"
            )

        qspl = QubitSparsePauliList.from_sparse_list(
            [(p, supp) for p, supp, _ in pauli_lindblad_map.to_sparse_list()],
            num_qubits=pauli_lindblad_map.num_qubits,
        )
        pauli_lindblad_error = PauliLindbladError(
            generators=qspl.to_pauli_list(),
            rates=self._noise_scale * pauli_lindblad_map.rates,
        )

        qc = QuantumCircuit(op_node.num_qubits)
        qc.append(op_node.op, qc.qubits)
        qc.append(pauli_lindblad_error, qc.qubits)
        return circuit_to_dag(qc)
