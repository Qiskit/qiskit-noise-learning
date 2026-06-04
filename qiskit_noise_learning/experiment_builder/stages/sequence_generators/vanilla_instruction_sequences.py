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

"""VanillaInstructionSequences stage."""

from itertools import product

from qiskit.quantum_info import QubitSparsePauli
from qiskit.transpiler import CouplingMap
from rustworkx import PyGraph

from qiskit_noise_learning.gate_sets import ModelGate
from qiskit_noise_learning.sequences import ApplyGate, InstructionSequence, PartialPauliPermutation

from ...experiment import Experiment
from ..utils import default_gates, default_meas_gate, default_prep_gate
from .add_instruction_sequences import AddInstructionSequences


class VanillaInstructionSequences(AddInstructionSequences):
    """Generate vanilla instruction sequences for each target gate.

    For each gate, generates 9 instruction sequences sufficient to measure any single- and
    two-qubit Pauli fidelity on a triangle-free coupling map.

    Args:
        prep_gate: The preparation gate. If ``None``, defaults to the gate named ``"P"``.
        meas_gate: The measurement gate. If ``None``, defaults to the gate named ``"M"``.
        gates: Gates to generate sequences for. If ``None``, defaults to all non-SPAM gates.
        coupling_map: The coupling map. If ``None``, defaults to the gate set's coupling map.
        randomization_multiplier: Randomization multiplier applied to all generated sequences.
    """

    @property
    def required_fields(self) -> tuple[str, ...]:
        if (
            self._gates is None
            or self._prep_gate is None
            or self._meas_gate is None
            or self._coupling_map is None
        ):
            return ("fidelity_model",)
        return ()

    def __init__(
        self,
        *,
        prep_gate: ModelGate | None = None,
        meas_gate: ModelGate | None = None,
        gates: list[ModelGate] | None = None,
        coupling_map: CouplingMap | None = None,
        randomization_multiplier: int = 1,
    ):
        self._prep_gate = prep_gate
        self._meas_gate = meas_gate
        self._gates = gates
        self._coupling_map = coupling_map
        self._randomization_multiplier = randomization_multiplier

    def _generate_sequences(
        self, experiment: Experiment
    ) -> tuple[list[InstructionSequence], list[int]]:
        gate_set = experiment.gate_set
        prep_gate = self._prep_gate or default_prep_gate(gate_set)
        meas_gate = self._meas_gate or default_meas_gate(gate_set)
        gates = self._gates or default_gates(gate_set)
        coupling_map = self._coupling_map or gate_set.coupling_map

        sequences = []
        for gate in gates:
            sequences.extend(
                _generate_vanilla_instruction_sequences(prep_gate, meas_gate, gate, coupling_map)
            )

        multipliers = [self._randomization_multiplier] * len(sequences)
        return sequences, multipliers


def _generate_vanilla_instruction_sequences(
    prep_gate: ModelGate, meas_gate: ModelGate, gate: ModelGate, coupling_map: CouplingMap
) -> list[InstructionSequence]:
    ret = []
    all_zs = QubitSparsePauli.from_label("Z" * len(coupling_map.graph.nodes()))
    repeatable_fragment = [ApplyGate(gate), ApplyGate(gate)]
    for basis in _generate_bases(coupling_map.graph):
        basis = QubitSparsePauli.from_label(basis)
        permutation = PartialPauliPermutation.from_qubit_sparse_paulis(all_zs, basis)
        ret.append(
            InstructionSequence(
                [ApplyGate(prep_gate), permutation],
                repeatable_fragment,
                [permutation.inverse, ApplyGate(meas_gate)],
            )
        )
    return ret


def _generate_bases(graph: PyGraph) -> list[str]:
    """Generate the 9 basis strings for measuring single- and two-qubit Pauli fidelities.

    For triangle-free topologies, 9 bases are always sufficient to measure all 9 non-identity
    Pauli pair combinations on every edge.
    """
    qubit_color = {}
    visited = set()

    for start_q in graph.node_indices():
        if start_q in visited:
            continue

        queue = [(start_q, 0)]
        visited.add(start_q)
        qubit_color[start_q] = 0

        while queue:
            q, color = queue.pop(0)

            for q1, q2 in graph.edge_list():
                neighbor = None
                if q1 == q and q2 not in visited:
                    neighbor = q2
                elif q2 == q and q1 not in visited:
                    neighbor = q1

                if neighbor:
                    visited.add(neighbor)
                    qubit_color[neighbor] = 1 - color
                    queue.append((neighbor, 1 - color))

    generated_bases = []
    all_pauli_pairs = list(product(["X", "Y", "Z"], repeat=2))

    for p0, p1 in all_pauli_pairs:
        basis_list = []
        for q in graph.node_indices():
            color = qubit_color[q]
            basis_list.append(p0 if color == 0 else p1)
        basis = "".join(basis_list[::-1])
        generated_bases.append(basis)

    return generated_bases
