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

from collections.abc import Iterable
from itertools import product

from qiskit.transpiler import CouplingMap
from rustworkx import PyGraph

from qiskit_noise_learning.gate_sets import ModelGate
from qiskit_noise_learning.sequences import ApplyGate, InstructionSequence, PartialPauliPermutation

from ...experiment import Experiment
from ..utils import default_meas_gate, default_prep_gate, default_unitary_gates
from .add_instruction_sequences import AddInstructionSequences

_IDENTITY_PERMUTATION_SET = frozenset({("Z", "Z"), ("X", "X"), ("Y", "Y")})
"""The single-qubit partial Pauli permutation set of the identity."""


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
        gates = self._gates or default_unitary_gates(gate_set)
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
    """Generate the 9 vanilla instruction sequences for a single gate.

    Args:
        prep_gate: The preparation gate.
        meas_gate: The measurement gate.
        gate: The gate to generate sequences for.
        coupling_map: The coupling map.

    Returns:
        One instruction sequence per Pauli pair.
    """
    num_qubits = len(coupling_map.graph.nodes())
    qubit_color = _two_color(coupling_map.graph, gate.qubit_idxs)
    repeatable_fragment = [ApplyGate(gate.name), ApplyGate(gate.name)]

    ret = []
    for pauli_0, pauli_1 in product(["X", "Y", "Z"], repeat=2):
        sets = [_IDENTITY_PERMUTATION_SET] * num_qubits
        for qubit, color in qubit_color.items():
            sets[qubit] = frozenset({("Z", pauli_0 if color == 0 else pauli_1)})

        permutation = PartialPauliPermutation.from_sets(sets)
        ret.append(
            InstructionSequence(
                [ApplyGate(prep_gate.name), permutation],
                repeatable_fragment,
                [permutation.inverse, ApplyGate(meas_gate.name)],
            )
        )
    return ret


def _two_color(graph: PyGraph, qubits: Iterable[int]) -> dict[int, int]:
    """Two-color the subgraph of a connectivity graph induced on some qubits.

    Each qubit's color is the parity of its distance, within the induced subgraph, to the
    lowest-indexed qubit of its connected component. The color of a qubit is therefore fixed by
    ``graph`` and ``qubits`` alone, independently of the order either is given in.

    Neighbouring qubits are given different colors exactly when the induced subgraph is bipartite.
    Coloring the induced subgraph rather than the whole graph succeeds in some cases where coloring
    the whole graph does not, because a subgraph of a non-bipartite graph may still be bipartite.

    Args:
        graph: The connectivity graph.
        qubits: The qubits whose induced subgraph is to be colored.

    Returns:
        The color, ``0`` or ``1``, of each qubit of ``qubits``.
    """
    qubits = set(qubits)
    edges = [(q0, q1) for q0, q1 in graph.edge_list() if q0 in qubits and q1 in qubits]

    qubit_color = {}
    visited = set()

    for start_q in sorted(qubits):
        if start_q in visited:
            continue

        queue = [(start_q, 0)]
        visited.add(start_q)
        qubit_color[start_q] = 0

        while queue:
            q, color = queue.pop(0)

            for q0, q1 in edges:
                neighbor = None
                if q0 == q and q1 not in visited:
                    neighbor = q1
                elif q1 == q and q0 not in visited:
                    neighbor = q0

                if neighbor is not None:
                    visited.add(neighbor)
                    qubit_color[neighbor] = 1 - color
                    queue.append((neighbor, 1 - color))

    return qubit_color
