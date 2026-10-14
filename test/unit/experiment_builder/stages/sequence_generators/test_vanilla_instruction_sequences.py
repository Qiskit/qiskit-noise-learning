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

from itertools import product

import pytest
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import CZGate
from qiskit.quantum_info import Clifford, QubitSparsePauliList
from qiskit.transpiler import CouplingMap

from qiskit_noise_learning.experiment_builder.experiment import Experiment
from qiskit_noise_learning.experiment_builder.stages import (
    EvenDepthVanillaPaths,
    VanillaInstructionSequences,
)
from qiskit_noise_learning.gate_sets import ModelGate, ModelGateSet
from qiskit_noise_learning.sequences import PartialPauliPermutation

IDENTITY_SET = frozenset({("Z", "Z"), ("X", "X"), ("Y", "Y")})


def _spam_gates(num_qubits):
    """A preparation and a measurement gate, both acting on every qubit."""
    identity = Clifford(QuantumCircuit(num_qubits))
    all_qubits = tuple(range(num_qubits))
    return (
        ModelGate("P", [(all_qubits, identity)], prep_idxs=all_qubits),
        ModelGate("M", [(all_qubits, identity)], meas_idxs=all_qubits),
    )


def _gate_set(num_qubits, coupling_map, gate_edges, support):
    """A gate set holding one CZ gate on ``support``, and SPAM gates on every qubit."""
    gate = ModelGate("CZ", [(edge, Clifford(CZGate())) for edge in gate_edges], qubit_idxs=support)
    prep, meas = _spam_gates(num_qubits)
    gate_set = ModelGateSet(num_qubits, coupling_map=coupling_map)
    for each in (gate, prep, meas):
        gate_set.add_gate(each)
    return gate_set, gate, prep, meas


def _permutations(num_qubits, coupling_map, gate_edges, support):
    """The permutations the stage generates for a single gate."""
    gate_set, gate, prep, meas = _gate_set(num_qubits, coupling_map, gate_edges, support)
    result = VanillaInstructionSequences(
        prep_gate=prep, meas_gate=meas, gates=[gate], coupling_map=coupling_map
    ).run(Experiment(fidelity_model=gate_set))

    return [
        instr
        for seq in result.instruction_sequences
        for instr in seq.start_fragment
        if isinstance(instr, PartialPauliPermutation)
    ]


class TestVanillaInstructionSequences:
    def test_requires_fidelity_model(self):
        stage = VanillaInstructionSequences()
        with pytest.raises(ValueError, match="requires 'fidelity_model'"):
            stage.run(Experiment())

    def test_generates_9_bases_on_ring(self):
        """On a ring, 9 vanilla sequences should be generated per gate."""
        num_qubits = 11
        coupling_map = CouplingMap([(idx, (idx + 1) % num_qubits) for idx in range(num_qubits)])
        gate_edges = [(idx, idx + 1) for idx in range(0, 10, 2)]
        gate_set, gate, prep, meas = _gate_set(
            num_qubits, coupling_map, gate_edges, range(num_qubits)
        )

        result = VanillaInstructionSequences(
            prep_gate=prep, meas_gate=meas, gates=[gate], coupling_map=coupling_map
        ).run(Experiment(fidelity_model=gate_set))

        assert len(result.instruction_sequences) == 9

    @pytest.mark.parametrize(
        "coupling_map, support",
        [
            (CouplingMap.from_line(5), (0, 1)),
            (CouplingMap([[0, 1], [1, 2], [2, 3], [3, 4], [4, 0]]), (2, 3)),
        ],
        ids=["bipartite_device", "non_bipartite_device"],
    )
    def test_bases_are_confined_to_the_gates_qubits(self, coupling_map, support):
        """Rotations reach only the gate's qubits, and still cover every pair on its edge."""
        permutations = _permutations(5, coupling_map, [support], support)

        assert len(permutations) == 9
        pairs = set()
        for permutation in permutations:
            sets = permutation.to_sets()
            elsewhere = [each for q, each in enumerate(sets) if q not in support]
            assert all(each == IDENTITY_SET for each in elsewhere), sets
            pairs.add(tuple(next(iter(sets[q]))[1] for q in support))

        assert pairs == set(product("XYZ", repeat=2))

    def test_bases_two_color_the_gates_support(self):
        """Each qubit maps Z to one of the pair, chosen by its distance parity in the support."""
        num_qubits = 5
        permutations = _permutations(
            num_qubits, CouplingMap.from_line(num_qubits), [(0, 1), (2, 3)], range(num_qubits)
        )

        # distance parity from qubit 0 along the line
        parity = {0: 0, 1: 1, 2: 0, 3: 1, 4: 0}
        observed = set()
        for permutation in permutations:
            sets = permutation.to_sets()
            targets = [next(iter(each))[1] for each in sets]
            # Z maps to a single Pauli on every qubit, including where that Pauli is Z itself
            assert sets == [frozenset({("Z", target)}) for target in targets]

            pauli_0 = {targets[q] for q, p in parity.items() if p == 0}
            pauli_1 = {targets[q] for q, p in parity.items() if p == 1}
            assert len(pauli_0) == 1 and len(pauli_1) == 1
            observed.add((pauli_0.pop(), pauli_1.pop()))

        assert observed == set(product("XYZ", repeat=2))

    @pytest.mark.parametrize(
        "num_qubits, coupling_map, gate_edges, support",
        [
            (
                11,
                CouplingMap([(idx, (idx + 1) % 11) for idx in range(11)]),
                [(idx, idx + 1) for idx in range(0, 10, 2)],
                range(11),
            ),
            (5, CouplingMap.from_line(5), [(0, 1)], (0, 1)),
            (5, CouplingMap([[0, 1], [1, 2], [2, 3], [3, 4], [4, 0]]), [(2, 3)], (2, 3)),
        ],
        ids=["full_width_ring", "narrow_line", "narrow_non_bipartite_device"],
    )
    def test_vanilla_paths_traversed_by_vanilla_sequences(
        self, num_qubits, coupling_map, gate_edges, support
    ):
        """Vanilla paths from EvenDepthVanillaPaths are all traversable by vanilla sequences."""
        gate_set, gate, prep, meas = _gate_set(num_qubits, coupling_map, gate_edges, support)

        paulis = ["".join(pauli) for pauli in product("IXZY", repeat=2)]
        input_paulis = QubitSparsePauliList.from_sparse_list(
            [(pauli, edge) for edge in gate_edges for pauli in paulis], num_qubits=num_qubits
        )

        exp = Experiment(fidelity_model=gate_set)
        exp = EvenDepthVanillaPaths(
            prep_gate=prep, meas_gate=meas, gates=[gate], input_paulis={"CZ": input_paulis}
        ).run(exp)
        exp = VanillaInstructionSequences(
            prep_gate=prep, meas_gate=meas, gates=[gate], coupling_map=coupling_map
        ).run(exp)

        for path in exp.paths:
            assert any(
                path.is_traversed_by(seq) for seq in exp.instruction_sequences
            ), f"Path not traversed by any sequence: {path}"
