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
    AddInstructionSequences,
    EvenDepthVanillaPaths,
    GenerateInstructionSequences,
    VanillaInstructionSequences,
)
from qiskit_noise_learning.gate_sets import ModelGate, ModelGateSet


class TestAddInstructionSequences:
    def test_adds_sequences(self, unbound_path_ix):
        seq = unbound_path_ix.to_instruction_sequence()
        exp = Experiment()
        result = AddInstructionSequences([seq]).run(exp)

        assert result.instruction_sequences == [seq]
        assert result.randomization_multipliers == [1]

    def test_appends_to_existing(self, unbound_path_ix, unbound_path_xi):
        seq_ix = unbound_path_ix.to_instruction_sequence()
        seq_xi = unbound_path_xi.to_instruction_sequence()
        exp = Experiment(instruction_sequences=[seq_ix], randomization_multipliers=[1])
        result = AddInstructionSequences([seq_xi], randomization_multipliers=[3]).run(exp)

        assert result.instruction_sequences == [seq_ix, seq_xi]
        assert result.randomization_multipliers == [1, 3]


class TestGenerateInstructionSequences:
    def test_requires_paths(self):
        stage = GenerateInstructionSequences()
        with pytest.raises(ValueError, match="requires 'paths'"):
            stage.run(Experiment())

    def test_generates_sequences_from_paths(self, gate_set_cz, unbound_path_ix, unbound_path_xi):
        exp = Experiment(
            fidelity_model=gate_set_cz,
            paths=[unbound_path_ix, unbound_path_xi],
        )
        result = GenerateInstructionSequences().run(exp)

        assert len(result.instruction_sequences) == 2
        assert result.relations == {(0, 0), (1, 1)}
        assert result.randomization_multipliers == [1, 1]

    def test_skips_already_related_paths(self, gate_set_cz, unbound_path_ix, unbound_path_xi):
        seq_ix = unbound_path_ix.to_instruction_sequence()
        exp = Experiment(
            fidelity_model=gate_set_cz,
            paths=[unbound_path_ix, unbound_path_xi],
            instruction_sequences=[seq_ix],
            relations={(0, 0)},
            randomization_multipliers=[1],
        )
        result = GenerateInstructionSequences().run(exp)

        assert len(result.instruction_sequences) == 2
        assert result.relations == {(0, 0), (1, 1)}

    def test_custom_multipliers(self, gate_set_cz, unbound_path_ix):
        bound_path = unbound_path_ix.bind_at(2)
        exp = Experiment(
            fidelity_model=gate_set_cz,
            paths=[unbound_path_ix, bound_path],
        )
        result = GenerateInstructionSequences(bound_multiplier=5, unbound_multiplier=3).run(exp)

        assert result.randomization_multipliers == [3, 5]


class TestVanillaInstructionSequences:
    def test_requires_fidelity_model(self):
        stage = VanillaInstructionSequences()
        with pytest.raises(ValueError, match="requires 'fidelity_model'"):
            stage.run(Experiment())

    def test_generates_9_bases_on_ring(self):
        """On a triangle-free ring, 9 vanilla sequences should be generated per gate."""
        num_qubits = 11
        edges = [(idx, (idx + 1) % num_qubits) for idx in range(num_qubits)]
        coupling_map = CouplingMap(edges)

        gate_idxs = [(idx, idx + 1) for idx in range(0, 10, 2)]
        gate = ModelGate("CZ", [(idxs, Clifford(CZGate())) for idxs in gate_idxs])
        prep = ModelGate(
            "P",
            [(tuple(range(num_qubits)), Clifford(QuantumCircuit(num_qubits)))],
            prep_idxs=range(num_qubits),
        )
        meas = ModelGate(
            "M",
            [(tuple(range(num_qubits)), Clifford(QuantumCircuit(num_qubits)))],
            meas_idxs=range(num_qubits),
        )

        gate_set = ModelGateSet(num_qubits)
        gate_set.add_gate(gate)
        gate_set.add_gate(prep)
        gate_set.add_gate(meas)

        exp = Experiment(fidelity_model=gate_set)
        stage = VanillaInstructionSequences(
            prep_gate=prep,
            meas_gate=meas,
            gates=[gate],
            coupling_map=coupling_map,
        )
        result = stage.run(exp)

        assert len(result.instruction_sequences) == 9

    def test_vanilla_paths_traversed_by_vanilla_sequences(self):
        """Vanilla paths from EvenDepthVanillaPaths are all traversable by vanilla sequences."""
        num_qubits = 11
        edges = [(idx, (idx + 1) % num_qubits) for idx in range(num_qubits)]
        coupling_map = CouplingMap(edges)

        gate_idxs = [(idx, idx + 1) for idx in range(0, 10, 2)]
        gate = ModelGate("CZ", [(idxs, Clifford(CZGate())) for idxs in gate_idxs])
        prep = ModelGate(
            "P",
            [(tuple(range(num_qubits)), Clifford(QuantumCircuit(num_qubits)))],
            prep_idxs=range(num_qubits),
        )
        meas = ModelGate(
            "M",
            [(tuple(range(num_qubits)), Clifford(QuantumCircuit(num_qubits)))],
            meas_idxs=range(num_qubits),
        )

        gate_set = ModelGateSet(num_qubits)
        gate_set.add_gate(gate)
        gate_set.add_gate(prep)
        gate_set.add_gate(meas)

        paulis = ["".join(p for p in comb) for comb in product("IXZY", repeat=2)]
        input_paulis = QubitSparsePauliList.from_sparse_list(
            [(pauli, idxs) for idxs in gate_idxs for pauli in paulis], num_qubits=num_qubits
        )

        exp = Experiment(fidelity_model=gate_set)

        paths_stage = EvenDepthVanillaPaths(
            prep_gate=prep,
            meas_gate=meas,
            gates=[gate],
            input_paulis={"CZ": input_paulis},
        )
        exp = paths_stage.run(exp)

        seqs_stage = VanillaInstructionSequences(
            prep_gate=prep,
            meas_gate=meas,
            gates=[gate],
            coupling_map=coupling_map,
        )
        exp = seqs_stage.run(exp)

        for path in exp.paths:
            assert any(
                path.is_traversed_by(seq) for seq in exp.instruction_sequences
            ), f"Path not traversed by any sequence: {path}"
