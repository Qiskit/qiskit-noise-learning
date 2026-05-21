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

"""Converters between runtime domain objects and pydantic serialization schemas."""

from __future__ import annotations

import numpy as np
from qiskit.quantum_info import Clifford, QubitSparsePauli, QubitSparsePauliList
from qiskit.transpiler import CouplingMap

from ..gate_sets import ModelGate, ModelGateSet
from ..models import PauliLindbladModel
from ..sequences import (
    ApplyGate,
    FidelityIndex,
    InstructionSequence,
    PartialPauliPermutation,
    Path,
)
from .schemas import (
    ApplyGateSchema,
    CliffordComponentSchema,
    FidelityIndexSchema,
    InstructionSequenceSchema,
    ModelGateSchema,
    ModelGateSetSchema,
    PartialPauliPermutationSchema,
    PathSchema,
    PauliLindbladModelSchema,
    QubitSparsePauliListSchema,
    QubitSparsePauliSchema,
)


def serialize_qubit_sparse_pauli(pauli: QubitSparsePauli) -> QubitSparsePauliSchema:
    return QubitSparsePauliSchema(
        num_qubits=pauli.num_qubits,
        paulis=[int(p) for p in pauli.paulis],
        indices=[int(i) for i in pauli.indices],
    )


def deserialize_qubit_sparse_pauli(schema: QubitSparsePauliSchema) -> QubitSparsePauli:
    return QubitSparsePauli.from_raw_parts(
        num_qubits=schema.num_qubits,
        paulis=schema.paulis,
        indices=schema.indices,
    )


def serialize_qubit_sparse_pauli_list(
    pauli_list: QubitSparsePauliList,
) -> QubitSparsePauliListSchema:
    return QubitSparsePauliListSchema(
        num_qubits=pauli_list.num_qubits,
        paulis=[serialize_qubit_sparse_pauli(p) for p in pauli_list],
    )


def deserialize_qubit_sparse_pauli_list(
    schema: QubitSparsePauliListSchema,
) -> QubitSparsePauliList:
    paulis = [deserialize_qubit_sparse_pauli(p) for p in schema.paulis]
    return QubitSparsePauliList.from_qubit_sparse_paulis(paulis)


def serialize_gate_set(gate_set: ModelGateSet) -> ModelGateSetSchema:
    coupling_map = gate_set.coupling_map
    edges = [list(edge) for edge in coupling_map.get_edges()]
    gates = []
    for name, gate in gate_set.items():
        cliffords = []
        for qubit_idxs, clifford in gate.cliffords:
            cliffords.append(
                CliffordComponentSchema(
                    qubit_idxs=list(qubit_idxs),
                    tableau=clifford.symplectic_matrix.tolist(),
                )
            )
        gates.append(
            ModelGateSchema(
                name=name,
                qubit_idxs=list(gate.qubit_idxs),
                meas_idxs=sorted(gate.meas_idxs),
                prep_idxs=sorted(gate.prep_idxs),
                cliffords=cliffords,
            )
        )
    return ModelGateSetSchema(
        num_qubits=gate_set.num_qubits,
        qubit_subset=sorted(gate_set.qubit_subset),
        coupling_map_edges=edges,
        gates=gates,
    )


def deserialize_gate_set(schema: ModelGateSetSchema) -> ModelGateSet:
    coupling_map = CouplingMap(schema.coupling_map_edges)
    gate_set = ModelGateSet(
        num_qubits=schema.num_qubits,
        qubit_subset=schema.qubit_subset,
        coupling_map=coupling_map,
    )
    for gate_schema in schema.gates:
        cliffords = []
        for comp in gate_schema.cliffords:
            tableau = np.array(comp.tableau, dtype=bool)
            cliffords.append((tuple(comp.qubit_idxs), Clifford(tableau)))
        gate = ModelGate(
            name=gate_schema.name,
            cliffords=cliffords,
            qubit_idxs=gate_schema.qubit_idxs,
            meas_idxs=gate_schema.meas_idxs,
            prep_idxs=gate_schema.prep_idxs,
        )
        gate_set.add_gate(gate)
    return gate_set


def serialize_instruction_sequence(seq: InstructionSequence) -> InstructionSequenceSchema:
    return InstructionSequenceSchema(
        depth=seq.depth,
        start_fragment=[serialize_instruction(i) for i in seq.start_fragment],
        repeatable_fragment=[serialize_instruction(i) for i in seq.repeatable_fragment],
        end_fragment=[serialize_instruction(i) for i in seq.end_fragment],
    )


def serialize_instruction(instr) -> ApplyGateSchema | PartialPauliPermutationSchema:
    if isinstance(instr, ApplyGate):
        return ApplyGateSchema(gate_name=instr.gate.name)
    elif isinstance(instr, PartialPauliPermutation):
        return PartialPauliPermutationSchema(
            partial_permutation_indices=[int(x) for x in instr.partial_permutation_indices],
        )
    raise TypeError(f"Unknown instruction type: {type(instr)}")


def deserialize_instruction_sequence(
    schema: InstructionSequenceSchema, gate_set: ModelGateSet
) -> InstructionSequence:
    return InstructionSequence(
        start_fragment=[deserialize_instruction(i, gate_set) for i in schema.start_fragment],
        repeatable_fragment=[
            deserialize_instruction(i, gate_set) for i in schema.repeatable_fragment
        ],
        end_fragment=[deserialize_instruction(i, gate_set) for i in schema.end_fragment],
        depth=schema.depth,
    )


def deserialize_instruction(
    schema: ApplyGateSchema | PartialPauliPermutationSchema, gate_set: ModelGateSet
):
    if isinstance(schema, ApplyGateSchema):
        return ApplyGate(gate_set[schema.gate_name].model_gate)
    elif isinstance(schema, PartialPauliPermutationSchema):
        return PartialPauliPermutation(np.array(schema.partial_permutation_indices, dtype=np.int8))
    raise TypeError(f"Unknown instruction schema type: {type(schema)}")


def serialize_path(path: Path) -> PathSchema:
    return PathSchema(
        depth=path.depth,
        start_fragment=[serialize_fidelity_index(fi) for fi in path.start_fragment],
        repeatable_fragment=[serialize_fidelity_index(fi) for fi in path.repeatable_fragment],
        end_fragment=[serialize_fidelity_index(fi) for fi in path.end_fragment],
    )


def serialize_fidelity_index(fi: FidelityIndex) -> FidelityIndexSchema:
    return FidelityIndexSchema(
        gate_name=fi.gate.name,
        pauli=serialize_qubit_sparse_pauli(fi.pauli),
        in_bit_indices=sorted(fi.in_bit_indices),
        out_bit_indices=sorted(fi.out_bit_indices),
    )


def deserialize_path(schema: PathSchema, gate_set: ModelGateSet) -> Path:
    return Path(
        start_fragment=[deserialize_fidelity_index(fi, gate_set) for fi in schema.start_fragment],
        repeatable_fragment=[
            deserialize_fidelity_index(fi, gate_set) for fi in schema.repeatable_fragment
        ],
        end_fragment=[deserialize_fidelity_index(fi, gate_set) for fi in schema.end_fragment],
        depth=schema.depth,
    )


def deserialize_fidelity_index(
    schema: FidelityIndexSchema, gate_set: ModelGateSet
) -> FidelityIndex:
    gate = gate_set[schema.gate_name].model_gate
    pauli = deserialize_qubit_sparse_pauli(schema.pauli)
    return FidelityIndex(
        gate=gate,
        pauli=pauli,
        in_bit_indices=frozenset(schema.in_bit_indices),
        out_bit_indices=frozenset(schema.out_bit_indices),
    )


def serialize_pauli_lindblad_model(model: PauliLindbladModel) -> PauliLindbladModelSchema:
    generators = {}
    for gate_name, pauli_list in model.generators.items():
        generators[gate_name] = serialize_qubit_sparse_pauli_list(pauli_list)
    return PauliLindbladModelSchema(
        generators=generators,
        noise_site=dict(model.noise_site),
    )


def deserialize_pauli_lindblad_model(
    schema: PauliLindbladModelSchema, gate_set: ModelGateSet
) -> PauliLindbladModel:
    generators = {}
    for gate_name, pauli_list_schema in schema.generators.items():
        generators[gate_name] = deserialize_qubit_sparse_pauli_list(pauli_list_schema)
    return PauliLindbladModel(
        gate_set=gate_set,
        generators=generators,
        noise_site=schema.noise_site,
    )
