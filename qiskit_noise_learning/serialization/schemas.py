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

"""Reusable pydantic serialization schemas for noise learning types."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Literal, Self

import numpy as np
from pydantic import BaseModel, Field
from qiskit.quantum_info import Clifford, QubitSparsePauli, QubitSparsePauliList
from qiskit.transpiler import CouplingMap

from ..gate_sets import ModelGate, ModelGateSet
from ..models import PauliLindbladModel
from ..sequences import (
    ApplyGate,
    FidelityIndex,
    InstructionPattern,
    InstructionSequence,
    PartialPauliPermutation,
    Path,
    PathPattern,
)

if TYPE_CHECKING:
    from ..sequences import Instruction


class QubitSparsePauliSchema(BaseModel):
    """Serialization schema for a ``QubitSparsePauli``.

    Stores as an integer lists of Paulis and their corresponding qubit indices.
    """

    num_qubits: int
    paulis: list[int]
    indices: list[int]

    @classmethod
    def serialize(cls, pauli: QubitSparsePauli) -> Self:
        """Serialize a :class:`~qiskit.quantum_info.QubitSparsePauli`."""
        return cls(
            num_qubits=pauli.num_qubits,
            paulis=[int(p) for p in pauli.paulis],
            indices=[int(i) for i in pauli.indices],
        )

    def deserialize(self) -> QubitSparsePauli:
        """Deserialize to a :class:`~qiskit.quantum_info.QubitSparsePauli`."""
        return QubitSparsePauli.from_raw_parts(
            num_qubits=self.num_qubits,
            paulis=self.paulis,
            indices=self.indices,
        )


class QubitSparsePauliListSchema(BaseModel):
    """Serialization schema for a ``QubitSparsePauliList``."""

    num_qubits: int
    paulis: list[QubitSparsePauliSchema]

    @classmethod
    def serialize(cls, pauli_list: QubitSparsePauliList) -> Self:
        """Serialize a :class:`~qiskit.quantum_info.QubitSparsePauliList`."""
        return cls(
            num_qubits=pauli_list.num_qubits,
            paulis=[QubitSparsePauliSchema.serialize(p) for p in pauli_list],
        )

    def deserialize(self) -> QubitSparsePauliList:
        """Deserialize to a :class:`~qiskit.quantum_info.QubitSparsePauliList`."""
        paulis = [p.deserialize() for p in self.paulis]
        return QubitSparsePauliList.from_qubit_sparse_paulis(paulis)


class CliffordComponentSchema(BaseModel):
    r"""Serialization schema for a single ``Clifford`` in a :class:`ModelGate`\'s clifford list.

    Stores each Clifford in tableau form.
    """

    qubit_idxs: list[int]
    tableau: list[list[bool]]


class ModelGateSchema(BaseModel):
    """Serialization schema for a :class:`ModelGate`."""

    name: str
    qubit_idxs: list[int]
    meas_idxs: list[int]
    prep_idxs: list[int]
    cliffords: list[CliffordComponentSchema]


class ModelGateSetSchema(BaseModel):
    """Serialization schema for a :class:`ModelGateSet`."""

    num_qubits: int
    qubit_subset: list[int]
    coupling_map_edges: list[list[int]]
    gates: list[ModelGateSchema]

    @classmethod
    def serialize(cls, gate_set: ModelGateSet) -> Self:
        """Serialize a :class:`~.ModelGateSet`."""
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
        return cls(
            num_qubits=gate_set.num_qubits,
            qubit_subset=sorted(gate_set.qubit_subset),
            coupling_map_edges=edges,
            gates=gates,
        )

    def deserialize(self) -> ModelGateSet:
        """Deserialize to a :class:`~.ModelGateSet`."""
        coupling_map = CouplingMap(self.coupling_map_edges)
        gate_set = ModelGateSet(
            num_qubits=self.num_qubits,
            qubit_subset=self.qubit_subset,
            coupling_map=coupling_map,
        )
        for gate_schema in self.gates:
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


class ApplyGateSchema(BaseModel):
    """Serialization schema for an :class:`ApplyGate` instruction."""

    type: Literal["apply_gate"] = "apply_gate"
    gate_name: str

    @classmethod
    def serialize(cls, instr: ApplyGate) -> Self:
        """Serialize an :class:`~.ApplyGate` instruction."""
        return cls(gate_name=instr.gate.name)

    def deserialize(self, gate_set: ModelGateSet) -> ApplyGate:
        """Deserialize to an :class:`~.ApplyGate` instruction."""
        return ApplyGate(gate_set[self.gate_name].model_gate)


class PartialPauliPermutationSchema(BaseModel):
    """Serialization schema for a :class:`PartialPauliPermutation` instruction."""

    type: Literal["partial_pauli_permutation"] = "partial_pauli_permutation"
    partial_permutation_indices: list[int]

    @classmethod
    def serialize(cls, instr: PartialPauliPermutation) -> Self:
        """Serialize a :class:`~.PartialPauliPermutation` instruction."""
        return cls(
            partial_permutation_indices=[int(x) for x in instr.partial_permutation_indices],
        )

    def deserialize(self, gate_set: ModelGateSet) -> PartialPauliPermutation:
        """Deserialize to a :class:`~.PartialPauliPermutation` instruction."""
        return PartialPauliPermutation(np.array(self.partial_permutation_indices, dtype=np.int8))


InstructionSchema = Annotated[
    ApplyGateSchema | PartialPauliPermutationSchema, Field(discriminator="type")
]


def _serialize_instruction(instr: Instruction) -> ApplyGateSchema | PartialPauliPermutationSchema:
    """Dispatch instruction serialization to the appropriate schema class."""
    if isinstance(instr, ApplyGate):
        return ApplyGateSchema.serialize(instr)
    elif isinstance(instr, PartialPauliPermutation):
        return PartialPauliPermutationSchema.serialize(instr)
    raise TypeError(f"Unknown instruction type: {type(instr)}")


class InstructionPatternSchema(BaseModel):
    """Serialization schema for an :class:`InstructionPattern`."""

    start_fragment: list[InstructionSchema]
    repeatable_fragment: list[InstructionSchema]
    end_fragment: list[InstructionSchema]


class InstructionSequenceSchema(BaseModel):
    """Serialization schema for an :class:`InstructionSequence`."""

    depth: int
    pattern: InstructionPatternSchema

    @classmethod
    def serialize(cls, seq: InstructionSequence) -> Self:
        """Serialize an :class:`~.InstructionSequence`."""
        return cls(
            depth=seq.depth,
            pattern=InstructionPatternSchema(
                start_fragment=[_serialize_instruction(i) for i in seq.pattern.start_fragment],
                repeatable_fragment=[
                    _serialize_instruction(i) for i in seq.pattern.repeatable_fragment
                ],
                end_fragment=[_serialize_instruction(i) for i in seq.pattern.end_fragment],
            ),
        )

    def deserialize(self, gate_set: ModelGateSet) -> InstructionSequence:
        """Deserialize to an :class:`~.InstructionSequence`."""
        pattern = InstructionPattern(
            start_fragment=[i.deserialize(gate_set) for i in self.pattern.start_fragment],
            repeatable_fragment=[i.deserialize(gate_set) for i in self.pattern.repeatable_fragment],
            end_fragment=[i.deserialize(gate_set) for i in self.pattern.end_fragment],
        )
        return InstructionSequence(pattern=pattern, depth=self.depth)


class FidelityIndexSchema(BaseModel):
    """Serialization schema for a :class:`FidelityIndex`."""

    gate_name: str
    pauli: QubitSparsePauliSchema
    in_bit_indices: list[int]
    out_bit_indices: list[int]

    @classmethod
    def serialize(cls, fi: FidelityIndex) -> Self:
        """Serialize a :class:`~.FidelityIndex`."""
        return cls(
            gate_name=fi.gate.name,
            pauli=QubitSparsePauliSchema.serialize(fi.pauli),
            in_bit_indices=sorted(fi.in_bit_indices),
            out_bit_indices=sorted(fi.out_bit_indices),
        )

    def deserialize(self, gate_set: ModelGateSet) -> FidelityIndex:
        """Deserialize to a :class:`~.FidelityIndex`."""
        gate = gate_set[self.gate_name].model_gate
        pauli = self.pauli.deserialize()
        return FidelityIndex(
            gate=gate,
            pauli=pauli,
            in_bit_indices=frozenset(self.in_bit_indices),
            out_bit_indices=frozenset(self.out_bit_indices),
        )


class PathPatternSchema(BaseModel):
    """Serialization schema for a :class:`PathPattern`."""

    start_fragment: list[FidelityIndexSchema]
    repeatable_fragment: list[FidelityIndexSchema]
    end_fragment: list[FidelityIndexSchema]


class PathSchema(BaseModel):
    """Serialization schema for a :class:`Path`."""

    depth: int
    pattern: PathPatternSchema

    @classmethod
    def serialize(cls, path: Path) -> Self:
        """Serialize a :class:`~.Path`."""
        return cls(
            depth=path.depth,
            pattern=PathPatternSchema(
                start_fragment=[
                    FidelityIndexSchema.serialize(fi) for fi in path.pattern.start_fragment
                ],
                repeatable_fragment=[
                    FidelityIndexSchema.serialize(fi) for fi in path.pattern.repeatable_fragment
                ],
                end_fragment=[
                    FidelityIndexSchema.serialize(fi) for fi in path.pattern.end_fragment
                ],
            ),
        )

    def deserialize(self, gate_set: ModelGateSet) -> Path:
        """Deserialize to a :class:`~.Path`."""
        pattern = PathPattern(
            start_fragment=[fi.deserialize(gate_set) for fi in self.pattern.start_fragment],
            repeatable_fragment=[
                fi.deserialize(gate_set) for fi in self.pattern.repeatable_fragment
            ],
            end_fragment=[fi.deserialize(gate_set) for fi in self.pattern.end_fragment],
        )
        return Path(pattern=pattern, depth=self.depth)


class PauliLindbladModelSchema(BaseModel):
    """Serialization schema for a :class:`PauliLindbladModel`."""

    generators: dict[str, QubitSparsePauliListSchema]
    noise_site: dict[str, str]

    @classmethod
    def serialize(cls, model: PauliLindbladModel) -> Self:
        """Serialize a :class:`~.PauliLindbladModel`."""
        generators = {}
        for gate_name, pauli_list in model.generators.items():
            generators[gate_name] = QubitSparsePauliListSchema.serialize(pauli_list)
        return cls(
            generators=generators,
            noise_site=dict(model.noise_site),
        )

    def deserialize(self, gate_set: ModelGateSet) -> PauliLindbladModel:
        """Deserialize to a :class:`~.PauliLindbladModel`."""
        generators = {}
        for gate_name, pauli_list_schema in self.generators.items():
            generators[gate_name] = pauli_list_schema.deserialize()
        return PauliLindbladModel(
            gate_set=gate_set,
            generators=generators,
            noise_site=self.noise_site,
        )
