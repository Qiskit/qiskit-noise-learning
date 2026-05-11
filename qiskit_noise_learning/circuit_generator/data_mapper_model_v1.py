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

"""Versioned serialization schema for the executor data mapper and analysis context."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field


class QubitSparsePauliSchema(BaseModel):
    """Serialization schema for a QubitSparsePauli."""

    num_qubits: int
    paulis: list[int]
    indices: list[int]


class QubitSparsePauliListSchema(BaseModel):
    """Serialization schema for a QubitSparsePauliList."""

    num_qubits: int
    paulis: list[QubitSparsePauliSchema]


class CliffordComponentSchema(BaseModel):
    """Serialization schema for a single Clifford in a ModelGate's clifford list."""

    qubit_idxs: list[int]
    tableau: list[list[bool]]


class ModelGateSchema(BaseModel):
    """Serialization schema for a ModelGate."""

    name: str
    qubit_idxs: list[int]
    meas_idxs: list[int]
    prep_idxs: list[int]
    cliffords: list[CliffordComponentSchema]


class ModelGateSetSchema(BaseModel):
    """Serialization schema for a ModelGateSet."""

    num_qubits: int
    qubit_subset: list[int]
    coupling_map_edges: list[list[int]]
    gates: list[ModelGateSchema]


class ApplyGateSchema(BaseModel):
    """Serialization schema for an ApplyGate instruction."""

    type: Literal["apply_gate"] = "apply_gate"
    gate_name: str


class PartialPauliPermutationSchema(BaseModel):
    """Serialization schema for a PartialPauliPermutation instruction."""

    type: Literal["partial_pauli_permutation"] = "partial_pauli_permutation"
    partial_permutation_indices: list[int]


InstructionSchema = Annotated[
    ApplyGateSchema | PartialPauliPermutationSchema, Field(discriminator="type")
]


class InstructionPatternSchema(BaseModel):
    """Serialization schema for an InstructionPattern."""

    start_fragment: list[InstructionSchema]
    repeatable_fragment: list[InstructionSchema]
    end_fragment: list[InstructionSchema]


class InstructionSequenceSchema(BaseModel):
    """Serialization schema for an InstructionSequence."""

    depth: int
    pattern: InstructionPatternSchema


class FidelityIndexSchema(BaseModel):
    """Serialization schema for a FidelityIndex."""

    gate_name: str
    pauli: QubitSparsePauliSchema
    in_bit_indices: list[int]
    out_bit_indices: list[int]


class PathPatternSchema(BaseModel):
    """Serialization schema for a PathPattern."""

    start_fragment: list[FidelityIndexSchema]
    repeatable_fragment: list[FidelityIndexSchema]
    end_fragment: list[FidelityIndexSchema]


class PathSchema(BaseModel):
    """Serialization schema for a Path."""

    depth: int
    pattern: PathPatternSchema


class PauliLindbladModelSchema(BaseModel):
    """Serialization schema for a PauliLindbladModel."""

    generators: dict[str, QubitSparsePauliListSchema]
    noise_site: dict[str, str]


class DataMapperModelV1(BaseModel):
    """Versioned serialization schema for ExecutorDataMapper + analysis context.

    This class is a pure data schema with no dependencies on executor-specific types.
    It can be serialized to/from the ``passthrough_data`` field of a ``QuantumProgram``.
    """

    version: Literal[1] = 1
    gate_set: ModelGateSetSchema
    item_sequence_indices: list[list[int]]
    creg_names: list[list[str]]
    measurement_maps: list[dict[str, list[int]]]
    num_randomizations: int
    instruction_sequences: list[InstructionSequenceSchema]
    paths: list[PathSchema]
    model: PauliLindbladModelSchema

    def to_passthrough_data(self) -> dict:
        """Serialize to a DataTree-compatible dict for QuantumProgram.passthrough_data."""
        return {"noise_learning_data_mapper": self.model_dump()}

    @classmethod
    def from_passthrough_data(cls, data: dict) -> DataMapperModelV1:
        """Reconstruct from a passthrough_data dict."""
        return cls.model_validate(data["noise_learning_data_mapper"])
