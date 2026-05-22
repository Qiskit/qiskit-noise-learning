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

from typing import Annotated, Literal

from pydantic import BaseModel, Field


class QubitSparsePauliSchema(BaseModel):
    """Serialization schema for a ``QubitSparsePauli``.

    Stores as an integer lists of Paulis and their corresponding qubit indices.
    """

    num_qubits: int
    paulis: list[int]
    indices: list[int]


class QubitSparsePauliListSchema(BaseModel):
    """Serialization schema for a ``QubitSparsePauliList``."""

    num_qubits: int
    paulis: list[QubitSparsePauliSchema]


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


class ApplyGateSchema(BaseModel):
    """Serialization schema for an :class:`ApplyGate` instruction."""

    type: Literal["apply_gate"] = "apply_gate"
    gate_name: str


class PartialPauliPermutationSchema(BaseModel):
    """Serialization schema for a :class:`PartialPauliPermutation` instruction."""

    type: Literal["partial_pauli_permutation"] = "partial_pauli_permutation"
    partial_permutation_indices: list[int]


InstructionSchema = Annotated[
    ApplyGateSchema | PartialPauliPermutationSchema, Field(discriminator="type")
]


class InstructionPatternSchema(BaseModel):
    """Serialization schema for an :class:`InstructionPattern`."""

    start_fragment: list[InstructionSchema]
    repeatable_fragment: list[InstructionSchema]
    end_fragment: list[InstructionSchema]


class InstructionSequenceSchema(BaseModel):
    """Serialization schema for an :class:`InstructionSequence`."""

    depth: int
    pattern: InstructionPatternSchema


class FidelityIndexSchema(BaseModel):
    """Serialization schema for a :class:`FidelityIndex`."""

    gate_name: str
    pauli: QubitSparsePauliSchema
    in_bit_indices: list[int]
    out_bit_indices: list[int]


class PathPatternSchema(BaseModel):
    """Serialization schema for a :class:`PathPattern`."""

    start_fragment: list[FidelityIndexSchema]
    repeatable_fragment: list[FidelityIndexSchema]
    end_fragment: list[FidelityIndexSchema]


class PathSchema(BaseModel):
    """Serialization schema for a :class:`Path`."""

    depth: int
    pattern: PathPatternSchema


class PauliLindbladModelSchema(BaseModel):
    """Serialization schema for a :class:`PauliLindbladModel`."""

    generators: dict[str, QubitSparsePauliListSchema]
    noise_site: dict[str, str]
