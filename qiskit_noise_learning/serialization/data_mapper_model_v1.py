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

"""Versioned top-level serialization schema for the executor data mapper and analysis context."""

from typing import Literal, Self

from pydantic import BaseModel

from .schemas import (
    InstructionSequenceSchema,
    ModelGateSetSchema,
    PathSchema,
    PauliLindbladModelSchema,
)


class DataMapperModelV1(BaseModel):
    """Versioned serialization schema for :class:`ExecutorDataMapper` + analysis context.

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
    def from_passthrough_data(cls, data: dict) -> Self:
        """Reconstruct from a passthrough_data dict."""
        return cls.model_validate(data["noise_learning_data_mapper"])
