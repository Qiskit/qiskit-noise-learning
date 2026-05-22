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

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Self

import numpy as np
from pydantic import BaseModel

from .schemas import (
    InstructionSequenceSchema,
    ModelGateSetSchema,
    PathSchema,
    PauliLindbladModelSchema,
)

if TYPE_CHECKING:
    from ..circuit_generator.executor_data_mapper import ExecutorDataMapper


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

    @classmethod
    def from_executor_data_mapper(cls, data_mapper: ExecutorDataMapper) -> Self:
        """Serialize an :class:`~.ExecutorDataMapper` to a :class:`DataMapperModelV1`.

        Requires ``fidelity_model`` and ``paths`` to be set on the data mapper.

        Raises:
            ValueError: If ``fidelity_model`` or ``paths`` is ``None``.
            ValueError: If ``fidelity_model`` is not a ``PauliLindbladModel``.
        """
        from ..models import PauliLindbladModel

        if data_mapper.fidelity_model is None or data_mapper.paths is None:
            raise ValueError("Cannot serialize: fidelity_model and paths must be set.")
        if not isinstance(data_mapper.fidelity_model, PauliLindbladModel):
            raise ValueError("Serialization is only supported for PauliLindbladModel instances.")

        gate_set = data_mapper.fidelity_model.gate_set
        return cls(
            gate_set=ModelGateSetSchema.serialize(gate_set),
            item_sequence_indices=data_mapper.item_sequence_indices,
            creg_names=data_mapper.creg_names,
            measurement_maps=[
                {k: v.tolist() for k, v in m.items()} for m in data_mapper.measurement_maps
            ],
            num_randomizations=data_mapper.num_randomizations,
            instruction_sequences=[
                InstructionSequenceSchema.serialize(seq)
                for seq in data_mapper.instruction_sequences
            ],
            paths=[PathSchema.serialize(p) for p in data_mapper.paths],
            model=PauliLindbladModelSchema.serialize(data_mapper.fidelity_model),
        )

    def to_executor_data_mapper(self) -> ExecutorDataMapper:
        """Deserialize to an :class:`~.ExecutorDataMapper`."""
        from ..circuit_generator.executor_data_mapper import ExecutorDataMapper

        gate_set = self.gate_set.deserialize()
        fidelity_model = self.model.deserialize(gate_set)
        paths = [p.deserialize(gate_set) for p in self.paths]
        instruction_sequences = [seq.deserialize(gate_set) for seq in self.instruction_sequences]
        measurement_maps = [
            {k: np.array(v, dtype=int) for k, v in m.items()} for m in self.measurement_maps
        ]
        return ExecutorDataMapper(
            item_sequence_indices=self.item_sequence_indices,
            creg_names=self.creg_names,
            measurement_maps=measurement_maps,
            instruction_sequences=instruction_sequences,
            num_randomizations=self.num_randomizations,
            fidelity_model=fidelity_model,
            paths=paths,
        )
