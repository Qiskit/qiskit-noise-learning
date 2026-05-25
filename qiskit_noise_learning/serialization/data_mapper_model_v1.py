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
    FidelityIndexSchema,
    InstructionSequenceSchema,
    ModelGateSetSchema,
    PathCompactSchema,
    PathPatternCompactSchema,
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
    fidelity_indices: list[FidelityIndexSchema]
    path_patterns: list[PathPatternCompactSchema]
    paths: list[PathCompactSchema]
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

        fi_to_index: dict = {}
        fi_list: list[FidelityIndexSchema] = []

        def _get_fi_index(fi) -> int:
            # Return the existing index for fi, or append it to fi_list and return the new index.
            if fi in fi_to_index:
                return fi_to_index[fi]
            idx = len(fi_list)
            fi_to_index[fi] = idx
            fi_list.append(FidelityIndexSchema.serialize(fi))
            return idx

        pattern_to_index: dict = {}
        pattern_list: list[PathPatternCompactSchema] = []

        def _get_pattern_index(pattern) -> int:
            # Return the existing index for pattern, or append it to pattern_list and return
            # the new index.
            if pattern in pattern_to_index:
                return pattern_to_index[pattern]
            idx = len(pattern_list)
            pattern_to_index[pattern] = idx
            pattern_list.append(
                PathPatternCompactSchema(
                    start_fragment=[_get_fi_index(fi) for fi in pattern.start_fragment],
                    repeatable_fragment=[_get_fi_index(fi) for fi in pattern.repeatable_fragment],
                    end_fragment=[_get_fi_index(fi) for fi in pattern.end_fragment],
                )
            )
            return idx

        compact_paths = []
        for path in data_mapper.paths:
            pattern_index = _get_pattern_index(path.pattern)
            compact_paths.append(PathCompactSchema(pattern_index=pattern_index, depth=path.depth))

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
            fidelity_indices=fi_list,
            path_patterns=pattern_list,
            paths=compact_paths,
            model=PauliLindbladModelSchema.serialize(data_mapper.fidelity_model),
        )

    def to_executor_data_mapper(self) -> ExecutorDataMapper:
        """Deserialize to an :class:`~.ExecutorDataMapper`."""
        from ..circuit_generator.executor_data_mapper import ExecutorDataMapper
        from ..sequences import Path, PathPattern

        gate_set = self.gate_set.deserialize()
        fidelity_model = self.model.deserialize(gate_set)

        fi_table = [fi.deserialize(gate_set) for fi in self.fidelity_indices]

        pattern_table = []
        for pattern_schema in self.path_patterns:
            pattern_table.append(
                PathPattern(
                    start_fragment=[fi_table[i] for i in pattern_schema.start_fragment],
                    repeatable_fragment=[fi_table[i] for i in pattern_schema.repeatable_fragment],
                    end_fragment=[fi_table[i] for i in pattern_schema.end_fragment],
                )
            )

        paths = [Path(pattern=pattern_table[p.pattern_index], depth=p.depth) for p in self.paths]

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
