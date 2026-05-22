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

"""ExecutorDataMapper class."""

from typing import Self

import numpy as np

from ..models import FidelityModel, PauliLindbladModel
from ..sequences import InstructionSequence, Path
from ..serialization import (
    DataMapperModelV1,
    deserialize_gate_set,
    deserialize_instruction_sequence,
    deserialize_path,
    deserialize_pauli_lindblad_model,
    serialize_gate_set,
    serialize_instruction_sequence,
    serialize_path,
    serialize_pauli_lindblad_model,
)


class ExecutorDataMapper:
    """Map executor results into standard results.

    As instruction sequences with similar structure are generated together with a single template
    circuit and different samplex arguments, the order of input sequences to
    :meth:`.ExecutorCircuitGenerator.generate` is not preserved during execution. This class
    contains properties to format the results of a
    :class:`qiskit_ibm_runtime.results.QuantumProgramResult` to the order of the
    input sequences.

    Args:
        item_sequence_indices: For each program item, an ordered list of instruction sequence
            indices. Position in the list corresponds to the configuration index within the result
            item.
        creg_names: The name of classical registers in each program item.
        measurement_maps: For each program item, a dictionary from creg names to an ordered array
            of measured qubit indices.
        instruction_sequences: The instruction sequences associated with the data.
        num_randomizations: The number of randomizations used per experiment.
        fidelity_model: The fidelity model associated with the data.
        paths: The analysis paths associated with the data.

    """

    def __init__(
        self,
        item_sequence_indices: list[list[int]],
        creg_names: list[list[str]],
        measurement_maps: list[dict[str, np.ndarray[int]]],
        instruction_sequences: list[InstructionSequence],
        num_randomizations: int,
        fidelity_model: FidelityModel,
        paths: list[Path],
    ):
        self._item_sequence_indices = item_sequence_indices
        self._creg_names = creg_names
        self._measurement_maps = measurement_maps
        self._instruction_sequences = instruction_sequences
        self._num_randomizations = num_randomizations
        self._fidelity_model = fidelity_model
        self._paths = paths

    @property
    def item_sequence_indices(self) -> list[list[int]]:
        """Per program item, the instruction sequence indices corresponding to each config."""
        return self._item_sequence_indices

    @property
    def creg_names(self) -> list[list[str]]:
        """List of names of the classical registers contained in the results.

        The list at a given index corresponds to names expected in the data of the
        :class:`qiskit_ibm_runtime.results.QuantumProgramResult` at the same index.
        """
        return self._creg_names

    @property
    def measurement_maps(self) -> list[dict[str, np.ndarray[int]]]:
        """A per-program-item map from creg name to an ordered array of measured qubit indices."""
        return self._measurement_maps

    @property
    def instruction_sequences(self) -> list:
        """The instruction sequences corresponding to the sequence indices in the sequence map."""
        return self._instruction_sequences

    @property
    def num_randomizations(self) -> int:
        """The number of randomizations used per experiment."""
        return self._num_randomizations

    @property
    def fidelity_model(self) -> FidelityModel:
        """The fidelity model associated with the data."""
        return self._fidelity_model

    @property
    def paths(self) -> list[Path]:
        """The analysis paths associated with the data."""
        return self._paths

    def to_data_mapper_model(self) -> DataMapperModelV1:
        """Serialize this data mapper to a :class:`DataMapperModelV1`.

        Requires :attr:`fidelity_model` and :attr:`paths` to be set.

        Raises:
            ValueError: If ``fidelity_model`` or ``paths`` is ``None``.
            ValueError: If ``fidelity_model`` is not a ``PauliLindbladModel``.
        """
        if self._fidelity_model is None or self._paths is None:
            raise ValueError("Cannot serialize: fidelity_model and paths must be set.")
        if not isinstance(self._fidelity_model, PauliLindbladModel):
            raise ValueError("Serialization is only supported for PauliLindbladModel instances.")

        gate_set = self._fidelity_model.gate_set
        return DataMapperModelV1(
            gate_set=serialize_gate_set(gate_set),
            item_sequence_indices=self._item_sequence_indices,
            creg_names=self._creg_names,
            measurement_maps=[
                {k: v.tolist() for k, v in m.items()} for m in self._measurement_maps
            ],
            num_randomizations=self._num_randomizations,
            instruction_sequences=[
                serialize_instruction_sequence(seq) for seq in self._instruction_sequences
            ],
            paths=[serialize_path(p) for p in self._paths],
            model=serialize_pauli_lindblad_model(self._fidelity_model),
        )

    @classmethod
    def from_data_mapper_model(cls, model: DataMapperModelV1) -> Self:
        """Reconstruct an :class:`ExecutorDataMapper` from a :class:`DataMapperModelV1`."""
        gate_set = deserialize_gate_set(model.gate_set)
        fidelity_model = deserialize_pauli_lindblad_model(model.model, gate_set)
        paths = [deserialize_path(p, gate_set) for p in model.paths]
        instruction_sequences = [
            deserialize_instruction_sequence(seq, gate_set) for seq in model.instruction_sequences
        ]
        measurement_maps = [
            {k: np.array(v, dtype=int) for k, v in m.items()} for m in model.measurement_maps
        ]
        return cls(
            item_sequence_indices=model.item_sequence_indices,
            creg_names=model.creg_names,
            measurement_maps=measurement_maps,
            instruction_sequences=instruction_sequences,
            num_randomizations=model.num_randomizations,
            fidelity_model=fidelity_model,
            paths=paths,
        )

    @classmethod
    def from_passthrough_data(cls, data: dict) -> Self:
        """Reconstruct from passthrough_data, dispatching on version.

        Args:
            data: The ``passthrough_data`` dict from a ``QuantumProgramResult``.

        Raises:
            ValueError: If the version is unsupported or the data is malformed.
        """
        raw = data["noise_learning_data_mapper"]
        version = raw.get("version")
        if version == 1:
            model = DataMapperModelV1.model_validate(raw)
            return cls.from_data_mapper_model(model)
        raise ValueError(f"Unsupported data mapper model version: {version}")
