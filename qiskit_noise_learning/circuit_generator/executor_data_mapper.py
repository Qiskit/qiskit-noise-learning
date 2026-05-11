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

"""ExecutorDataMapper and serialization helpers."""

from __future__ import annotations

import numpy as np
from qiskit.quantum_info import Clifford, QubitSparsePauli, QubitSparsePauliList
from qiskit.transpiler import CouplingMap

from ..gate_sets import ModelGate, ModelGateSet
from ..models import FidelityModel, PauliLindbladModel
from ..sequences import (
    ApplyGate,
    FidelityIndex,
    InstructionPattern,
    InstructionSequence,
    PartialPauliPermutation,
    Path,
    PathPattern,
)
from .data_mapper_model_v1 import (
    ApplyGateSchema,
    CliffordComponentSchema,
    DataMapperModelV1,
    FidelityIndexSchema,
    InstructionPatternSchema,
    InstructionSequenceSchema,
    ModelGateSchema,
    ModelGateSetSchema,
    PartialPauliPermutationSchema,
    PathPatternSchema,
    PathSchema,
    PauliLindbladModelSchema,
    QubitSparsePauliListSchema,
    QubitSparsePauliSchema,
)


class ExecutorDataMapper:
    """Map executor results into standard results.

    As instruction sequences with similar structure are generated together with a single template
    circuit and different samplex arguments, the order of input sequences to
    :meth:`.ExecutorCircuitGenerator.generate` is not preserved during execution. This class
    contains properties to format the results of a
    :class:`qiskit_ibm_runtime.quantum_program.QuantumProgramResult` to the order of the
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
        fidelity_model: FidelityModel | None = None,
        paths: list[Path] | None = None,
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
        :class:`qiskit_ibm_runtime.quantum_program.QuantumProgramResult` at the same index.
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
    def fidelity_model(self) -> FidelityModel | None:
        """The fidelity model associated with the data, if available."""
        return self._fidelity_model

    @property
    def paths(self) -> list[Path] | None:
        """The analysis paths associated with the data, if available."""
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
            gate_set=_serialize_gate_set(gate_set),
            item_sequence_indices=self._item_sequence_indices,
            creg_names=self._creg_names,
            measurement_maps=[
                {k: v.tolist() for k, v in m.items()} for m in self._measurement_maps
            ],
            num_randomizations=self._num_randomizations,
            instruction_sequences=[
                _serialize_instruction_sequence(seq) for seq in self._instruction_sequences
            ],
            paths=[_serialize_path(p) for p in self._paths],
            model=_serialize_pauli_lindblad_model(self._fidelity_model),
        )

    @classmethod
    def from_data_mapper_model(cls, model: DataMapperModelV1) -> ExecutorDataMapper:
        """Reconstruct an :class:`ExecutorDataMapper` from a :class:`DataMapperModelV1`."""
        gate_set = _deserialize_gate_set(model.gate_set)
        fidelity_model = _deserialize_pauli_lindblad_model(model.model, gate_set)
        paths = [_deserialize_path(p, gate_set) for p in model.paths]
        instruction_sequences = [
            _deserialize_instruction_sequence(seq, gate_set) for seq in model.instruction_sequences
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
    def from_passthrough_data(cls, data: dict) -> ExecutorDataMapper:
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


# --- Serialization helpers ---


def _serialize_qubit_sparse_pauli(pauli: QubitSparsePauli) -> QubitSparsePauliSchema:
    return QubitSparsePauliSchema(
        num_qubits=pauli.num_qubits,
        paulis=[int(p) for p in pauli.paulis],
        indices=[int(i) for i in pauli.indices],
    )


def _deserialize_qubit_sparse_pauli(schema: QubitSparsePauliSchema) -> QubitSparsePauli:
    return QubitSparsePauli.from_raw_parts(
        num_qubits=schema.num_qubits,
        paulis=schema.paulis,
        indices=schema.indices,
    )


def _serialize_qubit_sparse_pauli_list(
    pauli_list: QubitSparsePauliList,
) -> QubitSparsePauliListSchema:
    return QubitSparsePauliListSchema(
        num_qubits=pauli_list.num_qubits,
        paulis=[_serialize_qubit_sparse_pauli(p) for p in pauli_list],
    )


def _deserialize_qubit_sparse_pauli_list(
    schema: QubitSparsePauliListSchema,
) -> QubitSparsePauliList:
    paulis = [_deserialize_qubit_sparse_pauli(p) for p in schema.paulis]
    return QubitSparsePauliList.from_qubit_sparse_paulis(paulis)


def _serialize_gate_set(gate_set: ModelGateSet) -> ModelGateSetSchema:
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


def _deserialize_gate_set(schema: ModelGateSetSchema) -> ModelGateSet:
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


def _serialize_instruction_sequence(seq: InstructionSequence) -> InstructionSequenceSchema:
    return InstructionSequenceSchema(
        depth=seq.depth,
        pattern=InstructionPatternSchema(
            start_fragment=[_serialize_instruction(i) for i in seq.pattern.start_fragment],
            repeatable_fragment=[
                _serialize_instruction(i) for i in seq.pattern.repeatable_fragment
            ],
            end_fragment=[_serialize_instruction(i) for i in seq.pattern.end_fragment],
        ),
    )


def _serialize_instruction(instr) -> ApplyGateSchema | PartialPauliPermutationSchema:
    if isinstance(instr, ApplyGate):
        return ApplyGateSchema(gate_name=instr.gate.name)
    elif isinstance(instr, PartialPauliPermutation):
        return PartialPauliPermutationSchema(
            partial_permutation_indices=[int(x) for x in instr.partial_permutation_indices],
        )
    raise TypeError(f"Unknown instruction type: {type(instr)}")


def _deserialize_instruction_sequence(
    schema: InstructionSequenceSchema, gate_set: ModelGateSet
) -> InstructionSequence:
    pattern = InstructionPattern(
        start_fragment=[
            _deserialize_instruction(i, gate_set) for i in schema.pattern.start_fragment
        ],
        repeatable_fragment=[
            _deserialize_instruction(i, gate_set) for i in schema.pattern.repeatable_fragment
        ],
        end_fragment=[_deserialize_instruction(i, gate_set) for i in schema.pattern.end_fragment],
    )
    return InstructionSequence(pattern=pattern, depth=schema.depth)


def _deserialize_instruction(
    schema: ApplyGateSchema | PartialPauliPermutationSchema, gate_set: ModelGateSet
):
    if isinstance(schema, ApplyGateSchema):
        return ApplyGate(gate_set[schema.gate_name].model_gate)
    elif isinstance(schema, PartialPauliPermutationSchema):
        return PartialPauliPermutation(np.array(schema.partial_permutation_indices, dtype=np.int8))
    raise TypeError(f"Unknown instruction schema type: {type(schema)}")


def _serialize_path(path: Path) -> PathSchema:
    return PathSchema(
        depth=path.depth,
        pattern=PathPatternSchema(
            start_fragment=[_serialize_fidelity_index(fi) for fi in path.pattern.start_fragment],
            repeatable_fragment=[
                _serialize_fidelity_index(fi) for fi in path.pattern.repeatable_fragment
            ],
            end_fragment=[_serialize_fidelity_index(fi) for fi in path.pattern.end_fragment],
        ),
    )


def _serialize_fidelity_index(fi: FidelityIndex) -> FidelityIndexSchema:
    return FidelityIndexSchema(
        gate_name=fi.gate.name,
        pauli=_serialize_qubit_sparse_pauli(fi.pauli),
        in_bit_indices=sorted(fi.in_bit_indices),
        out_bit_indices=sorted(fi.out_bit_indices),
    )


def _deserialize_path(schema: PathSchema, gate_set: ModelGateSet) -> Path:
    pattern = PathPattern(
        start_fragment=[
            _deserialize_fidelity_index(fi, gate_set) for fi in schema.pattern.start_fragment
        ],
        repeatable_fragment=[
            _deserialize_fidelity_index(fi, gate_set) for fi in schema.pattern.repeatable_fragment
        ],
        end_fragment=[
            _deserialize_fidelity_index(fi, gate_set) for fi in schema.pattern.end_fragment
        ],
    )
    return Path(pattern=pattern, depth=schema.depth)


def _deserialize_fidelity_index(
    schema: FidelityIndexSchema, gate_set: ModelGateSet
) -> FidelityIndex:
    gate = gate_set[schema.gate_name].model_gate
    pauli = _deserialize_qubit_sparse_pauli(schema.pauli)
    return FidelityIndex(
        gate=gate,
        pauli=pauli,
        in_bit_indices=frozenset(schema.in_bit_indices),
        out_bit_indices=frozenset(schema.out_bit_indices),
    )


def _serialize_pauli_lindblad_model(model: PauliLindbladModel) -> PauliLindbladModelSchema:
    generators = {}
    for gate_name, pauli_list in model.generators.items():
        generators[gate_name] = _serialize_qubit_sparse_pauli_list(pauli_list)
    return PauliLindbladModelSchema(
        generators=generators,
        noise_site=dict(model.noise_site),
    )


def _deserialize_pauli_lindblad_model(
    schema: PauliLindbladModelSchema, gate_set: ModelGateSet
) -> PauliLindbladModel:
    generators = {}
    for gate_name, pauli_list_schema in schema.generators.items():
        generators[gate_name] = _deserialize_qubit_sparse_pauli_list(pauli_list_schema)
    return PauliLindbladModel(
        gate_set=gate_set,
        generators=generators,
        noise_site=schema.noise_site,
    )
