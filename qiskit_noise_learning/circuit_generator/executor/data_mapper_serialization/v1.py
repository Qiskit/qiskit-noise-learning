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

"""Version 1 of the serialized data mapper payload."""

from collections.abc import Hashable, Iterable, Sequence
from typing import Any, NotRequired, TypedDict

import numpy as np
from numpy.typing import NDArray
from qiskit.quantum_info import Clifford, QubitSparsePauliList
from qiskit.transpiler import CouplingMap

from ....gate_sets import ModelGate, ModelGateSet
from ....models import FidelityModel, IdentityFidelityModel, PauliLindbladModel
from ....sequences import (
    ApplyGate,
    FidelityIndex,
    InstructionSequence,
    PartialPauliPermutation,
    Path,
)
from ....sequences.instruction import Instruction
from ..executor_data_mapper import ExecutorDataMapper
from .arrays import IDX, pack_paulis, pack_ragged, unpack_paulis, unpack_ragged

VERSION = 1
"""The version number written into payloads by this module."""

_UNBOUND = -1
"""Written in place of an unbound sequence's fragment depth of ``None``."""

_APPLY_GATE = 0
_PAULI_PERMUTATION = 1
_IDENTITY_MODEL = "identity"
_PAULI_LINDBLAD_MODEL = "pauli_lindblad"


# ------------------------------------------------------------------------------------------------
# The format
# ------------------------------------------------------------------------------------------------


class Payload(TypedDict):
    """A serialized data mapper."""

    version: int
    """The format version."""

    num_qubits: int
    """The qubit count shared by every Pauli in the payload."""

    num_randomizations: int
    """How many randomizations were used per experiment."""

    gate_names: list[str]
    """Every gate name the payload refers to by position, sorted."""

    layout: "LayoutPayload"
    """How result arrays map back onto the instruction sequences."""

    sequences: "SequencesPayload"
    """The instruction sequences."""

    paths: "PathsPayload | None"
    """The analysis paths, or ``None`` if the mapper carried none."""

    relations: "RelationsPayload | None"
    """The path to sequence relations, or ``None`` if the mapper carried none."""

    model: "ModelPayload | None"
    """The fidelity model, or ``None`` if the mapper carried none."""


class LayoutPayload(TypedDict):
    """How the arrays in a program's results map back onto the instruction sequences."""

    item_sequence_idxs: NDArray[IDX]
    """The sequence positions belonging to every program item, concatenated."""

    item_sequence_lengths: NDArray[IDX]
    """How many sequence positions each program item holds."""

    item_creg_names: list[list[str]]
    """The classical register names in each program item."""

    item_clbit_qubit_idxs: NDArray[IDX]
    """The measured qubit of every classical bit, concatenated over registers and items.

    Ordered to match ``item_creg_names`` read item by item, and register by register within an item.
    """

    item_clbit_lengths: NDArray[IDX]
    """How many classical bits each of those registers has."""


class SequencesPayload(TypedDict):
    """The instruction sequences the results are indexed by."""

    structure: "SequenceStructurePayload"
    """How the sequences are cut out of ``instruction_idxs``."""

    instruction_idxs: NDArray[IDX]
    """For each instruction of each sequence in turn, its position in ``instructions``."""

    instructions: "InstructionsPayload"
    """The distinct instructions, stored once each."""


class PathsPayload(TypedDict):
    """The analysis paths."""

    structure: "SequenceStructurePayload"
    """How the paths are cut out of ``fidelity_idxs``."""

    fidelity_idxs: NDArray[IDX]
    """For each fidelity index of each path in turn, its position in ``fidelity_indices``."""

    fidelity_indices: "FidelityIndicesPayload"
    """The distinct fidelity indices, stored once each."""


class RelationsPayload(TypedDict):
    """Which paths are traversed by which instruction sequences, sorted."""

    path_idxs: NDArray[IDX]
    """The path position of each relation."""

    sequence_idxs: NDArray[IDX]
    """The instruction sequence position of each relation."""


class ModelPayload(TypedDict):
    """A fidelity model and the gate set it is built on.

    The gate set entries are always present. The generator entries are present only when ``kind`` is
    ``_PAULI_LINDBLAD_MODEL``, an identity model having no parameters of its own.
    """

    kind: str
    """Which model this is: ``_IDENTITY_MODEL`` or ``_PAULI_LINDBLAD_MODEL``."""

    num_qubits: int
    """How many qubits the modelled device has."""

    qubit_subset: NDArray[IDX]
    """The qubits of interest, sorted."""

    coupling_map_controls: NDArray[IDX]
    """The first qubit of each coupling map edge, the edges being sorted."""

    coupling_map_targets: NDArray[IDX]
    """The second qubit of each coupling map edge."""

    gate_names: list[str]
    """The gate names, sorted. Every other gate array here is in this order."""

    gate_qubit_idxs: NDArray[IDX]
    """The sorted qubits of every gate, concatenated."""

    gate_qubit_lengths: NDArray[IDX]
    """How many qubits each gate acts on."""

    gate_meas_idxs: NDArray[IDX]
    """The sorted measured qubits of every gate, concatenated."""

    gate_meas_lengths: NDArray[IDX]
    """How many qubits each gate measures."""

    gate_prep_idxs: NDArray[IDX]
    """The sorted prepared qubits of every gate, concatenated."""

    gate_prep_lengths: NDArray[IDX]
    """How many qubits each gate prepares."""

    clifford_counts: NDArray[IDX]
    """How many Cliffords each gate is composed of."""

    clifford_qubit_idxs: NDArray[IDX]
    """The qubits of every Clifford, concatenated, in each gate's temporal order."""

    clifford_qubit_lengths: NDArray[IDX]
    """How many qubits each Clifford acts on."""

    clifford_num_qubits: NDArray[IDX]
    """The qubit count of each Clifford, which fixes its tableau's shape."""

    clifford_tableaus: NDArray[np.bool_]
    """Every Clifford's tableau, flattened and concatenated.

    Booleans, which the transport bit-packs, so these cost an eighth of their length.
    """

    generator_gate_names: NotRequired[list[str]]
    """The gates that have generators, sorted."""

    generator_counts: NotRequired[NDArray[IDX]]
    """How many generators each of those gates has."""

    generator_terms: NotRequired[NDArray[np.uint8]]
    """The Pauli terms of every generator, concatenated."""

    generator_idxs: NotRequired[NDArray[IDX]]
    """The qubit indices of every generator, concatenated."""

    generator_lengths: NotRequired[NDArray[IDX]]
    """The number of terms in each generator."""

    noise_site: NotRequired[dict[str, str]]
    """Whether each gate's noise is modelled before or after it."""


class SequenceStructurePayload(TypedDict):
    """How a list of instruction sequences or paths is cut out of one flat list of its elements.

    Both are fragment chains, so both are described the same way: three fragment lengths and a
    fragment depth each.
    """

    start_lengths: NDArray[IDX]
    """The number of elements in each one's start fragment."""

    repeatable_lengths: NDArray[IDX]
    """The number of elements in each one's repeatable fragment."""

    end_lengths: NDArray[IDX]
    """The number of elements in each one's end fragment."""

    fragment_depths: NDArray[np.int32]
    """Each one's fragment depth, or ``_UNBOUND`` where it has none."""


class InstructionsPayload(TypedDict):
    """The distinct instructions a payload's sequences are built from."""

    kinds: NDArray[np.uint8]
    """Which kind of instruction each one is: ``_APPLY_GATE`` or ``_PAULI_PERMUTATION``."""

    gates: NDArray[IDX]
    """For an applied gate, its position in the payload's ``gate_names``; zero otherwise."""

    permutations: NDArray[np.int8]
    """The permutation indices of every Pauli permutation, concatenated."""

    permutation_lengths: NDArray[IDX]
    """The number of indices belonging to each Pauli permutation."""


class FidelityIndicesPayload(TypedDict):
    """The distinct fidelity indices a payload's paths are built from.

    Every array here has one entry per fidelity index, except the concatenations, which have one
    entry per Pauli term or qubit index and are cut up by their accompanying lengths.
    """

    gates: NDArray[IDX]
    """Each one's gate, as a position in the payload's ``gate_names``."""

    sign_flips: NDArray[np.bool_]
    """Whether each one flips the sign."""

    pauli_terms: NDArray[np.uint8]
    """The Pauli terms of every ``pauli``, concatenated."""

    pauli_idxs: NDArray[IDX]
    """The qubit indices of every ``pauli``, concatenated."""

    pauli_lengths: NDArray[IDX]
    """The number of terms in each ``pauli``."""

    input_terms: NDArray[np.uint8]
    """The Pauli terms of every input Pauli, concatenated."""

    input_idxs: NDArray[IDX]
    """The qubit indices of every input Pauli, concatenated."""

    input_lengths: NDArray[IDX]
    """The number of terms in each input Pauli."""

    output_terms: NDArray[np.uint8]
    """The Pauli terms of every output Pauli, concatenated."""

    output_idxs: NDArray[IDX]
    """The qubit indices of every output Pauli, concatenated."""

    output_lengths: NDArray[IDX]
    """The number of terms in each output Pauli."""

    in_z_idxs: NDArray[IDX]
    """The sorted incoming Z qubit indices of every one, concatenated."""

    in_z_lengths: NDArray[IDX]
    """How many incoming Z qubit indices each one has."""

    out_z_idxs: NDArray[IDX]
    """The sorted outgoing Z qubit indices of every one, concatenated."""

    out_z_lengths: NDArray[IDX]
    """How many outgoing Z qubit indices each one has."""

    meas_idxs: NDArray[IDX]
    """The sorted measured qubit indices of every one, concatenated."""

    meas_lengths: NDArray[IDX]
    """How many measured qubit indices each one has."""


# ------------------------------------------------------------------------------------------------
# Reading and writing a whole payload
# ------------------------------------------------------------------------------------------------


def write(mapper: ExecutorDataMapper) -> Payload:
    """Serialize a data mapper.

    Args:
        mapper: The data mapper to serialize.

    Returns:
        The payload, holding only the leaf types the executor's passthrough data accepts: arrays,
        strings, integers, booleans, ``None``, and lists and dictionaries of those.

    Raises:
        TypeError: If the mapper carries a fidelity model this version cannot write.
    """
    gate_names, gate_idxs = _gate_name_table(mapper)
    instructions, instruction_idxs = _intern(
        _flatten(mapper.instruction_sequences), _instruction_key
    )

    paths: PathsPayload | None = None
    if mapper.paths is not None:
        fidelity_indices, fidelity_idxs = _intern(_flatten(mapper.paths), _fidelity_index_key)
        paths = {
            "structure": _write_sequence_structure(mapper.paths),
            "fidelity_idxs": fidelity_idxs,
            "fidelity_indices": _write_fidelity_indices(fidelity_indices, gate_idxs),
        }

    return {
        "version": VERSION,
        "num_qubits": _num_qubits(mapper),
        "num_randomizations": int(mapper.num_randomizations),
        "gate_names": gate_names,
        "layout": _write_layout(mapper),
        "sequences": {
            "structure": _write_sequence_structure(mapper.instruction_sequences),
            "instruction_idxs": instruction_idxs,
            "instructions": _write_instructions(instructions, gate_idxs),
        },
        "paths": paths,
        "relations": _write_relations(mapper.relations),
        "model": _write_model(mapper.fidelity_model),
    }


def read(payload: Payload) -> ExecutorDataMapper:
    """Rebuild the data mapper that :func:`write` was given.

    Args:
        payload: A payload written by :func:`write`.

    Returns:
        The data mapper. Fields whose order carries no meaning, such as the relations, may come
        back ordered differently to the mapper that was written.
    """
    gate_names = [str(name) for name in payload["gate_names"]]
    num_qubits = int(payload["num_qubits"])

    sequences_payload = payload["sequences"]
    table = _read_instructions(sequences_payload["instructions"], gate_names)
    instructions = [table[n] for n in sequences_payload["instruction_idxs"].tolist()]
    sequences = _rebuild_sequences(
        InstructionSequence, instructions, sequences_payload["structure"]
    )

    paths = None
    if (paths_payload := payload["paths"]) is not None:
        fidelity_table = _read_fidelity_indices(
            paths_payload["fidelity_indices"], gate_names, num_qubits
        )
        fidelity_indices = [fidelity_table[n] for n in paths_payload["fidelity_idxs"].tolist()]
        paths = _rebuild_sequences(Path, fidelity_indices, paths_payload["structure"])

    return ExecutorDataMapper(
        instruction_sequences=sequences,
        num_randomizations=int(payload["num_randomizations"]),
        paths=paths,
        relations=_read_relations(payload["relations"]),
        fidelity_model=_read_model(payload["model"]),
        **_read_layout(payload["layout"]),
    )


# ------------------------------------------------------------------------------------------------
# Building the sections: interning
# ------------------------------------------------------------------------------------------------


def _instruction_key(instruction: Instruction) -> Hashable:
    """Return a key distinguishing instructions that are written differently."""
    if isinstance(instruction, ApplyGate):
        return (_APPLY_GATE, instruction.gate_name)
    return (_PAULI_PERMUTATION, instruction.partial_permutation_indices.tobytes())


def _fidelity_index_key(fidelity_index: FidelityIndex) -> Hashable:
    """Return a key covering all eight fields of a fidelity index."""
    input_pauli, output_pauli = fidelity_index.transition
    return (
        fidelity_index.gate_name,
        fidelity_index.pauli.paulis.tobytes(),
        fidelity_index.pauli.indices.tobytes(),
        input_pauli.paulis.tobytes(),
        input_pauli.indices.tobytes(),
        output_pauli.paulis.tobytes(),
        output_pauli.indices.tobytes(),
        fidelity_index.sign_flip,
        tuple(sorted(fidelity_index.in_z_idxs)),
        tuple(sorted(fidelity_index.out_z_idxs)),
        tuple(sorted(fidelity_index.meas_idxs)),
    )


def _intern(elements: Sequence[Any], key_of: Any) -> tuple[list[Any], NDArray[IDX]]:
    """Return the distinct elements in first-seen order, and each element's index into them.

    Args:
        elements: The elements to deduplicate.
        key_of: Returns the key that decides whether two elements are the same.

    Returns:
        A tuple of the distinct elements and the index of each original element into them.
    """
    table: list[Any] = []
    positions: dict[Hashable, int] = {}
    idxs = np.empty(len(elements), dtype=IDX)
    for n, element in enumerate(elements):
        key = key_of(element)
        if (position := positions.get(key)) is None:
            position = positions[key] = len(table)
            table.append(element)
        idxs[n] = position
    return table, idxs


# ------------------------------------------------------------------------------------------------
# Sequence and path structure
# ------------------------------------------------------------------------------------------------


def _flatten(sequences: Iterable[Any]) -> list[Any]:
    """Return every element of every sequence, each sequence's fragments in order."""
    return [
        element
        for sequence in sequences
        for element in (
            *sequence.start_fragment,
            *sequence.repeatable_fragment,
            *sequence.end_fragment,
        )
    ]


def _write_sequence_structure(sequences: Sequence[Any]) -> SequenceStructurePayload:
    """Write the fragment lengths and bound depth of each sequence or path.

    One array per fragment rather than one array of triples, for the reason given in
    :func:`_write_relations`: each array is compressed separately, and a column of like values
    compresses far better than three interleaved ones.
    """
    return {
        "start_lengths": np.array([len(s.start_fragment) for s in sequences], dtype=IDX),
        "repeatable_lengths": np.array([len(s.repeatable_fragment) for s in sequences], dtype=IDX),
        "end_lengths": np.array([len(s.end_fragment) for s in sequences], dtype=IDX),
        "fragment_depths": np.array(
            [_UNBOUND if s.fragment_depth is None else s.fragment_depth for s in sequences],
            dtype=np.int32,
        ),
    }


def _rebuild_sequences(
    cls: type, elements: Sequence[Any], payload: SequenceStructurePayload
) -> list[Any]:
    """Rebuild sequences or paths by cutting a flat element list at the written lengths."""
    out, cursor = [], 0
    for n_start, n_repeat, n_end, depth in zip(
        payload["start_lengths"].tolist(),
        payload["repeatable_lengths"].tolist(),
        payload["end_lengths"].tolist(),
        payload["fragment_depths"].tolist(),
    ):
        start, cursor = elements[cursor : cursor + n_start], cursor + n_start
        repeatable, cursor = elements[cursor : cursor + n_repeat], cursor + n_repeat
        end, cursor = elements[cursor : cursor + n_end], cursor + n_end
        out.append(
            cls(
                start_fragment=start,
                repeatable_fragment=repeatable,
                end_fragment=end,
                fragment_depth=None if depth == _UNBOUND else depth,
            )
        )
    return out


# ------------------------------------------------------------------------------------------------
# Instructions
# ------------------------------------------------------------------------------------------------


def _write_instructions(
    instructions: Sequence[Instruction], gate_idxs: dict[str, int]
) -> InstructionsPayload:
    """Write instructions as parallel arrays, permutations concatenated separately."""
    kinds = np.array(
        [
            _APPLY_GATE if isinstance(instruction, ApplyGate) else _PAULI_PERMUTATION
            for instruction in instructions
        ],
        dtype=np.uint8,
    )
    gates = np.array(
        [
            gate_idxs[instruction.gate_name] if isinstance(instruction, ApplyGate) else 0
            for instruction in instructions
        ],
        dtype=IDX,
    )
    permutations, lengths = pack_ragged(
        [
            instruction.partial_permutation_indices
            for instruction in instructions
            if isinstance(instruction, PartialPauliPermutation)
        ],
        dtype=np.int8,
    )
    return {
        "kinds": kinds,
        "gates": gates,
        "permutations": permutations,
        "permutation_lengths": lengths,
    }


def _read_instructions(
    payload: InstructionsPayload, gate_names: Sequence[str]
) -> list[Instruction]:
    """Rebuild the instructions written by :func:`_write_instructions`."""
    permutations = unpack_ragged(payload["permutations"], payload["permutation_lengths"])
    out: list[Instruction] = []
    cursor = 0
    for kind, gate in zip(payload["kinds"].tolist(), payload["gates"].tolist()):
        if kind == _APPLY_GATE:
            out.append(ApplyGate(gate_names[gate]))
        else:
            out.append(PartialPauliPermutation(permutations[cursor]))
            cursor += 1
    return out


# ------------------------------------------------------------------------------------------------
# Fidelity indices
# ------------------------------------------------------------------------------------------------


def _write_fidelity_indices(
    fidelity_indices: Sequence[FidelityIndex], gate_idxs: dict[str, int]
) -> FidelityIndicesPayload:
    """Write fidelity indices as parallel arrays, one concatenation per field."""
    transitions = [fidelity_index.transition for fidelity_index in fidelity_indices]
    pauli_terms, pauli_idxs, pauli_lengths = pack_paulis(
        [fidelity_index.pauli for fidelity_index in fidelity_indices]
    )
    input_terms, input_idxs, input_lengths = pack_paulis([pair[0] for pair in transitions])
    output_terms, output_idxs, output_lengths = pack_paulis([pair[1] for pair in transitions])
    in_z, in_z_lengths = pack_ragged(
        [sorted(fidelity_index.in_z_idxs) for fidelity_index in fidelity_indices]
    )
    out_z, out_z_lengths = pack_ragged(
        [sorted(fidelity_index.out_z_idxs) for fidelity_index in fidelity_indices]
    )
    meas, meas_lengths = pack_ragged(
        [sorted(fidelity_index.meas_idxs) for fidelity_index in fidelity_indices]
    )
    return {
        "gates": np.array(
            [gate_idxs[fidelity_index.gate_name] for fidelity_index in fidelity_indices],
            dtype=IDX,
        ),
        "sign_flips": np.array(
            [fidelity_index.sign_flip for fidelity_index in fidelity_indices], dtype=bool
        ),
        "pauli_terms": pauli_terms,
        "pauli_idxs": pauli_idxs,
        "pauli_lengths": pauli_lengths,
        "input_terms": input_terms,
        "input_idxs": input_idxs,
        "input_lengths": input_lengths,
        "output_terms": output_terms,
        "output_idxs": output_idxs,
        "output_lengths": output_lengths,
        "in_z_idxs": in_z,
        "in_z_lengths": in_z_lengths,
        "out_z_idxs": out_z,
        "out_z_lengths": out_z_lengths,
        "meas_idxs": meas,
        "meas_lengths": meas_lengths,
    }


def _read_fidelity_indices(
    payload: FidelityIndicesPayload, gate_names: Sequence[str], num_qubits: int
) -> list[FidelityIndex]:
    """Rebuild the fidelity indices written by :func:`_write_fidelity_indices`."""
    paulis = unpack_paulis(
        payload["pauli_terms"], payload["pauli_idxs"], payload["pauli_lengths"], num_qubits
    )
    inputs = unpack_paulis(
        payload["input_terms"], payload["input_idxs"], payload["input_lengths"], num_qubits
    )
    outputs = unpack_paulis(
        payload["output_terms"], payload["output_idxs"], payload["output_lengths"], num_qubits
    )
    in_z = _read_index_sets(payload["in_z_idxs"], payload["in_z_lengths"])
    out_z = _read_index_sets(payload["out_z_idxs"], payload["out_z_lengths"])
    meas = _read_index_sets(payload["meas_idxs"], payload["meas_lengths"])
    gates = payload["gates"].tolist()
    sign_flips = payload["sign_flips"].tolist()
    return [
        FidelityIndex(
            gate_name=gate_names[gates[n]],
            pauli=paulis[n],
            in_z_idxs=in_z[n],
            out_z_idxs=out_z[n],
            input_pauli=inputs[n],
            output_pauli=outputs[n],
            sign_flip=sign_flips[n],
            meas_idxs=meas[n],
        )
        for n in range(len(paulis))
    ]


def _read_index_sets(flat: NDArray[Any], lengths: NDArray[IDX]) -> list[frozenset[int]]:
    """Rebuild a ragged array of qubit indices as frozen sets of Python integers."""
    return [frozenset(row.tolist()) for row in unpack_ragged(flat, lengths)]


# ------------------------------------------------------------------------------------------------
# Layout
# ------------------------------------------------------------------------------------------------


def _write_layout(mapper: ExecutorDataMapper) -> LayoutPayload:
    """Write the fields recording how result arrays map back onto sequences."""
    sequence_idxs, sequence_lengths = pack_ragged(mapper.item_sequence_indices)
    clbit_idxs, clbit_lengths = pack_ragged(
        [
            mapper.item_clbit_qubit_idxs[item][name]
            for item, names in enumerate(mapper.item_creg_names)
            for name in names
        ]
    )
    return {
        "item_sequence_idxs": sequence_idxs,
        "item_sequence_lengths": sequence_lengths,
        "item_creg_names": [list(names) for names in mapper.item_creg_names],
        "item_clbit_qubit_idxs": clbit_idxs,
        "item_clbit_lengths": clbit_lengths,
    }


def _read_layout(payload: LayoutPayload) -> dict[str, Any]:
    """Rebuild the layout fields as keyword arguments for the data mapper."""
    creg_names = [list(names) for names in payload["item_creg_names"]]
    clbit_rows = unpack_ragged(payload["item_clbit_qubit_idxs"], payload["item_clbit_lengths"])
    clbit_qubit_idxs, cursor = [], 0
    for names in creg_names:
        clbit_qubit_idxs.append({name: clbit_rows[cursor + n] for n, name in enumerate(names)})
        cursor += len(names)
    sequence_rows = unpack_ragged(payload["item_sequence_idxs"], payload["item_sequence_lengths"])
    return {
        "item_sequence_indices": [row.tolist() for row in sequence_rows],
        "item_creg_names": creg_names,
        "item_clbit_qubit_idxs": clbit_qubit_idxs,
    }


# ------------------------------------------------------------------------------------------------
# Relations
# ------------------------------------------------------------------------------------------------


def _write_relations(relations: set[tuple[int, int]] | None) -> RelationsPayload | None:
    """Write relations as one sorted array of path indices and one of sequence indices.

    Sorted first, that being the canonical order for a set of index pairs. The two columns are
    written as separate arrays rather than one array of pairs because each is compressed on its own:
    a sorted column of path indices is nearly monotonic and compresses to very little, whereas
    interleaving the two destroys that structure. Measured at 196 qubits, interleaving cost nine
    times the bytes.
    """
    if relations is None:
        return None
    pairs = sorted(relations)
    return {
        "path_idxs": np.array([path for path, _ in pairs], dtype=IDX),
        "sequence_idxs": np.array([sequence for _, sequence in pairs], dtype=IDX),
    }


def _read_relations(payload: RelationsPayload | None) -> set[tuple[int, int]] | None:
    """Rebuild the relations written by :func:`_write_relations`."""
    if payload is None:
        return None
    return set(zip(payload["path_idxs"].tolist(), payload["sequence_idxs"].tolist()))


# ------------------------------------------------------------------------------------------------
# Gate set and model
# ------------------------------------------------------------------------------------------------


def _write_model(model: FidelityModel | None) -> ModelPayload | None:
    """Write a fidelity model, including the gate set it is built on."""
    if model is None:
        return None
    if not isinstance(model, IdentityFidelityModel | PauliLindbladModel):
        raise TypeError(
            f"Cannot serialize a fidelity model of type '{type(model).__name__}'; version "
            f"{VERSION} of the payload format writes identity and Pauli-Lindblad models only."
        )

    gate_set = model.gate_set.model_gate_set
    names = sorted(gate_set)
    gates = [gate_set[name] for name in names]
    cliffords = [clifford for gate in gates for clifford in gate.cliffords]
    clifford_qubits, clifford_qubit_lengths = pack_ragged([idxs for idxs, _ in cliffords])
    payload: ModelPayload = {
        "kind": (
            _PAULI_LINDBLAD_MODEL if isinstance(model, PauliLindbladModel) else _IDENTITY_MODEL
        ),
        "num_qubits": int(gate_set.num_qubits),
        "qubit_subset": np.array(sorted(gate_set.qubit_subset), dtype=IDX),
        "coupling_map_controls": np.array(
            [control for control, _ in sorted(gate_set.coupling_map.get_edges())], dtype=IDX
        ),
        "coupling_map_targets": np.array(
            [target for _, target in sorted(gate_set.coupling_map.get_edges())], dtype=IDX
        ),
        "gate_names": names,
        "gate_qubit_idxs": pack_ragged([sorted(gate.qubit_idxs) for gate in gates])[0],
        "gate_qubit_lengths": pack_ragged([sorted(gate.qubit_idxs) for gate in gates])[1],
        "gate_meas_idxs": pack_ragged([sorted(gate.meas_idxs) for gate in gates])[0],
        "gate_meas_lengths": pack_ragged([sorted(gate.meas_idxs) for gate in gates])[1],
        "gate_prep_idxs": pack_ragged([sorted(gate.prep_idxs) for gate in gates])[0],
        "gate_prep_lengths": pack_ragged([sorted(gate.prep_idxs) for gate in gates])[1],
        "clifford_counts": np.array([len(gate.cliffords) for gate in gates], dtype=IDX),
        "clifford_qubit_idxs": clifford_qubits,
        "clifford_qubit_lengths": clifford_qubit_lengths,
        "clifford_num_qubits": np.array(
            [clifford.num_qubits for _, clifford in cliffords], dtype=IDX
        ),
        # Tableaux are bool, which the transport bit-packs, so they cost an eighth of their length.
        "clifford_tableaus": (
            np.concatenate([clifford.tableau.reshape(-1) for _, clifford in cliffords])
            if cliffords
            else np.empty(0, dtype=bool)
        ),
    }

    if isinstance(model, PauliLindbladModel):
        generator_names = sorted(model.generators)
        terms = [term for name in generator_names for term in model.generators[name]]
        generator_terms, generator_idxs, generator_lengths = pack_paulis(terms)
        payload |= {
            "generator_gate_names": generator_names,
            "generator_counts": np.array(
                [len(model.generators[name]) for name in generator_names], dtype=IDX
            ),
            "generator_terms": generator_terms,
            "generator_idxs": generator_idxs,
            "generator_lengths": generator_lengths,
            "noise_site": dict(model.noise_site),
        }
    return payload


def _read_model(payload: ModelPayload | None) -> FidelityModel | None:
    """Rebuild the fidelity model written by :func:`_write_model`."""
    if payload is None:
        return None
    num_qubits = int(payload["num_qubits"])
    gate_set = ModelGateSet(
        num_qubits,
        qubit_subset=payload["qubit_subset"].tolist(),
        coupling_map=CouplingMap(
            list(
                zip(
                    payload["coupling_map_controls"].tolist(),
                    payload["coupling_map_targets"].tolist(),
                )
            )
        ),
    )

    qubit_rows = unpack_ragged(payload["gate_qubit_idxs"], payload["gate_qubit_lengths"])
    meas_rows = unpack_ragged(payload["gate_meas_idxs"], payload["gate_meas_lengths"])
    prep_rows = unpack_ragged(payload["gate_prep_idxs"], payload["gate_prep_lengths"])
    clifford_qubit_rows = unpack_ragged(
        payload["clifford_qubit_idxs"], payload["clifford_qubit_lengths"]
    )
    tableaus = _read_tableaus(payload["clifford_tableaus"], payload["clifford_num_qubits"])

    clifford_cursor = 0
    for n, (name, count) in enumerate(
        zip(payload["gate_names"], payload["clifford_counts"].tolist())
    ):
        gate_set.add_gate(
            ModelGate(
                str(name),
                cliffords=[
                    (
                        tuple(clifford_qubit_rows[clifford_cursor + m].tolist()),
                        tableaus[clifford_cursor + m],
                    )
                    for m in range(count)
                ],
                qubit_idxs=qubit_rows[n].tolist(),
                meas_idxs=meas_rows[n].tolist(),
                prep_idxs=prep_rows[n].tolist(),
            )
        )
        clifford_cursor += count

    if payload["kind"] == _IDENTITY_MODEL:
        return IdentityFidelityModel(gate_set)

    terms = unpack_paulis(
        payload["generator_terms"],
        payload["generator_idxs"],
        payload["generator_lengths"],
        num_qubits,
    )
    generators, cursor = {}, 0
    for name, count in zip(payload["generator_gate_names"], payload["generator_counts"].tolist()):
        generators[str(name)] = QubitSparsePauliList.from_qubit_sparse_paulis(
            terms[cursor : cursor + count], num_qubits
        )
        cursor += count
    return PauliLindbladModel(
        gate_set=gate_set, generators=generators, noise_site=dict(payload["noise_site"])
    )


def _read_tableaus(flat: NDArray[np.bool_], num_qubits: NDArray[IDX]) -> list[Clifford]:
    """Rebuild Cliffords from one concatenated tableau array and their per-Clifford qubit counts."""
    out, cursor = [], 0
    for n in num_qubits.tolist():
        rows, columns = 2 * n, 2 * n + 1
        out.append(Clifford(flat[cursor : cursor + rows * columns].reshape(rows, columns)))
        cursor += rows * columns
    return out


# ------------------------------------------------------------------------------------------------
# Whole payload
# ------------------------------------------------------------------------------------------------


def _gate_name_table(mapper: ExecutorDataMapper) -> tuple[list[str], dict[str, int]]:
    """Return every gate name the payload refers to by index, and the inverse lookup."""
    names = {
        instruction.gate_name
        for instruction in _flatten(mapper.instruction_sequences)
        if isinstance(instruction, ApplyGate)
    }
    names.update(fidelity_index.gate_name for fidelity_index in _flatten(mapper.paths or []))
    if mapper.fidelity_model is not None:
        names.update(mapper.fidelity_model.gate_set.model_gate_set)
    table = sorted(names)
    return table, {name: n for n, name in enumerate(table)}


def _num_qubits(mapper: ExecutorDataMapper) -> int:
    """Return the qubit count every Pauli in the payload shares.

    Taken from the model where there is one, and otherwise from a fidelity index, since the qubit
    count is not recoverable from a Pauli's raw parts alone.
    """
    if mapper.fidelity_model is not None:
        return int(mapper.fidelity_model.gate_set.model_gate_set.num_qubits)
    for fidelity_index in _flatten(mapper.paths or []):
        return int(fidelity_index.pauli.num_qubits)
    return 0
