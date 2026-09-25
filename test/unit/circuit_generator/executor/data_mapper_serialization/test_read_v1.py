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
"""Version 1 payloads that must always deserialize to the same experiment.

**Every payload in this module was written by hand and must never be regenerated.** Their whole
purpose is to be independent of the writer: a payload produced by the current writer can only show
that the reader and writer still agree with each other, whereas one written by hand also fails when
the writer is wrong.

Each is paired with a description of the experiment it holds, written in terms of gates and
observables rather than of arrays or of classes. **That description, not the arrays, is what a
future reader should work from.** If the package's classes change so that the expected mapper below
no longer expresses the same experiment, translate the description into the new classes and leave
the payload untouched.

Two things to know when reading the arrays. Pauli terms are coded ``X`` as 2, ``Y`` as 3 and ``Z``
as 1, and the identity stores no term at all. Ragged arrays are one concatenation plus a length per
row, so ``[0, 1]`` with lengths ``[1, 1]`` is two rows of one entry, not one row of two.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Clifford, QubitSparsePauli, QubitSparsePauliList

from qiskit_noise_learning.circuit_generator import ExecutorDataMapper
from qiskit_noise_learning.circuit_generator.executor.data_mapper_serialization import v1
from qiskit_noise_learning.gate_sets import ModelGate, ModelGateSet
from qiskit_noise_learning.models import PauliLindbladModel
from qiskit_noise_learning.sequences import (
    ApplyGate,
    FidelityIndex,
    InstructionSequence,
    PartialPauliPermutation,
    Path,
)

from .conftest import as_comparable

U32 = np.uint32


def u32(*values):
    """An index or length array, the dtype every one of them is written with."""
    return np.array(values, dtype=U32)


@dataclass(frozen=True)
class FrozenPayload:
    """A hand-written payload, kept together with the experiment it describes."""

    description: str
    """What experiment this payload holds, in terms independent of the format and of the classes."""

    payload: dict[str, Any] = field(default_factory=dict)
    """The payload itself, exactly as version 1 writes it."""


# --------------------------------------------------------------------------------------------------

NO_MODEL = FrozenPayload(
    description="""
    One unbound instruction sequence and nothing else.

    The sequence applies a preparation P, then repeats nothing, then applies a measurement M. It has
    no fragment depth, so it stands for the sequence at any depth. One program item holds it, and
    measures one qubit into a register named "meas".

    There is no model, so no gate set and no qubit count; there are no paths and no relations.
    """,
    payload={
        "version": 1,
        # No model and no paths, so nothing in the payload fixes a qubit count.
        "num_qubits": 0,
        "num_randomizations": 2,
        # Sorted, so M comes before P. Every "gates" array below indexes into this.
        "gate_names": ["M", "P"],
        "layout": {
            "item_sequence_idxs": u32(0),  # one item, holding sequence 0
            "item_sequence_lengths": u32(1),
            "item_creg_names": [["meas"]],
            "item_clbit_qubit_idxs": u32(0),  # that register's one bit measures qubit 0
            "item_clbit_lengths": u32(1),
        },
        "sequences": {
            "structure": {
                "start_lengths": u32(1),  # P
                "repeatable_lengths": u32(0),  # nothing repeats
                "end_lengths": u32(1),  # M
                "fragment_depths": np.array([-1], dtype=np.int32),  # unbound
            },
            # The two instructions in order: distinct, so one entry each.
            "instruction_idxs": u32(0, 1),
            "instructions": {
                "kinds": np.array([0, 0], dtype=np.uint8),  # both apply a gate
                "gates": u32(1, 0),  # gate_names[1] = P, then gate_names[0] = M
                "permutations": np.array([], dtype=np.int8),
                "permutation_lengths": u32(),
            },
        },
        "paths": None,
        "relations": None,
        "model": None,
    },
)


PATHS_AND_RELATIONS = FrozenPayload(
    description="""
    The sequence of NO_MODEL, bound at fragment depth 2, together with one path over one qubit.

    The path's three fragments each hold the same fidelity index: the measurement M observing Z on
    qubit 0, taking Z in and the identity out, with no sign flip. Because all three are the same,
    the payload stores that fidelity index once and points at it three times.

    One relation records that this path is traversed by this sequence. There is still no model, so
    the qubit count comes from the Paulis: one.
    """,
    payload={
        "version": 1,
        "num_qubits": 1,
        "num_randomizations": 2,
        "gate_names": ["M", "P"],
        "layout": {
            "item_sequence_idxs": u32(0),
            "item_sequence_lengths": u32(1),
            "item_creg_names": [["meas"]],
            "item_clbit_qubit_idxs": u32(0),
            "item_clbit_lengths": u32(1),
        },
        "sequences": {
            "structure": {
                "start_lengths": u32(1),
                "repeatable_lengths": u32(0),
                "end_lengths": u32(1),
                "fragment_depths": np.array([2], dtype=np.int32),  # bound, unlike NO_MODEL
            },
            "instruction_idxs": u32(0, 1),
            "instructions": {
                "kinds": np.array([0, 0], dtype=np.uint8),
                "gates": u32(1, 0),
                "permutations": np.array([], dtype=np.int8),
                "permutation_lengths": u32(),
            },
        },
        "paths": {
            "structure": {
                "start_lengths": u32(1),
                "repeatable_lengths": u32(1),
                "end_lengths": u32(1),
                "fragment_depths": np.array([2], dtype=np.int32),
            },
            # Three occurrences, all of the one stored fidelity index.
            "fidelity_idxs": u32(0, 0, 0),
            "fidelity_indices": {
                "gates": u32(0),  # gate_names[0] = M
                "sign_flips": np.array([False]),
                "pauli_terms": np.array([1], dtype=np.uint8),  # Z
                "pauli_idxs": u32(0),  # on qubit 0
                "pauli_lengths": u32(1),
                "input_terms": np.array([1], dtype=np.uint8),  # Z in
                "input_idxs": u32(0),
                "input_lengths": u32(1),
                "output_terms": np.array([], dtype=np.uint8),  # identity out, so no terms
                "output_idxs": u32(),
                "output_lengths": u32(0),
                "in_z_idxs": u32(0),  # Z arrives on qubit 0
                "in_z_lengths": u32(1),
                "out_z_idxs": u32(),  # and none leaves
                "out_z_lengths": u32(0),
                "meas_idxs": u32(0),  # M measures qubit 0
                "meas_lengths": u32(1),
            },
        },
        "relations": {
            "path_idxs": u32(0),
            "sequence_idxs": u32(0),
        },
        "model": None,
    },
)


WITH_MODEL = FrozenPayload(
    description="""
    The bound sequence of PATHS_AND_RELATIONS, with a Pauli-Lindblad model over two qubits.

    The gate set has three gates. P prepares both qubits and M measures both, neither carrying a
    Clifford. U is a unitary acting on both, built from two Cliffords applied in turn: the identity
    on qubit 0, then the identity on both. The device is fully connected, so its coupling map has an
    edge each way.

    The noise generators differ per gate, which is what makes the payload's per-gate splitting
    visible: M has one, X on qubit 0; P has two, X on qubit 0 and X on qubit 1; U has two, Z on
    qubit 0 and Z on both qubits at once. That last one is the only generator spanning two qubits.
    The preparation's noise is modelled after it and the other two before, which are the defaults
    for a pure preparation, a pure measurement and a unitary.

    One program item measures both qubits into a register named "meas". There are no paths.
    """,
    payload={
        "version": 1,
        "num_qubits": 2,
        "num_randomizations": 2,
        # The sequence names P and M; the gate set adds U.
        "gate_names": ["M", "P", "U"],
        "layout": {
            "item_sequence_idxs": u32(0),
            "item_sequence_lengths": u32(1),
            "item_creg_names": [["meas"]],
            "item_clbit_qubit_idxs": u32(0, 1),  # two bits, measuring qubits 0 and 1
            "item_clbit_lengths": u32(2),
        },
        "sequences": {
            "structure": {
                "start_lengths": u32(1),
                "repeatable_lengths": u32(0),
                "end_lengths": u32(1),
                "fragment_depths": np.array([2], dtype=np.int32),
            },
            "instruction_idxs": u32(0, 1),
            "instructions": {
                "kinds": np.array([0, 0], dtype=np.uint8),
                "gates": u32(1, 0),  # gate_names[1] = P, then gate_names[0] = M
                "permutations": np.array([], dtype=np.int8),
                "permutation_lengths": u32(),
            },
        },
        "paths": None,
        "relations": None,
        "model": {
            "kind": "pauli_lindblad",
            "num_qubits": 2,
            "qubit_subset": u32(0, 1),
            # Fully connected, so an edge each way, sorted: (0, 1) then (1, 0).
            "coupling_map_controls": u32(0, 1),
            "coupling_map_targets": u32(1, 0),
            # Sorted. Every gate array below is in this order: M, then P, then U.
            "gate_names": ["M", "P", "U"],
            "gate_qubit_idxs": u32(0, 1, 0, 1, 0, 1),  # all three act on both qubits
            "gate_qubit_lengths": u32(2, 2, 2),
            "gate_meas_idxs": u32(0, 1),  # only M measures, and it measures both
            "gate_meas_lengths": u32(2, 0, 0),
            "gate_prep_idxs": u32(0, 1),  # only P prepares, and it prepares both
            "gate_prep_lengths": u32(0, 2, 0),
            "clifford_counts": u32(0, 0, 2),  # only U carries Cliffords, and it carries two
            "clifford_qubit_idxs": u32(0, 0, 1),  # the first acts on qubit 0, the second on both
            "clifford_qubit_lengths": u32(1, 2),
            "clifford_num_qubits": u32(1, 2),
            # Both tableaus are identities, flattened row by row and concatenated: a 2 by 3 for the
            # one-qubit Clifford, then a 4 by 5 for the two-qubit one, whose last column is a phase.
            "clifford_tableaus": np.array(
                [1, 0, 0, 0, 1, 0] + [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
                dtype=bool,
            ),
            "generator_gate_names": ["M", "P", "U"],
            "generator_counts": u32(1, 2, 2),  # one for M, two each for P and U
            # M: X on 0. P: X on 0, then X on 1. U: Z on 0, then Z on 0 and Z on 1.
            "generator_terms": np.array([2, 2, 2, 1, 1, 1], dtype=np.uint8),
            "generator_idxs": u32(0, 0, 1, 0, 0, 1),
            "generator_lengths": u32(1, 1, 1, 1, 2),  # the last generator spans two qubits
            "noise_site": {"P": "after", "M": "before", "U": "before"},
        },
    },
)


PAULI_PERMUTATIONS = FrozenPayload(
    description="""
    One bound instruction sequence mixing gate applications with Pauli permutations, over two
    qubits.

    It applies the preparation P and then a permutation; repeats a second, different permutation;
    and ends by applying the first permutation again before the measurement M. That repeat is the
    point: the payload stores each distinct instruction once, pointing at the first twice.

    One program item measures both qubits into a register named "meas". There is no model and there
    are no paths, so nothing in the payload fixes a qubit count, even though the permutations are
    over two qubits.
    """,
    payload={
        "version": 1,
        # Neither a model nor a path, so no qubit count is recorded.
        "num_qubits": 0,
        "num_randomizations": 2,
        "gate_names": ["M", "P"],
        "layout": {
            "item_sequence_idxs": u32(0),
            "item_sequence_lengths": u32(1),
            "item_creg_names": [["meas"]],
            "item_clbit_qubit_idxs": u32(0, 1),
            "item_clbit_lengths": u32(2),
        },
        "sequences": {
            "structure": {
                "start_lengths": u32(2),  # P, then a permutation
                "repeatable_lengths": u32(1),  # a second permutation
                "end_lengths": u32(2),  # the first permutation again, then M
                "fragment_depths": np.array([1], dtype=np.int32),
            },
            # Five instructions in order, drawn from the four distinct ones below. Entry 1 recurs,
            # which is what shows they are stored once each.
            "instruction_idxs": u32(0, 1, 2, 1, 3),
            "instructions": {
                # Apply a gate, permute, permute, apply a gate.
                "kinds": np.array([0, 1, 1, 0], dtype=np.uint8),
                # Only the gate applications name a gate: gate_names[1] = P and gate_names[0] = M.
                # A permutation carries no gate, and is written as zero.
                "gates": u32(1, 0, 0, 0),
                # The two permutations, concatenated: [0, 3] then [1, 2].
                "permutations": np.array([0, 3, 1, 2], dtype=np.int8),
                "permutation_lengths": u32(2, 2),
            },
        },
        "paths": None,
        "relations": None,
        "model": None,
    },
)


# --------------------------------------------------------------------------------------------------
# The experiments the payloads above hold, built with the classes of the moment
# --------------------------------------------------------------------------------------------------


def _spam_sequence(fragment_depth):
    """A preparation, then nothing repeating, then a measurement."""
    return InstructionSequence(
        start_fragment=[ApplyGate("P")],
        repeatable_fragment=[],
        end_fragment=[ApplyGate("M")],
        fragment_depth=fragment_depth,
    )


def _layout():
    """One program item, measuring one qubit into one register."""
    return {
        "item_sequence_indices": [[0]],
        "item_creg_names": [["meas"]],
        "item_clbit_qubit_idxs": [{"meas": np.array([0])}],
        "num_randomizations": 2,
    }


def _expected_no_model():
    """The experiment NO_MODEL describes."""
    return ExecutorDataMapper(**_layout(), instruction_sequences=[_spam_sequence(None)])


def _expected_paths_and_relations():
    """The experiment PATHS_AND_RELATIONS describes."""
    measure = FidelityIndex(
        gate_name="M",
        pauli=QubitSparsePauli.from_label("Z"),
        in_z_idxs=frozenset({0}),
        out_z_idxs=frozenset(),
        input_pauli=QubitSparsePauli.from_label("Z"),
        output_pauli=QubitSparsePauli.from_label("I"),
        sign_flip=False,
        meas_idxs=frozenset({0}),
    )
    return ExecutorDataMapper(
        **_layout(),
        instruction_sequences=[_spam_sequence(2)],
        paths=[
            Path(
                start_fragment=[measure],
                repeatable_fragment=[measure],
                end_fragment=[measure],
                fragment_depth=2,
            )
        ],
        relations={(0, 0)},
    )


def _two_qubit_gate_set():
    """The gate set WITH_MODEL describes: a preparation, a measurement, and a unitary."""
    gate_set = ModelGateSet(2)
    gate_set.add_gate(ModelGate("P", qubit_idxs=[0, 1], prep_idxs=[0, 1]))
    gate_set.add_gate(ModelGate("M", qubit_idxs=[0, 1], meas_idxs=[0, 1]))
    gate_set.add_gate(
        ModelGate(
            "U",
            cliffords=[
                ((0,), Clifford(QuantumCircuit(1))),
                ((0, 1), Clifford(QuantumCircuit(2))),
            ],
            qubit_idxs=[0, 1],
        )
    )
    return gate_set


def _expected_pauli_permutations():
    """The experiment PAULI_PERMUTATIONS describes."""
    first = PartialPauliPermutation(np.array([0, 3], dtype=np.int8))
    second = PartialPauliPermutation(np.array([1, 2], dtype=np.int8))
    return ExecutorDataMapper(
        item_sequence_indices=[[0]],
        item_creg_names=[["meas"]],
        item_clbit_qubit_idxs=[{"meas": np.array([0, 1])}],
        num_randomizations=2,
        instruction_sequences=[
            InstructionSequence(
                start_fragment=[ApplyGate("P"), first],
                repeatable_fragment=[second],
                end_fragment=[first, ApplyGate("M")],
                fragment_depth=1,
            )
        ],
    )


def _expected_with_model():
    """The experiment WITH_MODEL describes."""
    return ExecutorDataMapper(
        item_sequence_indices=[[0]],
        item_creg_names=[["meas"]],
        item_clbit_qubit_idxs=[{"meas": np.array([0, 1])}],
        num_randomizations=2,
        instruction_sequences=[_spam_sequence(2)],
        fidelity_model=PauliLindbladModel(
            _two_qubit_gate_set(),
            generators={
                "M": QubitSparsePauliList.from_list(["IX"]),
                "P": QubitSparsePauliList.from_list(["IX", "XI"]),
                "U": QubitSparsePauliList.from_list(["IZ", "ZZ"]),
            },
        ),
    )


FROZEN = {
    "no_model": (NO_MODEL, _expected_no_model),
    "paths_and_relations": (PATHS_AND_RELATIONS, _expected_paths_and_relations),
    "pauli_permutations": (PAULI_PERMUTATIONS, _expected_pauli_permutations),
    "with_model": (WITH_MODEL, _expected_with_model),
}
"""Each frozen payload against the experiment it is meant to deserialize to."""


@pytest.mark.parametrize("name", sorted(FROZEN))
def test_read_gives_the_described_experiment(name):
    """Reading a frozen payload gives the experiment its description names."""
    frozen, expected = FROZEN[name]
    assert as_comparable(v1.read(frozen.payload)) == as_comparable(expected())
