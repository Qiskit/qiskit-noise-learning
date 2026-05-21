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

"""Round-trip tests for serialization converters."""

import numpy as np
import pytest
from qiskit.circuit.library import CZGate
from qiskit.quantum_info import Clifford, QubitSparsePauli, QubitSparsePauliList

from qiskit_noise_learning.gate_sets import ModelGate, ModelGateSet
from qiskit_noise_learning.models import PauliLindbladModel
from qiskit_noise_learning.sequences import (
    ApplyGate,
    FidelityIndex,
    InstructionSequence,
    PartialPauliPermutation,
    Path,
)
from qiskit_noise_learning.serialization import (
    deserialize_fidelity_index,
    deserialize_gate_set,
    deserialize_instruction,
    deserialize_instruction_sequence,
    deserialize_path,
    deserialize_pauli_lindblad_model,
    deserialize_qubit_sparse_pauli,
    deserialize_qubit_sparse_pauli_list,
    serialize_fidelity_index,
    serialize_gate_set,
    serialize_instruction,
    serialize_instruction_sequence,
    serialize_path,
    serialize_pauli_lindblad_model,
    serialize_qubit_sparse_pauli,
    serialize_qubit_sparse_pauli_list,
)

from .utils import assert_gate_sets_equal, assert_pauli_lindblad_models_equal


@pytest.fixture
def model_gate_set():
    gate_set = ModelGateSet(2)
    gate_set.add_gate(ModelGate("CZ", [((0, 1), Clifford(CZGate()))]))
    gate_set.add_gate(ModelGate("P", qubit_idxs=range(2), prep_idxs=range(2)))
    gate_set.add_gate(ModelGate("M", qubit_idxs=range(2), meas_idxs=range(2)))
    return gate_set


def test_qubit_sparse_pauli():
    original = QubitSparsePauli("IX")
    schema = serialize_qubit_sparse_pauli(original)
    restored = deserialize_qubit_sparse_pauli(schema)
    assert original == restored


def test_qubit_sparse_pauli_list():
    original = QubitSparsePauliList(["ZI", "IX", "XX"])
    schema = serialize_qubit_sparse_pauli_list(original)
    restored = deserialize_qubit_sparse_pauli_list(schema)
    assert len(original) == len(restored)
    for p1, p2 in zip(original, restored):
        assert p1 == p2


def test_gate_set(model_gate_set):
    schema = serialize_gate_set(model_gate_set)
    restored = deserialize_gate_set(schema)
    assert_gate_sets_equal(model_gate_set, restored)


def test_instruction_apply_gate(model_gate_set):
    original = ApplyGate(model_gate_set["CZ"].model_gate)
    schema = serialize_instruction(original)
    restored = deserialize_instruction(schema, model_gate_set)
    assert original == restored


def test_instruction_partial_pauli_permutation(model_gate_set):
    original = PartialPauliPermutation(np.array([0, 3], dtype=np.int8))
    schema = serialize_instruction(original)
    restored = deserialize_instruction(schema, model_gate_set)
    assert original == restored


def test_instruction_sequence(model_gate_set):
    cz_gate = model_gate_set["CZ"].model_gate
    perm = PartialPauliPermutation(np.array([0, 3], dtype=np.int8))
    original = InstructionSequence(
        start_fragment=[ApplyGate(cz_gate)],
        repeatable_fragment=[perm, ApplyGate(cz_gate)],
        end_fragment=[],
        depth=3,
    )
    schema = serialize_instruction_sequence(original)
    restored = deserialize_instruction_sequence(schema, model_gate_set)
    assert original == restored


def test_fidelity_index(model_gate_set):
    cz_gate = model_gate_set["CZ"].model_gate
    pauli = QubitSparsePauli("IX")
    original = FidelityIndex(gate=cz_gate, pauli=pauli)
    schema = serialize_fidelity_index(original)
    restored = deserialize_fidelity_index(schema, model_gate_set)
    assert original == restored


def test_path(model_gate_set):
    cz_gate = model_gate_set["CZ"].model_gate
    fi1 = FidelityIndex(gate=cz_gate, pauli=QubitSparsePauli("IX"))
    fi2 = FidelityIndex(gate=cz_gate, pauli=QubitSparsePauli("ZI"))
    original = Path(
        start_fragment=[fi1],
        repeatable_fragment=[fi2],
        end_fragment=[fi1],
        depth=2,
    )
    schema = serialize_path(original)
    restored = deserialize_path(schema, model_gate_set)
    assert original == restored


def test_pauli_lindblad_model(model_gate_set):
    generators = {
        "CZ": QubitSparsePauliList(["ZI", "IX", "XX"]),
        "P": QubitSparsePauliList(["XI", "IX"]),
        "M": QubitSparsePauliList(["XI", "IX"]),
    }
    original = PauliLindbladModel(model_gate_set, generators)
    schema = serialize_pauli_lindblad_model(original)
    restored = deserialize_pauli_lindblad_model(schema, model_gate_set)
    assert_pauli_lindblad_models_equal(original, restored)
