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

"""Round-trip tests for DataMapperModelV1 serialization."""

import numpy as np
import pytest
from qiskit.circuit.library import CZGate
from qiskit.quantum_info import Clifford, QubitSparsePauli, QubitSparsePauliList

from qiskit_noise_learning.circuit_generator import ExecutorDataMapper
from qiskit_noise_learning.gate_sets import ModelGate, ModelGateSet
from qiskit_noise_learning.models import PauliLindbladModel
from qiskit_noise_learning.sequences import (
    ApplyGate,
    FidelityIndex,
    InstructionPattern,
    InstructionSequence,
    PartialPauliPermutation,
    Path,
    PathPattern,
)
from qiskit_noise_learning.serialization import DataMapperModelV1

from .utils import assert_data_mappers_equal


@pytest.fixture
def model_gate_set():
    gate_set = ModelGateSet(2)
    gate_set.add_gate(ModelGate("CZ", [((0, 1), Clifford(CZGate()))]))
    gate_set.add_gate(ModelGate("P", qubit_idxs=range(2), prep_idxs=range(2)))
    gate_set.add_gate(ModelGate("M", qubit_idxs=range(2), meas_idxs=range(2)))
    return gate_set


def test_data_mapper_model_v1_round_trip(model_gate_set):
    """Test ExecutorDataMapper -> DataMapperModelV1 -> ExecutorDataMapper round-trip."""
    cz_gate = model_gate_set["CZ"].model_gate
    perm = PartialPauliPermutation(np.array([0, 3], dtype=np.int8))
    pattern = InstructionPattern(
        start_fragment=[ApplyGate(cz_gate)],
        repeatable_fragment=[perm, ApplyGate(cz_gate)],
        end_fragment=[],
    )
    sequences = [
        InstructionSequence(pattern=pattern, depth=2),
        InstructionSequence(pattern=pattern, depth=4),
    ]

    fi = FidelityIndex(gate=cz_gate, pauli=QubitSparsePauli("IX"))
    path = Path(
        pattern=PathPattern(start_fragment=[fi], repeatable_fragment=[fi], end_fragment=[]),
        depth=2,
    )

    generators = {
        "CZ": QubitSparsePauliList(["ZI", "IX", "XX"]),
        "P": QubitSparsePauliList(["XI", "IX"]),
        "M": QubitSparsePauliList(["XI", "IX"]),
    }
    fidelity_model = PauliLindbladModel(model_gate_set, generators)

    original = ExecutorDataMapper(
        item_sequence_indices=[[0, 1]],
        creg_names=[["meas0", "meas1"]],
        measurement_maps=[{"meas0": np.array([0], dtype=int), "meas1": np.array([1], dtype=int)}],
        instruction_sequences=sequences,
        num_randomizations=50,
        fidelity_model=fidelity_model,
        paths=[path],
    )

    v1_model = DataMapperModelV1.from_executor_data_mapper(original)
    restored = v1_model.to_executor_data_mapper()
    assert_data_mappers_equal(original, restored)


def test_data_mapper_model_v1_passthrough_round_trip(model_gate_set):
    """Test the full passthrough_data path: serialize -> dict -> deserialize."""
    cz_gate = model_gate_set["CZ"].model_gate
    pattern = InstructionPattern(
        start_fragment=[ApplyGate(cz_gate)],
        repeatable_fragment=[],
        end_fragment=[],
    )
    sequences = [InstructionSequence(pattern=pattern, depth=1)]

    fi = FidelityIndex(gate=cz_gate, pauli=QubitSparsePauli("ZI"))
    path = Path(
        pattern=PathPattern(start_fragment=[fi], repeatable_fragment=[], end_fragment=[]),
        depth=1,
    )

    generators = {
        "CZ": QubitSparsePauliList(["ZI", "IX"]),
        "P": QubitSparsePauliList(["XI"]),
        "M": QubitSparsePauliList(["IX"]),
    }
    fidelity_model = PauliLindbladModel(model_gate_set, generators)

    original = ExecutorDataMapper(
        item_sequence_indices=[[0]],
        creg_names=[["meas0"]],
        measurement_maps=[{"meas0": np.array([0, 1], dtype=int)}],
        instruction_sequences=sequences,
        num_randomizations=10,
        fidelity_model=fidelity_model,
        paths=[path],
    )

    passthrough_data = original.to_passthrough_data()
    restored = ExecutorDataMapper.from_passthrough_data(passthrough_data)
    assert_data_mappers_equal(original, restored)
