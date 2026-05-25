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

"""Shared assertion helpers for serialization tests."""

import numpy as np

from qiskit_noise_learning.circuit_generator import ExecutorDataMapper
from qiskit_noise_learning.gate_sets import ModelGateSet
from qiskit_noise_learning.models import PauliLindbladModel


def assert_gate_sets_equal(gs1: ModelGateSet, gs2: ModelGateSet):
    assert gs1.num_qubits == gs2.num_qubits
    assert sorted(gs1.qubit_subset) == sorted(gs2.qubit_subset)
    assert set(gs1.coupling_map.get_edges()) == set(gs2.coupling_map.get_edges())
    assert set(gs1.keys()) == set(gs2.keys())
    for name in gs1:
        assert gs1[name].model_gate == gs2[name].model_gate


def assert_pauli_lindblad_models_equal(m1: PauliLindbladModel, m2: PauliLindbladModel):
    assert_gate_sets_equal(m1.gate_set, m2.gate_set)
    assert m1.noise_site == m2.noise_site
    assert set(m1.generators.keys()) == set(m2.generators.keys())
    for name in m1.generators:
        list1 = m1.generators[name]
        list2 = m2.generators[name]
        assert len(list1) == len(list2)
        for p1, p2 in zip(list1, list2):
            assert p1 == p2


def assert_data_mappers_equal(dm1: ExecutorDataMapper, dm2: ExecutorDataMapper):
    assert dm1.item_sequence_indices == dm2.item_sequence_indices
    assert dm1.creg_names == dm2.creg_names
    assert dm1.num_randomizations == dm2.num_randomizations

    assert len(dm1.measurement_maps) == len(dm2.measurement_maps)
    for m1, m2 in zip(dm1.measurement_maps, dm2.measurement_maps):
        assert set(m1.keys()) == set(m2.keys())
        for k in m1:
            np.testing.assert_array_equal(m1[k], m2[k])

    assert dm1.instruction_sequences == dm2.instruction_sequences
    assert dm1.paths == dm2.paths
    assert_pauli_lindblad_models_equal(dm1.fidelity_model, dm2.fidelity_model)
