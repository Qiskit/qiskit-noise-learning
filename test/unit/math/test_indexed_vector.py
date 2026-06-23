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

import pytest
from qiskit.circuit.library import CZGate
from qiskit.quantum_info import Clifford, QubitSparsePauli

from qiskit_noise_learning.gate_sets import ModelGate
from qiskit_noise_learning.math import IndexedVector
from qiskit_noise_learning.sequences import FidelityIndex


@pytest.fixture()
def fidelity_list():
    gate = ModelGate("L0", [((0, 1), Clifford(CZGate()))])
    return [
        FidelityIndex.from_gate(
            gate=gate,
            pauli=QubitSparsePauli("ZX"),
            in_bit_indices=frozenset(),
            out_bit_indices=frozenset(),
        ),
        FidelityIndex.from_gate(
            gate=gate,
            pauli=QubitSparsePauli("XX"),
            in_bit_indices=frozenset(),
            out_bit_indices=frozenset(),
        ),
    ]


def test_construction(fidelity_list):
    dict_version = {k: idx for idx, k in enumerate(fidelity_list)}
    indexed_vector = IndexedVector(dict_version)
    assert indexed_vector == dict_version


def test_add(fidelity_list):
    # overlap
    indexed_vector = IndexedVector({fidelity_list[0]: 1.0, fidelity_list[1]: 2.0})
    indexed_vector2 = IndexedVector({fidelity_list[0]: 1.0})
    assert indexed_vector + indexed_vector2 == IndexedVector(
        {fidelity_list[0]: 2.0, fidelity_list[1]: 2.0}
    )

    # no overlap
    gate = ModelGate("L0", [((0, 1), Clifford(CZGate()))])
    new_index = FidelityIndex.from_gate(
        gate=gate,
        pauli=QubitSparsePauli("ZZ"),
        in_bit_indices=frozenset(),
        out_bit_indices=frozenset(),
    )
    indexed_vector3 = IndexedVector({new_index: 3.0})
    assert indexed_vector + indexed_vector3 == IndexedVector(
        {fidelity_list[0]: 1.0, fidelity_list[1]: 2.0, new_index: 3.0}
    )
    indexed_vector += indexed_vector3
    assert indexed_vector == IndexedVector(
        {fidelity_list[0]: 1.0, fidelity_list[1]: 2.0, new_index: 3.0}
    )


def test_mul(fidelity_list):
    indexed_vector = IndexedVector({fidelity_list[0]: 1.0, fidelity_list[1]: 2.0})
    assert 1.5 * indexed_vector == IndexedVector({fidelity_list[0]: 1.5, fidelity_list[1]: 3.0})
