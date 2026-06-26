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

from qiskit.quantum_info import QubitSparsePauli

from qiskit_noise_learning.models import LogFidelitySpace
from qiskit_noise_learning.sequences import FidelityIndex


def test_gate_set_property(gate_set_1q):
    space = LogFidelitySpace(gate_set_1q)
    assert space.gate_set is gate_set_1q


def test_dim_1q(gate_set_1q):
    # P: 4^0 * 2^0 * 2^1 - 1 = 1
    # M: 4^0 * 2^1 * 2^1 - 1 = 3
    # L0: 4^1 * 2^0 * 2^0 - 1 = 3
    # L1: 4^1 * 2^0 * 2^0 - 1 = 3
    assert LogFidelitySpace(gate_set_1q).dim == 10


def test_dim_cz(gate_set_cz):
    # CZ: 4^2 - 1 = 15 ; P: 2^2 - 1 = 3 ; M: 2^2 * 2^2 - 1 = 15
    assert LogFidelitySpace(gate_set_cz).dim == 33


def test_contains_valid_fidelity(gate_set_1q):
    space = LogFidelitySpace(gate_set_1q)
    fidelity = FidelityIndex.from_gate(gate=gate_set_1q["L0"], pauli=QubitSparsePauli("X"))
    assert fidelity in space


def test_contains_rejects_trivial_identity(gate_set_1q):
    space = LogFidelitySpace(gate_set_1q)
    trivial = FidelityIndex.from_gate(gate=gate_set_1q["L0"], pauli=QubitSparsePauli.identity(1))
    assert trivial not in space


def test_contains_rejects_unknown_gate(gate_set_1q, gate_set_cz):
    space = LogFidelitySpace(gate_set_1q)
    # a CZ fidelity is not a member of the 1-qubit gate set's space
    cz_fidelity = FidelityIndex.from_gate(gate=gate_set_cz["CZ"], pauli=QubitSparsePauli("XZ"))
    assert cz_fidelity not in space


def test_contains_rejects_non_fidelity(gate_set_1q):
    space = LogFidelitySpace(gate_set_1q)
    assert "not a fidelity" not in space
    assert 42 not in space
