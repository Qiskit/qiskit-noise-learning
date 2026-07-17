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

from qiskit_noise_learning.math import IndexedVector
from qiskit_noise_learning.models import IdentityFidelityModel, LogFidelitySpace
from qiskit_noise_learning.sequences import FidelityIndex


def test_gate_set(gate_set_1q):
    model = IdentityFidelityModel(gate_set_1q)
    assert model.gate_set is gate_set_1q


def test_spaces_are_the_same_fidelity_space(gate_set_1q):
    model = IdentityFidelityModel(gate_set_1q)
    assert isinstance(model.output_space, LogFidelitySpace)
    assert model.input_space is model.output_space


def test_rows_are_identity(gate_set_1q):
    model = IdentityFidelityModel(gate_set_1q)
    fidelity_x = FidelityIndex.from_gate(gate=gate_set_1q["L0"], pauli=QubitSparsePauli("X"))
    fidelity_y = FidelityIndex.from_gate(gate=gate_set_1q["L0"], pauli=QubitSparsePauli("Y"))

    matrix = model.rows([fidelity_x, fidelity_y])

    assert matrix[fidelity_x] == IndexedVector({fidelity_x: 1.0, fidelity_y: 0.0})
    assert matrix[fidelity_y] == IndexedVector({fidelity_x: 0.0, fidelity_y: 1.0})
