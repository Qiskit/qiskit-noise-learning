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
from qiskit import QuantumCircuit
from qiskit.circuit.library import XGate
from qiskit.quantum_info import Clifford, QubitSparsePauli

from qiskit_noise_learning.gate_sets import ModelGate, ModelGateSet
from qiskit_noise_learning.math import IndexedVector
from qiskit_noise_learning.models import CompleteFidelityModel
from qiskit_noise_learning.sequences import FidelityIndex, Path


@pytest.fixture()
def gate_set_1q():
    model_gate_set = ModelGateSet(1)
    ident = Clifford(QuantumCircuit(1))
    model_gate_set.add_gate(ModelGate("P", [((0,), ident)], prep_idxs=range(1)))
    model_gate_set.add_gate(ModelGate("M", [((0,), ident)], meas_idxs=range(1)))
    # Clifford maps X -> -Y, Y -> Z, Z -> -X
    model_gate_set.add_gate(
        ModelGate("L0", [((0,), Clifford([[True, True, True], [True, False, True]]))])
    )
    model_gate_set.add_gate(ModelGate("L1", [((0,), Clifford(XGate()))]))
    return model_gate_set


def test_row_from_fidelity(gate_set_1q):
    complete_model = CompleteFidelityModel(gate_set_1q)
    fidelity = FidelityIndex.from_gate(
        gate=gate_set_1q["L0"],
        pauli=QubitSparsePauli("X"),
        in_bit_indices=frozenset(),
        out_bit_indices=frozenset(),
    )
    assert complete_model.row_from_fidelity(fidelity) == IndexedVector({fidelity: 1.0})


def test_row_from_unmixed_fidelity(gate_set_1q):
    """Should be same result as mixed version."""
    complete_model = CompleteFidelityModel(gate_set_1q)
    fidelity = FidelityIndex.from_gate(
        gate=gate_set_1q["L0"],
        pauli=QubitSparsePauli("X"),
        in_bit_indices=frozenset(),
        out_bit_indices=frozenset(),
    )
    assert complete_model.row_from_unmixed_fidelity(fidelity) == IndexedVector({fidelity: 1.0})


def test_row_from_unbound_path(gate_set_1q):
    """Test row_from_path with an unbound path returns only the repeatable fragment row."""
    complete_model = CompleteFidelityModel(gate_set_1q)
    fidelityX = FidelityIndex.from_gate(
        gate=gate_set_1q["L0"],
        pauli=QubitSparsePauli("X"),
        in_bit_indices=frozenset(),
        out_bit_indices=frozenset(),
    )
    fidelityY = FidelityIndex.from_gate(
        gate=gate_set_1q["L0"],
        pauli=QubitSparsePauli("Y"),
        in_bit_indices=frozenset(),
        out_bit_indices=frozenset(),
    )

    unbound_path = Path(
        start_fragment=[fidelityX],
        repeatable_fragment=[fidelityX, fidelityX, fidelityY],
        end_fragment=[fidelityY],
    )
    assert complete_model.row_from_path(unbound_path) == IndexedVector(
        {fidelityX: 2.0, fidelityY: 1.0}
    )


def test_row_from_bound_path(gate_set_1q):
    """Test row_from_path with a bound path returns the full row scaled by depth."""
    complete_model = CompleteFidelityModel(gate_set_1q)
    fidelityX = FidelityIndex.from_gate(
        gate=gate_set_1q["L0"],
        pauli=QubitSparsePauli("X"),
        in_bit_indices=frozenset(),
        out_bit_indices=frozenset(),
    )
    fidelityY = FidelityIndex.from_gate(
        gate=gate_set_1q["L0"],
        pauli=QubitSparsePauli("Y"),
        in_bit_indices=frozenset(),
        out_bit_indices=frozenset(),
    )

    path = Path(
        start_fragment=[fidelityX],
        repeatable_fragment=[fidelityX, fidelityX, fidelityY],
        end_fragment=[fidelityY],
        depth=5,
    )
    assert complete_model.row_from_path(path) == IndexedVector({fidelityX: 11.0, fidelityY: 6.0})
