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

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit.library import CZGate
from qiskit.quantum_info import Clifford, QubitSparsePauli

from qiskit_noise_learning.gate_sets import ModelGate
from qiskit_noise_learning.sequences import FidelityIndex


def test_construction():
    """Test construction and attributes."""

    ident = Clifford(QuantumCircuit(2))
    gate = ModelGate("L0", [((0, 1), ident)], qubit_idxs=range(2), meas_idxs=[0])

    fidelity_index = FidelityIndex(
        gate=gate,
        pauli=QubitSparsePauli("XI"),
        in_bit_indices=frozenset([0]),
        out_bit_indices=frozenset(),
    )

    assert fidelity_index.gate == gate
    assert fidelity_index.pauli == QubitSparsePauli("XI")
    assert fidelity_index.in_bit_indices == frozenset([0])
    assert fidelity_index.out_bit_indices == frozenset()


def test_construction_validation():
    """Test construction and attributes."""

    ident = Clifford(QuantumCircuit(2))
    gate = ModelGate("L0", [((0, 1), ident)], qubit_idxs=range(2), meas_idxs=[0])

    with pytest.raises(ValueError, match="pauli.indices must lie"):
        FidelityIndex(
            gate=gate,
            pauli=QubitSparsePauli("IX"),
            in_bit_indices=frozenset([0]),
            out_bit_indices=frozenset(),
        )

    with pytest.raises(ValueError, match="in_bit_indices must be a subset"):
        FidelityIndex(
            gate=gate,
            pauli=QubitSparsePauli("XI"),
            in_bit_indices=frozenset([1]),
            out_bit_indices=frozenset(),
        )

    with pytest.raises(ValueError, match="out_bit_indices must be a subset"):
        FidelityIndex(
            gate=gate,
            pauli=QubitSparsePauli("XI"),
            in_bit_indices=frozenset(),
            out_bit_indices=frozenset([1]),
        )


def test_observable_indices_and_mask():
    """Test observable indices and mask properties."""
    qc = QuantumCircuit(1)
    qc.sx(0)
    gate = ModelGate("L0", cliffords=[((1,), qc)], qubit_idxs=range(2), meas_idxs=[0])

    # qubit 0 Z measured but not reset
    fid_idx = FidelityIndex(gate, QubitSparsePauli("ZI"), in_bit_indices=frozenset([0]))
    assert fid_idx.observable_indices == [0]
    assert np.array_equal(fid_idx.mask, np.array([True], np.bool_))

    # qubit 0 Z measured and reset to Z
    fid_idx = FidelityIndex(
        gate, QubitSparsePauli("ZI"), in_bit_indices=frozenset([0]), out_bit_indices=frozenset([0])
    )
    assert fid_idx.observable_indices == []
    assert np.array_equal(fid_idx.mask, np.array([False], np.bool_))

    # qubit 0 Z measured, qubit 1 reset to Z (set bits of out_bit_indices don't matter)
    gate = ModelGate(
        "L0", cliffords=[((1,), qc)], qubit_idxs=range(2), meas_idxs=[0], prep_idxs=[1]
    )
    fid_idx = FidelityIndex(
        gate, QubitSparsePauli("II"), in_bit_indices=frozenset([0]), out_bit_indices=frozenset([1])
    )
    assert fid_idx.observable_indices == [0]
    assert np.array_equal(fid_idx.mask, np.array([True], np.bool_))

    gate = ModelGate(
        "L0", cliffords=[((1,), qc)], qubit_idxs=range(2), meas_idxs=[0, 1], prep_idxs=[1]
    )
    fid_idx = FidelityIndex(
        gate, QubitSparsePauli("II"), in_bit_indices=frozenset([0]), out_bit_indices=frozenset([1])
    )
    assert fid_idx.observable_indices == [0, 1]
    assert np.array_equal(fid_idx.mask, np.array([True, True], np.bool_))

    fid_idx = FidelityIndex(
        gate,
        QubitSparsePauli("II"),
        in_bit_indices=frozenset([0, 1]),
        out_bit_indices=frozenset([1]),
    )
    assert np.array_equal(fid_idx.mask, np.array([True, False], np.bool_))

    # a case where the first entry in the list isn't being included
    fid_idx = FidelityIndex(
        gate,
        QubitSparsePauli("II"),
        in_bit_indices=frozenset([1]),
        out_bit_indices=frozenset(),
    )
    assert np.array_equal(fid_idx.mask, np.array([False, True], np.bool_))


def test_transition():
    """Test transition property."""
    qc = QuantumCircuit(2)
    qc.sx(1)
    gate = ModelGate("L0", [((0, 1), Clifford(qc))], qubit_idxs=range(2), meas_idxs=[0])

    fidelity_index = FidelityIndex(
        gate=gate,
        pauli=QubitSparsePauli("ZI"),
        in_bit_indices=frozenset([0]),
        out_bit_indices=frozenset(),
    )

    in_pauli, out_pauli = fidelity_index.transition

    assert in_pauli == QubitSparsePauli("YZ")
    assert out_pauli == QubitSparsePauli("ZI")


def test_sign_flip():
    """Test sign flip property."""

    qc = QuantumCircuit(2)
    qc.sx(1)
    gate = ModelGate("L0", cliffords=[(range(2), Clifford(qc))], meas_idxs=[0])

    fidelity_index = FidelityIndex(
        gate=gate,
        pauli=QubitSparsePauli("YI"),
        in_bit_indices=frozenset([0]),
        out_bit_indices=frozenset(),
    )

    in_pauli, out_pauli = fidelity_index.transition

    assert in_pauli == QubitSparsePauli("ZZ")
    assert out_pauli == QubitSparsePauli("YI")
    assert fidelity_index.sign_flip

    qc = QuantumCircuit(2)
    qc.x(1)
    gate = ModelGate("L0", cliffords=[(range(2), Clifford(qc))], meas_idxs=[0])

    fidelity_index = FidelityIndex(
        gate=gate,
        pauli=QubitSparsePauli("YI"),
        in_bit_indices=frozenset([0]),
        out_bit_indices=frozenset(),
    )

    in_pauli, out_pauli = fidelity_index.transition

    assert in_pauli == QubitSparsePauli("YZ")
    assert out_pauli == QubitSparsePauli("YI")
    assert fidelity_index.sign_flip


def test_from_transition():
    """Test from_transition constructor."""

    # preparation
    gate = ModelGate("P", [((0, 1), Clifford(QuantumCircuit(2)))], prep_idxs=range(2))
    fidelity_idx = FidelityIndex.from_transition(
        gate=gate, in_pauli=QubitSparsePauli("II"), out_pauli=QubitSparsePauli("IZ")
    )
    assert fidelity_idx == FidelityIndex(
        gate=gate,
        pauli=QubitSparsePauli("II"),
        in_bit_indices=frozenset(),
        out_bit_indices=frozenset([0]),
    )

    # measurement
    gate = ModelGate("M", [((0, 1), Clifford(QuantumCircuit(2)))], meas_idxs=range(2))
    fidelity_idx = FidelityIndex.from_transition(
        gate=gate, in_pauli=QubitSparsePauli("ZI"), out_pauli=QubitSparsePauli("IZ")
    )
    assert fidelity_idx == FidelityIndex(
        gate=gate,
        pauli=QubitSparsePauli("II"),
        in_bit_indices=frozenset([1]),
        out_bit_indices=frozenset([0]),
    )

    # pure unitary test
    gate = ModelGate("L0", [((0, 1), Clifford(CZGate()))])
    fidelity_idx = FidelityIndex.from_transition(
        gate=gate, in_pauli=QubitSparsePauli("IX"), out_pauli=QubitSparsePauli("ZX")
    )
    assert fidelity_idx == FidelityIndex(
        gate=gate,
        pauli=QubitSparsePauli("ZX"),
        in_bit_indices=frozenset([]),
        out_bit_indices=frozenset([]),
    )

    # unitary and reset: gate maps ZX -> IX, and we reset qubit 1 to I
    gate = ModelGate("L0", [((0, 1), Clifford(CZGate()))], prep_idxs=[1])
    fidelity_idx = FidelityIndex.from_transition(
        gate=gate, in_pauli=QubitSparsePauli("ZX"), out_pauli=QubitSparsePauli("IX")
    )
    assert fidelity_idx == FidelityIndex(
        gate=gate,
        pauli=QubitSparsePauli("IX"),
        in_bit_indices=frozenset([]),
        out_bit_indices=frozenset([]),
    )

    # unitary and reset: gate maps ZX -> IX, and we reset qubit 1 to Z
    gate = ModelGate("L0", [((0, 1), Clifford(CZGate()))], prep_idxs=[1])
    fidelity_idx = FidelityIndex.from_transition(
        gate=gate, in_pauli=QubitSparsePauli("ZX"), out_pauli=QubitSparsePauli("ZX")
    )
    assert fidelity_idx == FidelityIndex(
        gate=gate,
        pauli=QubitSparsePauli("IX"),
        in_bit_indices=frozenset([]),
        out_bit_indices=frozenset([1]),
    )

    # measure and reset
    gate = ModelGate("L0", [((0, 1, 2), Clifford(QuantumCircuit(3)))], meas_idxs=[1], prep_idxs=[2])
    fidelity_idx = FidelityIndex.from_transition(
        gate=gate, in_pauli=QubitSparsePauli("IZZ"), out_pauli=QubitSparsePauli("ZZZ")
    )
    assert fidelity_idx == FidelityIndex(
        gate=gate,
        pauli=QubitSparsePauli("IIZ"),
        in_bit_indices=frozenset([1]),
        out_bit_indices=frozenset([1, 2]),
    )

    # measure and reset different configuration
    gate = ModelGate("L0", [((0, 1, 2), Clifford(QuantumCircuit(3)))], meas_idxs=[1], prep_idxs=[2])
    fidelity_idx = FidelityIndex.from_transition(
        gate=gate, in_pauli=QubitSparsePauli("IIZ"), out_pauli=QubitSparsePauli("ZIZ")
    )
    assert fidelity_idx == FidelityIndex(
        gate=gate,
        pauli=QubitSparsePauli("IIZ"),
        in_bit_indices=frozenset([]),
        out_bit_indices=frozenset([2]),
    )


def test_from_transition_errors():
    # in indices out of bounds
    with pytest.raises(ValueError, match="in_pauli.indices is not contained in gate.qubit_idxs"):
        FidelityIndex.from_transition(
            gate=ModelGate("P", [((0, 1), Clifford(QuantumCircuit(2)))], prep_idxs=range(2)),
            in_pauli=QubitSparsePauli("ZII"),
            out_pauli=QubitSparsePauli("IZ"),
        )
    # out indices out of bounds
    with pytest.raises(ValueError, match="out_pauli.indices is not contained in gate.qubit_idxs"):
        FidelityIndex.from_transition(
            gate=ModelGate("P", [((0, 1), Clifford(QuantumCircuit(2)))], prep_idxs=range(2)),
            in_pauli=QubitSparsePauli("II"),
            out_pauli=QubitSparsePauli("ZIZ"),
        )

    # out_pauli restricted to measured and reset qubits has non-Z component
    with pytest.raises(ValueError, match="out_pauli restricted to measured and reset qubits must"):
        FidelityIndex.from_transition(
            gate=ModelGate("P", [((0, 1), Clifford(QuantumCircuit(2)))], prep_idxs=range(2)),
            in_pauli=QubitSparsePauli("II"),
            out_pauli=QubitSparsePauli("XI"),
        )

    # in_pauli restricted to measured qubits has non-Z component
    with pytest.raises(ValueError, match="in_pauli mapped by Clifford and restricted to measured"):
        FidelityIndex.from_transition(
            gate=ModelGate("P", [((0, 1), Clifford(QuantumCircuit(2)))], meas_idxs=range(2)),
            in_pauli=QubitSparsePauli("IX"),
            out_pauli=QubitSparsePauli("II"),
        )

    # non-identity component fed into unmeasured and reset qubit
    with pytest.raises(
        ValueError, match="in_pauli mapped by Clifford and restricted to unmeasured and reset"
    ):
        FidelityIndex.from_transition(
            gate=ModelGate("P", [((0, 1), Clifford(QuantumCircuit(2)))], prep_idxs=range(2)),
            in_pauli=QubitSparsePauli("IZ"),
            out_pauli=QubitSparsePauli("IZ"),
        )

    # consistency of unmeasured and unreset registers
    with pytest.raises(
        ValueError, match="out_pauli and in_pauli mapped by Clifford must agree on unmeasured"
    ):
        FidelityIndex.from_transition(
            gate=ModelGate("L0", [((0, 1), Clifford(CZGate()))]),
            in_pauli=QubitSparsePauli("IX"),
            out_pauli=QubitSparsePauli("IX"),
        )


def test_hash():
    gate = ModelGate("L0", [], qubit_idxs=range(2), meas_idxs=[0])

    fidelity_index = FidelityIndex(
        gate=gate,
        pauli=QubitSparsePauli("XI"),
        in_bit_indices=frozenset([0]),
        out_bit_indices=frozenset(),
    )

    assert isinstance(hash(fidelity_index), int)
