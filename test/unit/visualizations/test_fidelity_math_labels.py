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
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Clifford, QubitSparsePauli

from qiskit_noise_learning.gate_sets import ModelGate, ModelGateSet
from qiskit_noise_learning.sequences import FidelityIndex, Path
from qiskit_noise_learning.visualizations import fidelity_index_math_label, path_math_label


@pytest.fixture()
def gate_set():
    model_gate_set = ModelGateSet(2)
    ident_2q = Clifford(QuantumCircuit(2))
    model_gate_set.add_gate(
        ModelGate("CZ", [((0, 1), ident_2q)], qubit_idxs=[0, 1], latex_str=r"\mathrm{CZ}")
    )
    return model_gate_set


@pytest.fixture()
def fidelity_index(gate_set):
    return FidelityIndex.from_gate(
        gate=gate_set["CZ"],
        pauli=QubitSparsePauli.from_sparse_label(("X", [0]), num_qubits=2),
        in_bit_indices=frozenset(),
        out_bit_indices=frozenset(),
    )


class TestFidelityIndexMathLabel:
    def test_transition_format(self, gate_set, fidelity_index):
        result = fidelity_index_math_label(gate_set, fidelity_index, style="transition")
        assert isinstance(result, str)

    def test_formula_format(self, gate_set, fidelity_index):
        result = fidelity_index_math_label(gate_set, fidelity_index, style="formula")
        assert isinstance(result, str)

    def test_invalid_format_raises(self, gate_set, fidelity_index):
        with pytest.raises(ValueError):
            fidelity_index_math_label(gate_set, fidelity_index, style="bad")

    def test_qubit_labels_remap_subscript(self, gate_set, fidelity_index):
        # The X on qubit 0 renders as X_{i} once qubit 0 is relabeled to "i".
        labeled = fidelity_index_math_label(
            gate_set, fidelity_index, style="formula", qubit_labels={0: "i"}
        )
        assert "X_{i}" in labeled
        assert "X_{0}" not in labeled

    def test_qubit_labels_partial_falls_back_to_index(self, gate_set, fidelity_index):
        # An index absent from the map renders as its integer value.
        labeled = fidelity_index_math_label(
            gate_set, fidelity_index, style="formula", qubit_labels={5: "z"}
        )
        assert "X_{0}" in labeled


class TestPathMathLabel:
    def test_transition_format(self, gate_set, fidelity_index):
        path = Path(
            start_fragment=[fidelity_index],
            repeatable_fragment=[fidelity_index, fidelity_index],
            end_fragment=[fidelity_index],
            depth=3,
        )
        result = path_math_label(gate_set, path, style="transition")
        assert isinstance(result, str)

    def test_formula_format(self, gate_set, fidelity_index):
        path = Path(
            start_fragment=[],
            repeatable_fragment=[fidelity_index, fidelity_index],
            end_fragment=[],
            depth=5,
        )
        result = path_math_label(gate_set, path, style="formula")
        assert isinstance(result, str)

    def test_qubit_labels_remap_subscript(self, gate_set, fidelity_index):
        path = Path(
            start_fragment=[],
            repeatable_fragment=[fidelity_index],
            end_fragment=[],
            depth=5,
        )
        labeled = path_math_label(gate_set, path, style="formula", qubit_labels={0: "i"})
        assert "X_{i}" in labeled
        assert "X_{0}" not in labeled

    def test_repeatable_only(self, gate_set, fidelity_index):
        path = Path(
            start_fragment=[fidelity_index],
            repeatable_fragment=[fidelity_index],
            end_fragment=[fidelity_index],
            depth=3,
        )
        result = path_math_label(gate_set, path, repeatable_only=True)
        assert isinstance(result, str)


class TestNoiseSiteMathLabel:
    @pytest.mark.parametrize("site", ["before", "after"])
    def test_formula_format(self, gate_set, fidelity_index, site):
        result = fidelity_index_math_label(
            gate_set, fidelity_index, style="formula", noise_site={"CZ": site}
        )
        assert isinstance(result, str)

    def test_transition_format(self, gate_set, fidelity_index):
        result = fidelity_index_math_label(
            gate_set, fidelity_index, style="transition", noise_site={"CZ": "before"}
        )
        assert isinstance(result, str)
