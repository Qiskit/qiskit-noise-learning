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
from qiskit_noise_learning.models import CompleteFidelityModel, PauliLindbladModel
from qiskit_noise_learning.sequences import FidelityIndex, Path


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


class TestFidelityIndexLatexStr:
    def test_transition_format(self, gate_set, fidelity_index):
        model = CompleteFidelityModel(gate_set)
        result = model.fidelity_index_latex_str(fidelity_index, format="transition")
        assert isinstance(result, str)

    def test_formula_format(self, gate_set, fidelity_index):
        model = CompleteFidelityModel(gate_set)
        result = model.fidelity_index_latex_str(fidelity_index, format="formula")
        assert isinstance(result, str)

    def test_invalid_format_raises(self, gate_set, fidelity_index):
        model = CompleteFidelityModel(gate_set)
        with pytest.raises(ValueError):
            model.fidelity_index_latex_str(fidelity_index, format="bad")


class TestPathLatexStr:
    def test_transition_format(self, gate_set, fidelity_index):
        model = CompleteFidelityModel(gate_set)
        path = Path(
            start_fragment=[fidelity_index],
            repeatable_fragment=[fidelity_index, fidelity_index],
            end_fragment=[fidelity_index],
            depth=3,
        )
        result = model.path_latex_str(path, format="transition")
        assert isinstance(result, str)

    def test_formula_format(self, gate_set, fidelity_index):
        model = CompleteFidelityModel(gate_set)
        path = Path(
            start_fragment=[],
            repeatable_fragment=[fidelity_index, fidelity_index],
            end_fragment=[],
            depth=5,
        )
        result = model.path_latex_str(path, format="formula")
        assert isinstance(result, str)

    def test_repeatable_only(self, gate_set, fidelity_index):
        model = CompleteFidelityModel(gate_set)
        path = Path(
            start_fragment=[fidelity_index],
            repeatable_fragment=[fidelity_index],
            end_fragment=[fidelity_index],
            depth=3,
        )
        result = model.path_latex_str(path, repeatable_only=True)
        assert isinstance(result, str)


@pytest.fixture()
def gate_set_with_spam():
    model_gate_set = ModelGateSet(2)
    ident_2q = Clifford(QuantumCircuit(2))
    model_gate_set.add_gate(ModelGate("P", [], prep_idxs=[0, 1], qubit_idxs=[0, 1]))
    model_gate_set.add_gate(ModelGate("M", [], meas_idxs=[0, 1], qubit_idxs=[0, 1]))
    model_gate_set.add_gate(
        ModelGate("CZ", [((0, 1), ident_2q)], qubit_idxs=[0, 1], latex_str=r"\mathrm{CZ}")
    )
    return model_gate_set


class TestPauliLindbladModelLatexStr:
    def test_formula_format(self, gate_set_with_spam):
        model = PauliLindbladModel.k_local(gate_set_with_spam, k=2)
        fi = FidelityIndex.from_gate(
            gate=gate_set_with_spam["CZ"],
            pauli=QubitSparsePauli.from_sparse_label(("X", [0]), num_qubits=2),
            in_bit_indices=frozenset(),
            out_bit_indices=frozenset(),
        )
        result = model.fidelity_index_latex_str(fi, format="formula")
        assert isinstance(result, str)

    def test_transition_format(self, gate_set_with_spam):
        model = PauliLindbladModel.k_local(gate_set_with_spam, k=2)
        fi = FidelityIndex.from_gate(
            gate=gate_set_with_spam["CZ"],
            pauli=QubitSparsePauli.from_sparse_label(("X", [0]), num_qubits=2),
            in_bit_indices=frozenset(),
            out_bit_indices=frozenset(),
        )
        result = model.fidelity_index_latex_str(fi, format="transition")
        assert isinstance(result, str)
