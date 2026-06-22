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
from qiskit_noise_learning.visualizations import fidelity_index_latex_str, path_latex_str


@pytest.fixture()
def gate_set():
    model_gate_set = ModelGateSet(2)
    ident_2q = Clifford(QuantumCircuit(2))
    model_gate_set.add_gate(ModelGate("P", [], prep_idxs=[0, 1], qubit_idxs=[0, 1]))
    model_gate_set.add_gate(ModelGate("M", [], meas_idxs=[0, 1], qubit_idxs=[0, 1]))
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


def test_transition_format(gate_set, fidelity_index):
    model = CompleteFidelityModel(gate_set)
    assert (
        fidelity_index_latex_str(fidelity_index, model, format="transition")
        == r"X_{0} \xrightarrow{\mathrm{CZ}} X_{0}"
    )


def test_generic_formula_format(gate_set, fidelity_index):
    model = CompleteFidelityModel(gate_set)
    assert (
        fidelity_index_latex_str(fidelity_index, model, format="formula")
        == r"f^{\mathrm{CZ}}(X_{0})"
    )


def test_invalid_format_raises(gate_set, fidelity_index):
    model = CompleteFidelityModel(gate_set)
    with pytest.raises(ValueError):
        fidelity_index_latex_str(fidelity_index, model, format="bad")


def test_path_transition_format(gate_set, fidelity_index):
    model = CompleteFidelityModel(gate_set)
    path = Path(
        start_fragment=[fidelity_index],
        repeatable_fragment=[fidelity_index, fidelity_index],
        end_fragment=[fidelity_index],
        depth=3,
    )
    arrow = r"X_{0} \xrightarrow{\mathrm{CZ}} X_{0}"
    rep = r"X_{0} \xrightarrow{\mathrm{CZ}} X_{0} \xrightarrow{\mathrm{CZ}} X_{0}"
    assert path_latex_str(path, model, format="transition") == (
        rf"{arrow} \rightarrow [{rep}]^{{3}} \rightarrow {arrow}"
    )


def test_path_repeatable_only(gate_set, fidelity_index):
    model = CompleteFidelityModel(gate_set)
    path = Path(
        start_fragment=[fidelity_index],
        repeatable_fragment=[fidelity_index],
        end_fragment=[fidelity_index],
        depth=3,
    )
    # Only the repeatable fragment, no surrounding brackets or depth exponent.
    assert (
        path_latex_str(path, model, repeatable_only=True)
        == r"X_{0} \xrightarrow{\mathrm{CZ}} X_{0}"
    )


def test_pauli_lindblad_formula_is_simplified(gate_set, fidelity_index):
    # Pauli-Lindblad models render the simplified f^{G}_{P} label rather than the generic formula.
    model = PauliLindbladModel.k_local(gate_set, k=2)
    assert (
        fidelity_index_latex_str(fidelity_index, model, format="formula")
        == r"f^{\mathrm{CZ}}_{X_{0}}"
    )


def test_composed_uses_contained_pauli_lindblad_formula(gate_set, fidelity_index):
    # A ComposedFidelityModel containing a PauliLindbladModel renders the simplified Pauli-Lindblad
    # label, not the generic one.
    pauli_lindblad_model = PauliLindbladModel.k_local(gate_set, k=2)
    composed = CompleteFidelityModel(gate_set) @ pauli_lindblad_model
    assert (
        fidelity_index_latex_str(fidelity_index, composed, format="formula")
        == r"f^{\mathrm{CZ}}_{X_{0}}"
    )
