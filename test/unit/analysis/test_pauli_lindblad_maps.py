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
from qiskit.circuit.library import CZGate
from qiskit.quantum_info import Clifford, QubitSparsePauliList

from qiskit_noise_learning.analysis import to_pauli_lindblad_maps
from qiskit_noise_learning.data import ModelData
from qiskit_noise_learning.gate_sets import ModelGate, ModelGateSet
from qiskit_noise_learning.math import IndexedVector, LinearMap
from qiskit_noise_learning.models import CompleteFidelityModel, PauliLindbladModel


class _IdentityMap(LinearMap):
    """Identity map on a given parameter space."""

    def __init__(self, space):
        super().__init__(input_space=space, output_space=space)

    def row(self, output_index):
        return IndexedVector({output_index: 1.0})


@pytest.fixture()
def gate_set():
    model_gate_set = ModelGateSet(2)
    model_gate_set.add_gate(ModelGate("CZ", [((0, 1), Clifford(CZGate()))]))
    model_gate_set.add_gate(ModelGate("P", qubit_idxs=range(2), prep_idxs=range(2)))
    model_gate_set.add_gate(ModelGate("M", qubit_idxs=range(2), meas_idxs=range(2)))
    return model_gate_set


@pytest.fixture()
def generators():
    return {
        "CZ": QubitSparsePauliList(["ZI", "IZ", "XX"]),
        "P": QubitSparsePauliList(["XI", "IX"]),
        "M": QubitSparsePauliList(["XI", "IX"]),
    }


def _model_data(model):
    indices = list(model.input_space)
    n = len(indices)
    return ModelData.from_arrays(
        parameter_indices=indices,
        parameter_values=np.full(n, 0.01),
        covariance=np.eye(n),
        time_lbs=np.empty(n, dtype="datetime64[us]"),
        time_ubs=np.empty(n, dtype="datetime64[us]"),
    )


def test_standalone_matches_method(gate_set, generators):
    model = PauliLindbladModel(gate_set, generators)
    model_data = _model_data(model)

    result = to_pauli_lindblad_maps(model, model_data, include_spam=True)
    expected = model.to_pauli_lindblad_maps(model_data, include_spam=True)

    assert result.keys() == expected.keys()
    for gate_name in expected:
        assert result[gate_name] == expected[gate_name]


def test_composed_propagates_then_converts(gate_set, generators):
    # Pre-composing an identity reparameterization on the rate space leaves the rates unchanged,
    # so the composed result must match the standalone Pauli-Lindblad model.
    model = PauliLindbladModel(gate_set, generators)
    model_data = _model_data(model)

    composed = model.pre_compose(_IdentityMap(model.input_space))
    # the Pauli-Lindblad model is now behind one map in the chain
    assert composed.maps[-1] is model

    result = to_pauli_lindblad_maps(composed, model_data, include_spam=True)
    expected = model.to_pauli_lindblad_maps(model_data, include_spam=True)

    assert result.keys() == expected.keys()
    for gate_name in expected:
        assert result[gate_name] == expected[gate_name]


def test_raises_without_pauli_lindblad_model(gate_set):
    model = CompleteFidelityModel(gate_set)
    # the model contains no Pauli-Lindblad model, so model_data content is irrelevant
    model_data = ModelData.from_arrays(
        parameter_indices=["x"],
        parameter_values=np.array([0.0]),
        covariance=np.array([[1.0]]),
        time_lbs=np.empty(1, dtype="datetime64[us]"),
        time_ubs=np.empty(1, dtype="datetime64[us]"),
    )

    with pytest.raises(ValueError, match="does not contain a PauliLindbladModel"):
        to_pauli_lindblad_maps(model, model_data)
