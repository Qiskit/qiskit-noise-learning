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

from qiskit_noise_learning.gate_sets import ModelGate, ModelGateSet
from qiskit_noise_learning.math import (
    ComposedLinearMap,
    IndexedVector,
    LinearMap,
)
from qiskit_noise_learning.models import (
    CompleteFidelityModel,
    ComposedFidelityModel,
    FidelityModel,
)
from qiskit_noise_learning.sequences import FidelityIndex


@pytest.fixture()
def gate_set():
    model_gate_set = ModelGateSet(2)
    model_gate_set.add_gate(ModelGate("CZ", [((0, 1), Clifford(CZGate()))]))
    model_gate_set.add_gate(ModelGate("P", qubit_idxs=range(2), prep_idxs=range(2)))
    model_gate_set.add_gate(ModelGate("M", qubit_idxs=range(2), meas_idxs=range(2)))
    return model_gate_set


@pytest.fixture()
def fidelity_index(gate_set):
    return FidelityIndex.from_gate(
        gate=gate_set["CZ"],
        pauli=QubitSparsePauli.from_sparse_label(("X", [0]), num_qubits=2),
        in_bit_indices=frozenset(),
        out_bit_indices=frozenset(),
    )


class _NonFidelityMap(LinearMap):
    """A minimal linear map whose output space is not a fidelity index space."""

    def row(self, output_index):
        return IndexedVector()


def test_compose_returns_composed_fidelity_model(gate_set):
    result = CompleteFidelityModel(gate_set).compose(CompleteFidelityModel(gate_set))
    assert isinstance(result, ComposedFidelityModel)
    assert isinstance(result, FidelityModel)


def test_pre_compose_returns_composed_fidelity_model(gate_set):
    result = CompleteFidelityModel(gate_set).pre_compose(CompleteFidelityModel(gate_set))
    assert isinstance(result, ComposedFidelityModel)
    assert isinstance(result, FidelityModel)


def test_matmul_is_pre_compose(gate_set):
    a = CompleteFidelityModel(gate_set)
    b = CompleteFidelityModel(gate_set)
    # ``a @ b`` means a applied after b, i.e. chain [b, a].
    result = a @ b
    assert isinstance(result, ComposedFidelityModel)
    assert result.maps[-1] is a


def test_compose_flattens_chain(gate_set):
    a, b, c = (CompleteFidelityModel(gate_set) for _ in range(3))
    result = a.compose(b.compose(c))
    assert result.maps == [a, b, c]
    assert not any(isinstance(m, ComposedLinearMap) for m in result.maps)


def test_pre_compose_flattens_chain(gate_set):
    a, b, c = (CompleteFidelityModel(gate_set) for _ in range(3))
    result = a.pre_compose(b.compose(c))
    assert result.maps == [b, c, a]
    assert not any(isinstance(m, ComposedLinearMap) for m in result.maps)


def test_composed_compose_flattens_chain(gate_set):
    a, b, c = (CompleteFidelityModel(gate_set) for _ in range(3))
    composed = a.compose(b)
    result = composed.compose(c)
    assert result.maps == [a, b, c]


def test_gate_set_from_last_map(gate_set):
    result = CompleteFidelityModel(gate_set).compose(CompleteFidelityModel(gate_set))
    assert result.gate_set is gate_set.model_gate_set
