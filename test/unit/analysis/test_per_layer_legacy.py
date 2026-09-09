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

"""Unit tests for PerLayerLegacySolve."""

import numpy as np
import pytest
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Clifford, QubitSparsePauli

from qiskit_noise_learning.analysis import Fit
from qiskit_noise_learning.analysis.per_layer_legacy import PerLayerLegacySolve
from qiskit_noise_learning.data import AggregatedObservableData, ModelData
from qiskit_noise_learning.gate_sets import ModelGate, ModelGateSet
from qiskit_noise_learning.sequences import FidelityIndex, Path


@pytest.fixture()
def gate_set_2q_identity():
    mgs = ModelGateSet(2)
    mgs.add_gate(ModelGate("LL", [((0, 1), Clifford(QuantumCircuit(2)))]))
    return mgs


@pytest.fixture()
def gate_set_two_layers():
    mgs = ModelGateSet(2)
    mgs.add_gate(ModelGate("LL", [((0, 1), Clifford(QuantumCircuit(2)))]))
    mgs.add_gate(ModelGate("MM", [((0, 1), Clifford(QuantumCircuit(2)))]))
    return mgs


def _pp(gate, gate_set: ModelGateSet, in_pauli: str, out_pauli: str) -> Path:
    return Path(
        start_fragment=[],
        repeatable_fragment=[
            FidelityIndex.from_transition(
                gate=gate,
                in_pauli=QubitSparsePauli(in_pauli),
                out_pauli=QubitSparsePauli(out_pauli),
            ),
            FidelityIndex.from_transition(
                gate=gate,
                in_pauli=QubitSparsePauli(out_pauli),
                out_pauli=QubitSparsePauli(in_pauli),
            ),
        ],
        end_fragment=[],
    )


def _make_aggregated_observable_data(pps: list, fidelities: np.ndarray) -> AggregatedObservableData:
    n = len(pps)
    return AggregatedObservableData.from_arrays(
        unbound_paths=pps,
        fragment_depths=[-1] * n,
        estimate_values=fidelities,
        estimate_std=np.full(n, 0.001),
        time_lbs=np.empty(n, dtype="datetime64[us]"),
        time_ubs=np.empty(n, dtype="datetime64[us]"),
    )


@pytest.fixture()
def single_layer_fit(gate_set_2q_identity) -> Fit:
    """Single-gate fit with the same known rates as the LegacySolve fixture."""
    gate = gate_set_2q_identity["LL"]
    pps = [
        _pp(gate, gate_set_2q_identity, "XI", "XI"),
        _pp(gate, gate_set_2q_identity, "ZI", "ZI"),
    ]
    f_xi = np.exp(-0.2)
    f_zi = np.exp(-0.4)
    fit = Fit()
    fit[AggregatedObservableData] = _make_aggregated_observable_data(pps, np.array([f_xi, f_zi]))
    return fit


@pytest.fixture()
def two_layer_fit(gate_set_two_layers) -> Fit:
    """Two-gate fit.  LL has λ_XI=0.1, λ_ZI=0.05; MM has λ_XI=0.14, λ_ZI=0.10.

    Under symmetric_fidelities, fit_vector[i] = -log(f_i)/4 and M is the non-commuting matrix.
    Choosing fids_mm = [exp(-0.4), exp(-0.56)] yields λ_XI=0.14 and λ_ZI=0.10 for MM.
    """
    gate_ll = gate_set_two_layers["LL"]
    gate_mm = gate_set_two_layers["MM"]
    pps_ll = [
        _pp(gate_ll, gate_set_two_layers, "XI", "XI"),
        _pp(gate_ll, gate_set_two_layers, "ZI", "ZI"),
    ]
    pps_mm = [
        _pp(gate_mm, gate_set_two_layers, "XI", "XI"),
        _pp(gate_mm, gate_set_two_layers, "ZI", "ZI"),
    ]
    fids_ll = np.array([np.exp(-0.2), np.exp(-0.4)])
    fids_mm = np.array([np.exp(-0.4), np.exp(-0.56)])
    fit = Fit()
    fit[AggregatedObservableData] = _make_aggregated_observable_data(
        pps_ll + pps_mm, np.concatenate([fids_ll, fids_mm])
    )
    return fit


class TestPerLayerLegacySolveSingleLayer:
    def test_writes_model_data(self, single_layer_fit):
        result = PerLayerLegacySolve().run(single_layer_fit)
        assert isinstance(result.model_data, ModelData)

    def test_gate_name_matches_layer(self, single_layer_fit):
        result = PerLayerLegacySolve().run(single_layer_fit)
        indices = result.model_data.dataset["parameter_index"].values.tolist()
        assert all(idx.gate_name == "LL" for idx in indices)

    def test_recovers_known_rates(self, single_layer_fit):
        result = PerLayerLegacySolve().run(single_layer_fit)
        md = result.model_data
        rates_by_gen = {
            idx.generator.to_pauli().to_label(): float(val)
            for idx, val in zip(
                md.dataset["parameter_index"].values, md.dataset["parameter_values"].values
            )
        }
        assert rates_by_gen["XI"] == pytest.approx(0.1, abs=1e-6)
        assert rates_by_gen["ZI"] == pytest.approx(0.05, abs=1e-6)

    def test_covariance_is_zero(self, single_layer_fit):
        result = PerLayerLegacySolve().run(single_layer_fit)
        cov = result.model_data.dataset["covariance"].values
        assert cov.shape == (2, 2)
        assert np.allclose(cov, 0.0)


class TestPerLayerLegacySolveTwoLayers:
    def test_writes_model_data(self, two_layer_fit):
        result = PerLayerLegacySolve().run(two_layer_fit)
        assert isinstance(result.model_data, ModelData)

    def test_both_gate_names_present(self, two_layer_fit):
        result = PerLayerLegacySolve().run(two_layer_fit)
        indices = result.model_data.dataset["parameter_index"].values.tolist()
        gate_names = {idx.gate_name for idx in indices}
        assert gate_names == {"LL", "MM"}

    def test_layer_order_is_first_seen(self, two_layer_fit):
        result = PerLayerLegacySolve().run(two_layer_fit)
        indices = result.model_data.dataset["parameter_index"].values.tolist()
        # LL rows come before MM rows in the fixture, so LL labels must appear first.
        first_name = indices[0].gate_name
        assert first_name == "LL"

    def test_recovers_ll_rates(self, two_layer_fit):
        result = PerLayerLegacySolve().run(two_layer_fit)
        md = result.model_data
        rates_by_gen = {
            (idx.gate_name, idx.generator.to_pauli().to_label()): float(val)
            for idx, val in zip(
                md.dataset["parameter_index"].values, md.dataset["parameter_values"].values
            )
        }
        assert rates_by_gen[("LL", "XI")] == pytest.approx(0.1, abs=1e-6)
        assert rates_by_gen[("LL", "ZI")] == pytest.approx(0.05, abs=1e-6)

    def test_recovers_mm_rates(self, two_layer_fit):
        result = PerLayerLegacySolve().run(two_layer_fit)
        md = result.model_data
        rates_by_gen = {
            (idx.gate_name, idx.generator.to_pauli().to_label()): float(val)
            for idx, val in zip(
                md.dataset["parameter_index"].values, md.dataset["parameter_values"].values
            )
        }
        # fids_mm = [exp(-0.4), exp(-0.56)] → λ_XI=0.14 (from ZI row), λ_ZI=0.10 (from XI row)
        assert rates_by_gen[("MM", "XI")] == pytest.approx(0.14, abs=1e-6)
        assert rates_by_gen[("MM", "ZI")] == pytest.approx(0.10, abs=1e-6)

    def test_covariance_shape_and_zero(self, two_layer_fit):
        result = PerLayerLegacySolve().run(two_layer_fit)
        cov = result.model_data.dataset["covariance"].values
        assert cov.shape == (4, 4)
        assert np.allclose(cov, 0.0)


class TestPerLayerLegacySolveFailures:
    def test_wrong_fragment_length_raises(self, gate_set_2q_identity):
        gate = gate_set_2q_identity["LL"]
        fi = FidelityIndex.from_transition(
            gate=gate, in_pauli=QubitSparsePauli("XI"), out_pauli=QubitSparsePauli("XI")
        )
        pp_3 = Path(start_fragment=[], repeatable_fragment=[fi, fi, fi], end_fragment=[])
        fit = Fit()
        fit[AggregatedObservableData] = _make_aggregated_observable_data([pp_3], np.array([0.9]))
        with pytest.raises(ValueError, match="repeatable_fragment"):
            PerLayerLegacySolve().run(fit)

    def test_empty_repeatable_fragment_raises(self):
        pp_empty = Path(start_fragment=[], repeatable_fragment=[], end_fragment=[])
        fit = Fit()
        fit[AggregatedObservableData] = _make_aggregated_observable_data(
            [pp_empty], np.array([0.9])
        )
        with pytest.raises(ValueError, match="repeatable_fragment"):
            PerLayerLegacySolve().run(fit)
