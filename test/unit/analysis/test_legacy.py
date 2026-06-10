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

"""Unit tests for the legacy noise-model fitter."""

import numpy as np
import pytest
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Clifford, PauliLindbladMap, QubitSparsePauli

from qiskit_noise_learning.analysis import Fit
from qiskit_noise_learning.analysis.legacy import (
    LegacySolve,
    fit_noise_model_legacy,
    get_fid_pairs,
    make_canonical_fid_dict,
    make_conj_pauli_list,
)
from qiskit_noise_learning.data import AveragedData, ModelData
from qiskit_noise_learning.gate_sets import ModelGate, ModelGateSet
from qiskit_noise_learning.sequences import FidelityIndex, Path


@pytest.fixture()
def gate_set_2q_identity():
    """A 2-qubit gate set with one gate "LL" whose Clifford is the identity."""
    mgs = ModelGateSet(2)
    mgs.add_gate(ModelGate("LL", [((0, 1), Clifford(QuantumCircuit(2)))]))
    return mgs


def _pp(gate_set: ModelGateSet, in_pauli: str, out_pauli: str) -> Path:
    """Build an unbound Path with a 2-entry repeatable fragment that loops in_pauli↔out_pauli.

    The two ``FidelityIndex`` entries form the closed (in→out, out→in) cycle expected
    by experiment generators, with empty start/end fragments. Tests only need the
    repeatable fragment.
    """
    gate = gate_set["LL"]
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


def _make_averaged_data(pps: list, fidelities: np.ndarray) -> AveragedData:
    n = len(pps)
    return AveragedData.from_arrays(
        unbound_paths=pps,
        depths=[-1] * n,
        observables=fidelities,
        std=np.full(n, 0.001),
        time_lbs=np.empty(n, dtype="datetime64[us]"),
        time_ubs=np.empty(n, dtype="datetime64[us]"),
    )


class TestMakeCanonicalFidDict:
    def test_filters_weight_ge_three(self):
        # XXXI has weight 3 → must be filtered out.
        result = make_canonical_fid_dict(["IIII", "XXXI"], ["IIII", "XXXI"], np.array([0.95, 0.5]))
        assert "XXXI" not in result
        assert "IIII" in result

    def test_keeps_weight_lt_three(self):
        result = make_canonical_fid_dict(["XIII", "XXII"], ["YIII", "YYII"], np.array([0.9, 0.8]))
        assert set(result) == {"XIII", "XXII", "YIII", "YYII"}

    def test_averages_when_pauli_appears_in_both_lists(self):
        # The function uses the same index i to look up data for both lists, so each
        # row i of fid_pairs_data is associated with both fid_ps_1[i] and fid_ps_2[i].
        # Here XI shows up at fid_ps_1[0] (value 0.9) and at fid_ps_2[1] (value 0.7),
        # so its canonical fidelity is the mean of those two values.
        result = make_canonical_fid_dict(["XI", "ZI"], ["YI", "XI"], np.array([0.9, 0.9]))
        assert result["XI"] == pytest.approx((0.9 + 0.9) / 2)
        assert result["ZI"] == pytest.approx(0.9)
        assert result["YI"] == pytest.approx(0.9)


class TestMakeConjPauliList:
    def test_lookup_in_first_list(self):
        # P appears at index i in fid_ps_1 → conjugate is fid_ps_2[i].
        out = make_conj_pauli_list(["XI"], ["XI", "ZI"], ["YI", "IY"])
        assert out == ["YI"]

    def test_lookup_in_second_list(self):
        # P appears in fid_ps_2 only → conjugate is the matching fid_ps_1 entry.
        out = make_conj_pauli_list(["IY"], ["XI", "ZI"], ["YI", "IY"])
        assert out == ["ZI"]

    def test_mixed_lookup(self):
        out = make_conj_pauli_list(["XI", "IY"], ["XI", "ZI"], ["YI", "IY"])
        assert out == ["YI", "ZI"]

    def test_missing_pauli_silently_skipped(self):
        # A pauli that appears in neither list is silently dropped — documents current
        # behaviour. If this is ever changed to raise, update this test.
        out = make_conj_pauli_list(["NOT_PRESENT"], ["XI"], ["YI"])
        assert out == []


def test_get_fid_pairs_returns_two_qubit_sparse_pauli_lists(gate_set_2q_identity):
    # Self-conjugate XI and ZI under the identity Clifford: each Path's
    # repeatable_fragment[0].pauli and [1].pauli are the same.
    pps = [_pp(gate_set_2q_identity, "XI", "XI"), _pp(gate_set_2q_identity, "ZI", "ZI")]
    ad = _make_averaged_data(pps, np.array([0.9, 0.8]))
    fit = Fit()
    fit[AveragedData] = ad

    fid_ps_1, fid_ps_2 = get_fid_pairs(fit.averaged_data.dataset.observables.unbound_path.data)
    assert fid_ps_1.to_pauli_list().to_labels() == ["XI", "ZI"]
    assert fid_ps_2.to_pauli_list().to_labels() == ["XI", "ZI"]


def test_get_fid_pairs_raises_on_wrong_fragment_length(gate_set_2q_identity):
    fi = FidelityIndex.from_transition(
        gate=gate_set_2q_identity["LL"],
        in_pauli=QubitSparsePauli("XI"),
        out_pauli=QubitSparsePauli("XI"),
    )
    pp_3 = Path(start_fragment=[], repeatable_fragment=[fi, fi, fi], end_fragment=[])

    ad = _make_averaged_data([pp_3], np.array([0.9]))
    fit = Fit()
    fit[AveragedData] = ad

    with pytest.raises(ValueError, match="repeatable_fragment"):
        get_fid_pairs(fit.averaged_data.dataset.observables.unbound_path.data)


@pytest.fixture()
def two_qubit_anticomm_fit(gate_set_2q_identity) -> Fit:
    """A Fit whose canonical pair fidelities are generated by a known noise model.

    Basis paulis are XI and ZI (anticommuting) with self-conjugate pairs (identity
    Clifford), so the non-commuting matrix M is anti-diagonal. We pick rates
    λ_XI = 0.1, λ_ZI = 0.05 and choose pauli fidelities so that ``M @ λ = -log(f)/4``
    (the equation solved by symmetric_fidelities).
    """
    pps = [_pp(gate_set_2q_identity, "XI", "XI"), _pp(gate_set_2q_identity, "ZI", "ZI")]
    f_xi = np.exp(-0.2)  # so that -log(f)/4 = 0.05 = M-row dot λ for XI
    f_zi = np.exp(-0.4)  # so that -log(f)/4 = 0.10 = M-row dot λ for ZI
    fit = Fit()
    fit[AveragedData] = _make_averaged_data(pps, np.array([f_xi, f_zi]))
    return fit


@pytest.mark.parametrize("optimizer", ["nnls", "lsq_linear_sparse", "cvxpy"])
def test_recovers_known_rates_symmetric_fidelities(two_qubit_anticomm_fit, optimizer):
    if optimizer == "cvxpy":
        pytest.importorskip("cvxpy")

    nm = fit_noise_model_legacy(
        two_qubit_anticomm_fit,
        noise_assumption="symmetric_fidelities",
        optimizer_name=optimizer,
    )

    # cvxpy's interior-point solver is less precise than NNLS / BVLS, so loosen the tolerance.
    tol = 1e-4 if optimizer == "cvxpy" else 1e-6
    rates_by_label = {g.to_pauli().to_label(): r for g, r in zip(nm.generators(), nm.rates)}
    assert rates_by_label["XI"] == pytest.approx(0.1, abs=tol)
    assert rates_by_label["ZI"] == pytest.approx(0.05, abs=tol)


def test_returns_pauli_lindblad_map(two_qubit_anticomm_fit):
    nm = fit_noise_model_legacy(two_qubit_anticomm_fit)
    assert isinstance(nm, PauliLindbladMap)
    assert len(list(nm.generators())) == 2


def test_decimals_rounds_rates(two_qubit_anticomm_fit):
    nm = fit_noise_model_legacy(two_qubit_anticomm_fit, decimals=1)
    rates = sorted(nm.rates, reverse=True)
    # 0.10 stays 0.1; 0.05 rounds to 0.1 (banker's rounding via numpy → 0.0 or 0.1).
    # Either way both are 1-decimal-rounded values.
    for r in rates:
        assert r == round(r, 1)


def test_unrecognized_optimizer_raises(two_qubit_anticomm_fit):
    with pytest.raises(ValueError, match="Optimizer name"):
        fit_noise_model_legacy(two_qubit_anticomm_fit, optimizer_name="not_a_solver")


def test_unrecognized_assumption_raises(two_qubit_anticomm_fit):
    with pytest.raises(ValueError, match="Noise assumption"):
        fit_noise_model_legacy(two_qubit_anticomm_fit, noise_assumption="not_an_assumption")


def test_nnls_with_constrained_false_raises(two_qubit_anticomm_fit):
    with pytest.raises(ValueError, match="constrained=False"):
        fit_noise_model_legacy(two_qubit_anticomm_fit, optimizer_name="nnls", constrained=False)


def test_cvxpy_branch_raises_when_unavailable(two_qubit_anticomm_fit, monkeypatch):
    # Only test that requires a mock: substitute legacy.HAS_CVXPY with a stub whose
    # require_now raises ImportError, simulating cvxpy not being installed.
    from qiskit_noise_learning.analysis import legacy

    class _UnavailableCVXPY:
        @staticmethod
        def require_now(feature):
            raise ImportError(f"no cvxpy ({feature})")

    monkeypatch.setattr(legacy, "HAS_CVXPY", _UnavailableCVXPY)

    with pytest.raises(ImportError):
        fit_noise_model_legacy(two_qubit_anticomm_fit, optimizer_name="cvxpy")


def test_zero_noise_yields_zero_rates(gate_set_2q_identity):
    # Perfect (1.0) fidelities → all rates zero (NNLS lower bound is 0).
    pps = [_pp(gate_set_2q_identity, "XI", "XI"), _pp(gate_set_2q_identity, "ZI", "ZI")]
    fit = Fit()
    fit[AveragedData] = _make_averaged_data(pps, np.array([1.0, 1.0]))
    nm = fit_noise_model_legacy(fit)
    assert all(r == pytest.approx(0.0, abs=1e-12) for r in nm.rates)


def test_legacy_solve_writes_model_data(two_qubit_anticomm_fit):
    result = LegacySolve().run(two_qubit_anticomm_fit)

    md = result.model_data
    assert isinstance(md, ModelData)

    # Parameter labels should be GeneratorIndex objects with the gate name from the
    # unbound path's repeatable_fragment[0].gate_name (the gate set fixture uses "LL").
    indices = md.dataset["parameter"].values.tolist()
    assert all(idx.gate_name == "LL" for idx in indices)

    # Recovered rates should match the analytical values.
    rates_by_gen_label = {
        idx.generator.to_pauli().to_label(): float(val)
        for idx, val in zip(indices, md.dataset["parameter_values"].values)
    }
    assert rates_by_gen_label["XI"] == pytest.approx(0.1, abs=1e-6)
    assert rates_by_gen_label["ZI"] == pytest.approx(0.05, abs=1e-6)


def test_legacy_solve_covariance_is_zero(two_qubit_anticomm_fit):
    # LegacySolve does not currently compute a covariance — it reports zero.
    result = LegacySolve().run(two_qubit_anticomm_fit)
    cov = result.model_data.dataset["covariance"].values
    assert cov.shape == (2, 2)
    assert np.allclose(cov, 0.0)
