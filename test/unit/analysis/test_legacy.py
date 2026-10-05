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
from qiskit.quantum_info import Clifford, PauliLindbladMap, QubitSparsePauli, QubitSparsePauliList

from qiskit_noise_learning.analysis import Fit
from qiskit_noise_learning.analysis.legacy import (
    LegacySolve,
    fit_noise_model_legacy,
    get_fid_pairs,
    make_canonical_fid_dict,
    make_conj_pauli_list,
)
from qiskit_noise_learning.data import AggregatedObservableData, ModelData
from qiskit_noise_learning.gate_sets import ModelGate, ModelGateSet
from qiskit_noise_learning.models import PauliLindbladModel
from qiskit_noise_learning.sequences import FidelityIndex, Path


@pytest.fixture()
def gate_set_2q_identity():
    """A 2-qubit gate set with one gate "LL" whose Clifford is the identity."""
    mgs = ModelGateSet(2)
    mgs.add_gate(ModelGate("LL", [((0, 1), Clifford(QuantumCircuit(2)))]))
    return mgs


@pytest.fixture()
def gate_set_1q_order_3():
    """A 1-qubit gate set with one gate "C3" cycling the Paulis, X → Z → Y → X."""
    circuit = QuantumCircuit(1)
    circuit.h(0)
    circuit.s(0)
    mgs = ModelGateSet(1)
    mgs.add_gate(ModelGate("C3", [((0,), Clifford(circuit))]))
    return mgs


@pytest.fixture()
def gate_set_3q_weight_3_conjugate():
    """A 3-qubit gate set whose gate "G" maps ``IXI`` to ``YYZ``, which anticommutes with it."""
    circuit = QuantumCircuit(3)
    circuit.cz(0, 1)
    circuit.cx(1, 2)
    circuit.cz(1, 2)
    mgs = ModelGateSet(3)
    mgs.add_gate(ModelGate("G", [((0, 1, 2), Clifford(circuit))]))
    return mgs


def _pp(gate_set: ModelGateSet, in_pauli: str, out_pauli: str, gate_name: str = "LL") -> Path:
    """Build an unbound Path with a 2-entry repeatable fragment that loops in_pauli↔out_pauli.

    The two ``FidelityIndex`` entries form the closed (in→out, out→in) cycle expected
    by experiment generators, with empty start/end fragments. Tests only need the
    repeatable fragment.
    """
    gate = gate_set[gate_name]
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


def _fitted_rates(model_data: ModelData) -> dict[tuple[str, str], float]:
    """The fitted rates, keyed by gate name and generator label."""
    return {
        (index.gate_name, index.generator.to_pauli().to_label()): float(rate)
        for index, rate in zip(
            model_data.dataset["parameter_index"].data,
            model_data.dataset["parameter_values"].data,
        )
    }


def _make_aggregated_observable_data(
    pps: list, fidelities: np.ndarray, fragment_depth: int = -1
) -> AggregatedObservableData:
    n = len(pps)
    return AggregatedObservableData.from_arrays(
        unbound_paths=pps,
        fragment_depths=[fragment_depth] * n,
        estimate_values=fidelities,
        estimate_std=np.full(n, 0.001),
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
    ad = _make_aggregated_observable_data(pps, np.array([0.9, 0.8]))
    fit = Fit()
    fit[AggregatedObservableData] = ad

    fid_ps_1, fid_ps_2 = get_fid_pairs(
        fit.aggregated_observable_data.dataset.estimate_values.unbound_path.data
    )
    assert fid_ps_1.to_pauli_list().to_labels() == ["XI", "ZI"]
    assert fid_ps_2.to_pauli_list().to_labels() == ["XI", "ZI"]


def test_get_fid_pairs_raises_on_fragment_that_does_not_close(gate_set_1q_order_3):
    # A case where the path is valid but single qubit Cliffords are necessary
    gate = gate_set_1q_order_3["C3"]
    x = QubitSparsePauli("X")
    cx = gate.clifford_propagate(x)
    ccx = gate.clifford_propagate(cx)
    chains_but_does_not_close = Path(
        start_fragment=[],
        repeatable_fragment=[
            FidelityIndex.from_transition(gate=gate, in_pauli=x, out_pauli=cx),
            FidelityIndex.from_transition(gate=gate, in_pauli=cx, out_pauli=ccx),
        ],
        end_fragment=[],
    )

    with pytest.raises(ValueError, match="single qubit Cliffords"):
        get_fid_pairs([chains_but_does_not_close])


def test_legacy_solve_rejects_gate_rows_that_are_not_decays(gate_set_2q_identity):
    pps = [_pp(gate_set_2q_identity, "XI", "XI"), _pp(gate_set_2q_identity, "ZI", "ZI")]
    fit = Fit()
    fit[AggregatedObservableData] = _make_aggregated_observable_data(
        pps, np.array([0.9, 0.8]), fragment_depth=2
    )

    with pytest.raises(ValueError, match="exponential decay data"):
        LegacySolve().run(fit)


def test_legacy_solve_raises_on_mixed_gate_fragment(gate_set_two_layers):
    pauli = QubitSparsePauli("XI")
    mixed = Path(
        start_fragment=[],
        repeatable_fragment=[
            FidelityIndex.from_transition(
                gate=gate_set_two_layers[name], in_pauli=pauli, out_pauli=pauli
            )
            for name in ("LL", "MM")
        ],
        end_fragment=[],
    )
    fit = Fit()
    fit[AggregatedObservableData] = _make_aggregated_observable_data([mixed], np.array([0.9]))

    with pytest.raises(ValueError, match="same gate"):
        LegacySolve().run(fit)


def test_get_fid_pairs_raises_on_wrong_fragment_length(gate_set_2q_identity):
    fi = FidelityIndex.from_transition(
        gate=gate_set_2q_identity["LL"],
        in_pauli=QubitSparsePauli("XI"),
        out_pauli=QubitSparsePauli("XI"),
    )
    pp_3 = Path(start_fragment=[], repeatable_fragment=[fi, fi, fi], end_fragment=[])

    ad = _make_aggregated_observable_data([pp_3], np.array([0.9]))
    fit = Fit()
    fit[AggregatedObservableData] = ad

    with pytest.raises(ValueError, match="repeatable_fragment"):
        get_fid_pairs(fit.aggregated_observable_data.dataset.estimate_values.unbound_path.data)


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
    fit[AggregatedObservableData] = _make_aggregated_observable_data(pps, np.array([f_xi, f_zi]))
    return fit


@pytest.mark.parametrize("optimizer", ["nnls", "lsq_linear_sparse", "cvxpy"])
def test_recovers_known_rates_symmetric_fidelities(two_qubit_anticomm_fit, optimizer):
    if optimizer == "cvxpy":
        pytest.importorskip("cvxpy")

    nm = fit_noise_model_legacy(
        two_qubit_anticomm_fit.aggregated_observable_data,
        noise_assumption="symmetric_fidelities",
        optimizer_name=optimizer,
    )

    # cvxpy's interior-point solver is less precise than NNLS / BVLS, so loosen the tolerance.
    tol = 1e-4 if optimizer == "cvxpy" else 1e-6
    rates_by_label = {g.to_pauli().to_label(): r for g, r in zip(nm.generators(), nm.rates)}
    assert rates_by_label["XI"] == pytest.approx(0.1, abs=tol)
    assert rates_by_label["ZI"] == pytest.approx(0.05, abs=tol)


def test_fits_a_design_with_a_single_generator(gate_set_3q_weight_3_conjugate):
    gate = gate_set_3q_weight_3_conjugate["G"]
    in_pauli = QubitSparsePauli("IXI")
    out_pauli = gate.clifford_propagate(in_pauli)
    path = Path(
        start_fragment=[],
        repeatable_fragment=[
            FidelityIndex.from_transition(gate=gate, in_pauli=in_pauli, out_pauli=out_pauli),
            FidelityIndex.from_transition(gate=gate, in_pauli=out_pauli, out_pauli=in_pauli),
        ],
        end_fragment=[],
    )
    fidelity = 0.9

    noise_map = fit_noise_model_legacy(
        _make_aggregated_observable_data([path], np.array([fidelity]))
    )

    rates_by_label = {
        g.to_pauli().to_label(): r for g, r in zip(noise_map.generators(), noise_map.rates)
    }
    assert list(rates_by_label) == ["IXI"]
    assert rates_by_label["IXI"] == pytest.approx(-np.log(fidelity) / 4)


@pytest.mark.parametrize("fidelity", [0.0, -0.1])
def test_rejects_non_positive_fidelities(gate_set_2q_identity, fidelity):
    pps = [_pp(gate_set_2q_identity, "XI", "XI"), _pp(gate_set_2q_identity, "ZI", "ZI")]
    ad = _make_aggregated_observable_data(pps, np.array([fidelity, 0.8]))

    with pytest.raises(ValueError, match="Pair fidelities must be positive"):
        fit_noise_model_legacy(ad)


def test_accepts_fidelity_above_one(gate_set_2q_identity):
    """A fidelity above 1 is ordinary shot noise; the non-negativity constraint absorbs it."""
    pps = [_pp(gate_set_2q_identity, "XI", "XI"), _pp(gate_set_2q_identity, "ZI", "ZI")]
    ad = _make_aggregated_observable_data(pps, np.array([1.01, 0.8]))

    noise_map = fit_noise_model_legacy(ad)

    assert all(rate >= 0 for rate in noise_map.rates)


def test_returns_pauli_lindblad_map(two_qubit_anticomm_fit):
    nm = fit_noise_model_legacy(two_qubit_anticomm_fit.aggregated_observable_data)
    assert isinstance(nm, PauliLindbladMap)
    assert len(list(nm.generators())) == 2


def test_decimals_rounds_rates(two_qubit_anticomm_fit):
    nm = fit_noise_model_legacy(two_qubit_anticomm_fit.aggregated_observable_data, decimals=1)
    rates = sorted(nm.rates, reverse=True)
    # 0.10 stays 0.1; 0.05 rounds to 0.1 (banker's rounding via numpy → 0.0 or 0.1).
    # Either way both are 1-decimal-rounded values.
    for r in rates:
        assert r == round(r, 1)


def test_unrecognized_optimizer_raises(two_qubit_anticomm_fit):
    with pytest.raises(ValueError, match="Optimizer name"):
        fit_noise_model_legacy(
            two_qubit_anticomm_fit.aggregated_observable_data, optimizer_name="not_a_solver"
        )


def test_unrecognized_assumption_raises(two_qubit_anticomm_fit):
    with pytest.raises(ValueError, match="Noise assumption"):
        fit_noise_model_legacy(
            two_qubit_anticomm_fit.aggregated_observable_data,
            noise_assumption="not_an_assumption",
        )


def test_nnls_with_constrained_false_raises(two_qubit_anticomm_fit):
    with pytest.raises(ValueError, match="constrained=False"):
        fit_noise_model_legacy(
            two_qubit_anticomm_fit.aggregated_observable_data,
            optimizer_name="nnls",
            constrained=False,
        )


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
        fit_noise_model_legacy(
            two_qubit_anticomm_fit.aggregated_observable_data, optimizer_name="cvxpy"
        )


def test_zero_noise_yields_zero_rates(gate_set_2q_identity):
    # Perfect (1.0) fidelities → all rates zero (NNLS lower bound is 0).
    pps = [_pp(gate_set_2q_identity, "XI", "XI"), _pp(gate_set_2q_identity, "ZI", "ZI")]
    fit = Fit()
    fit[AggregatedObservableData] = _make_aggregated_observable_data(pps, np.array([1.0, 1.0]))
    nm = fit_noise_model_legacy(fit.aggregated_observable_data)
    assert all(r == pytest.approx(0.0, abs=1e-12) for r in nm.rates)


@pytest.fixture()
def gate_set_two_layers(gate_set_2q_identity):
    gate_set_2q_identity.add_gate(ModelGate("MM", [((0, 1), Clifford(QuantumCircuit(2)))]))
    return gate_set_2q_identity


@pytest.mark.parametrize("gate_names", [("LL",), ("LL", "MM"), ("MM", "LL")])
def test_legacy_solve_recovers_layer_rates(gate_set_two_layers, gate_names):
    # Pair fidelities are exp(-4 * the anticommuting generator's rate).
    rates_by_gate = {"LL": (0.1, 0.05), "MM": (0.14, 0.10)}
    paths, fidelities, expected_rates = [], [], {}
    for name in gate_names:
        x_rate, z_rate = rates_by_gate[name]
        paths.extend(_pp(gate_set_two_layers, p, p, name) for p in ("XI", "ZI"))
        fidelities.extend(np.exp(-4 * np.array([z_rate, x_rate])))
        expected_rates.update({(name, "XI"): x_rate, (name, "ZI"): z_rate})
    fit = Fit()
    fit[AggregatedObservableData] = _make_aggregated_observable_data(paths, np.array(fidelities))

    md = LegacySolve().run(fit).model_data
    assert isinstance(md, ModelData)
    indices = md.dataset["parameter_index"].values
    rates = _fitted_rates(md)
    assert rates == pytest.approx(expected_rates, abs=1e-6)
    assert [idx.gate_name for idx in indices] == [name for name in gate_names for _ in range(2)]
    # LegacySolve reports zero covariance for all returned parameters.
    np.testing.assert_array_equal(
        md.dataset["covariance"].values, np.zeros((len(rates), len(rates)))
    )


def test_legacy_solve_omits_unestimated_generators(two_qubit_anticomm_fit, gate_set_two_layers):
    gate_set_two_layers.add_gate(ModelGate("P", qubit_idxs=[0, 1], prep_idxs=[0, 1]))
    gate_set_two_layers.add_gate(ModelGate("M", qubit_idxs=[0, 1], meas_idxs=[0, 1]))
    model = PauliLindbladModel(
        gate_set_two_layers,
        generators={
            "LL": QubitSparsePauliList(["XI", "ZI", "ZZ"]),
            "MM": QubitSparsePauliList(["XI", "ZI"]),
            "P": QubitSparsePauliList(["XI", "IX"]),
            "M": QubitSparsePauliList(["XI", "IX"]),
        },
    )
    fit = Fit(model=model)
    fit[AggregatedObservableData] = two_qubit_anticomm_fit.aggregated_observable_data

    expected = LegacySolve().run(two_qubit_anticomm_fit).model_data.dataset
    actual = LegacySolve().run(fit).model_data.dataset
    # Adding a model must not add rates for extra generators, unsolved layers, or SPAM.
    assert actual.identical(expected)


def test_legacy_solve_rejects_invalid_fragments(gate_set_2q_identity):
    fi = _pp(gate_set_2q_identity, "XI", "XI").repeatable_fragment[0]
    path = Path(start_fragment=[], repeatable_fragment=[fi] * 3, end_fragment=[])
    fit = Fit()
    fit[AggregatedObservableData] = _make_aggregated_observable_data([path], np.array([0.9]))
    with pytest.raises(ValueError, match="repeatable_fragment"):
        LegacySolve().run(fit)


@pytest.fixture()
def gate_set_spam(gate_set_2q_identity):
    """The 2-qubit layer gate set, plus a preparation and a measurement on both qubits."""
    gate_set_2q_identity.add_gate(ModelGate("P", qubit_idxs=[0, 1], prep_idxs=[0, 1]))
    gate_set_2q_identity.add_gate(ModelGate("M", qubit_idxs=[0, 1], meas_idxs=[0, 1]))
    return gate_set_2q_identity


def _spam_model(gate_set: ModelGateSet, **overrides: list[str]) -> PauliLindbladModel:
    """A model declaring exactly what the SPAM fit assumes, unless an override says otherwise."""
    labels = {"LL": ["XI", "ZI"], "P": [], "M": ["XI", "IX"]} | overrides
    return PauliLindbladModel(
        gate_set,
        generators={
            name: QubitSparsePauliList(these) if these else QubitSparsePauliList.empty(2)
            for name, these in labels.items()
        },
    )


def _spam_path(gate_set: ModelGateSet, qubits: tuple[int, ...]) -> Path:
    """A path preparing and measuring ``Z`` on ``qubits``, as SPAMPaths generates.

    Left unbound: the aggregated data's coordinate holds the unbound path, with the depth of 0 in
    its own column, which is what the analysis pipeline writes for a bound SPAM path.
    """
    identity = QubitSparsePauli.identity(gate_set.num_qubits)
    return Path(
        start_fragment=[
            FidelityIndex.from_gate(
                gate=gate_set["P"],
                pauli=identity,
                in_z_idxs=frozenset(),
                out_z_idxs=frozenset(qubits),
            )
        ],
        repeatable_fragment=[],
        end_fragment=[
            FidelityIndex.from_gate(
                gate=gate_set["M"],
                pauli=identity,
                in_z_idxs=frozenset(qubits),
                out_z_idxs=frozenset(),
            )
        ],
    )


def _spam_data(paths: list, fidelities: np.ndarray) -> AggregatedObservableData:
    """Aggregated data for bound SPAM rows, which carry their real fragment depth of 0."""
    return _make_aggregated_observable_data(paths, fidelities, fragment_depth=0)


@pytest.mark.parametrize("with_model", [False, True])
def test_spam_rates_are_recovered(gate_set_spam, with_model):
    """Planted fidelities come back exactly whether or not a model is present: the fit is
    model-free, and a model declaring what it fits is accepted unchanged."""
    planted = {0: 1.3e-2, 1: 7.0e-3}
    paths = [_spam_path(gate_set_spam, (qubit,)) for qubit in planted]
    fit = Fit(model=_spam_model(gate_set_spam) if with_model else None)
    fit[AggregatedObservableData] = _spam_data(
        paths, np.array([np.exp(-2 * rate) for rate in planted.values()])
    )

    rates = _fitted_rates(LegacySolve().run(fit).model_data)

    assert rates == pytest.approx({("M", "IX"): 1.3e-2, ("M", "XI"): 7.0e-3})


def test_spam_rate_is_clipped_at_zero(gate_set_spam):
    """A fidelity above 1 is shot noise, and would otherwise give a negative rate."""
    fit = Fit()
    fit[AggregatedObservableData] = _spam_data([_spam_path(gate_set_spam, (0,))], np.array([1.02]))

    model_data = LegacySolve().run(fit).model_data

    assert _fitted_rates(model_data) == {("M", "IX"): 0.0}
    # on the non-negativity boundary, where ModelSolve likewise reports no variance
    assert model_data.dataset["covariance"].data.tolist() == [[0.0]]


def test_spam_covariance_is_propagated(gate_set_spam):
    """Each SPAM rate comes from one row, so its variance is that row's uncertainty propagated."""
    fidelity = 0.97
    data = _spam_data([_spam_path(gate_set_spam, (0,))], np.array([fidelity]))
    fidelity_std = float(data.dataset["estimate_std"].data[0])
    fit = Fit()
    fit[AggregatedObservableData] = data

    covariance = LegacySolve().run(fit).model_data.dataset["covariance"].data

    np.testing.assert_allclose(covariance, [[(fidelity_std / (2 * fidelity)) ** 2]])


def test_spam_rows_do_not_change_the_gate_fit(gate_set_spam):
    """Adding SPAM rows must leave every gate rate bit-identical: the blocks are independent."""
    paths = [_pp(gate_set_spam, pauli, pauli) for pauli in ("XI", "ZI")]
    fidelities = np.array([0.9, 0.8])
    without_spam = Fit()
    without_spam[AggregatedObservableData] = _make_aggregated_observable_data(paths, fidelities)
    with_spam = Fit()
    with_spam[AggregatedObservableData] = _make_aggregated_observable_data(paths, fidelities).merge(
        _spam_data([_spam_path(gate_set_spam, (0,))], np.array([0.97]))
    )

    expected = _fitted_rates(LegacySolve().run(without_spam).model_data)
    actual = _fitted_rates(LegacySolve().run(with_spam).model_data)

    assert {key: rate for key, rate in actual.items() if key[0] == "LL"} == expected


@pytest.mark.parametrize(
    ("make_path", "fidelity", "match"),
    [
        (lambda gate_set: _spam_path(gate_set, (0, 1)), 0.95, "one qubit at a time"),
        (
            lambda _: Path(start_fragment=[], repeatable_fragment=[], end_fragment=[]),
            0.9,
            "one preparation and one measurement",
        ),
        (lambda gate_set: _spam_path(gate_set, (0,)), 0.0, "SPAM fidelities must be positive"),
    ],
)
def test_legacy_solve_rejects_invalid_spam_rows(gate_set_spam, make_path, fidelity, match):
    fit = Fit()
    fit[AggregatedObservableData] = _spam_data([make_path(gate_set_spam)], np.array([fidelity]))

    with pytest.raises(ValueError, match=match):
        LegacySolve().run(fit)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        # a 2-local M makes -log(F)/2 the sum of X_i and X_iX_j, not the rate of X_i
        ({"M": ["XI", "IX", "XX"]}, "only have single-qubit generators"),
        # preparation noise would be folded into the measurement rate
        ({"P": ["IX"]}, "must have no generators"),
        # the generators fit come from the data, so a narrower model would be silently widened
        ({"M": ["XI"]}, "model does not have"),
        ({"LL": ["XI"]}, "model does not have"),
    ],
)
def test_legacy_solve_rejects_models_it_cannot_fit(gate_set_spam, overrides, match):
    gate_paths = [_pp(gate_set_spam, pauli, pauli) for pauli in ("XI", "ZI")]
    fit = Fit(model=_spam_model(gate_set_spam, **overrides))
    fit[AggregatedObservableData] = _make_aggregated_observable_data(
        gate_paths, np.array([0.9, 0.8])
    ).merge(_spam_data([_spam_path(gate_set_spam, (0,))], np.array([0.97])))

    with pytest.raises(ValueError, match=match):
        LegacySolve().run(fit)
