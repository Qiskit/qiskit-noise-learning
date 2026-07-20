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

import plotly.graph_objects as go
import pytest

from qiskit_noise_learning.analysis import Fit
from qiskit_noise_learning.data import AveragedData, ModelData, ObservableData
from qiskit_noise_learning.models import IdentityFidelityModel, LogFidelitySpace
from qiskit_noise_learning.sequences import LogPathMap


def test_accepts_fidelity_model_and_none(gate_set_cz):
    assert Fit(model=IdentityFidelityModel(gate_set_cz)).model is not None
    assert Fit(model=None).model is None


def test_rejects_non_linear_map(gate_set_cz):
    # a gate set is not a LinearMap at all
    with pytest.raises(TypeError, match="fidelity model"):
        Fit(model=gate_set_cz)


def test_rejects_non_fidelity_linear_map(gate_set_cz):
    # a LinearMap whose output space is a LogPathSpace, not a LogFidelitySpace
    with pytest.raises(TypeError, match="fidelity model"):
        Fit(model=LogPathMap(LogFidelitySpace(gate_set_cz)))


@pytest.fixture()
def fit_with_data(make_cz_path, make_observable_data, make_averaged_data, make_fidelity_model_data):
    """A real :class:`~.Fit` carrying observable / averaged / model data for one decay path."""
    p = make_cz_path("XI")
    model, model_data = make_fidelity_model_data([p])
    fit = Fit(model=model)
    fit[ObservableData] = make_observable_data([(p, 1.0, 0.9, [0, 1, 2])])
    fit[AveragedData] = make_averaged_data([(p, -1, 0.9)])
    fit[ModelData] = model_data
    return fit


# --------------------------------------------------------------------------------------------------
# Fit.plot_qubit_pair_decays
# --------------------------------------------------------------------------------------------------


def test_plot_returns_figure(fit_with_data):
    fig = fit_with_data.plot_qubit_pair_decays([(0, 1)])
    assert isinstance(fig, go.Figure)


def test_plot_model_prediction_end_to_end(fit_with_data):
    # Regression: the model-prediction path used to raise a KeyError on SPAM/non-decay paths.
    fig = fit_with_data.plot_qubit_pair_decays(
        [(0, 1)], observable_type="means", exponential_fit=True, model_prediction=True
    )
    assert isinstance(fig, go.Figure)


@pytest.mark.parametrize(
    "kwargs",
    [
        # Each case leaves two layers on and drops exactly one, so the drop is actually exercised.
        {"observable_type": "means", "exponential_fit": True, "model_prediction": False},
        {"observable_type": "means", "exponential_fit": False, "model_prediction": True},
        {"observable_type": None, "exponential_fit": True, "model_prediction": True},
    ],
)
def test_plot_drops_one_layer_without_error(fit_with_data, kwargs):
    assert isinstance(fit_with_data.plot_qubit_pair_decays([(0, 1)], **kwargs), go.Figure)


def test_plot_without_model_raises():
    fit = Fit(model=None)
    with pytest.raises(ValueError, match="model"):
        fit.plot_qubit_pair_decays([(0, 1)])


def test_plot_warns_when_requested_layer_data_absent(make_cz_path, make_fidelity_model_data):
    # A fit with a model but no observable data: requesting observable points should warn, not skip
    # silently.
    p = make_cz_path("XI")
    model, _ = make_fidelity_model_data([p])
    fit = Fit(model=model, paths=[p])
    with pytest.warns(UserWarning, match="observable points requested"):
        fit.plot_qubit_pair_decays([(0, 1)], observable_type="raw")


def test_plot_model_only_uses_fit_paths(make_cz_path, make_fidelity_model_data):
    # No observable/averaged data: the model curve is drawn for the fit's own paths (a decay path
    # on qubits (0, 1)), rather than silently producing an empty figure.
    p = make_cz_path("XI")
    model, model_data = make_fidelity_model_data([p])
    fit = Fit(model=model, paths=[p])
    fit[ModelData] = model_data
    fig = fit.plot_qubit_pair_decays([(0, 1)], model_prediction=True)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0
