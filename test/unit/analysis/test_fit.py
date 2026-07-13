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

from qiskit_noise_learning.analysis import Fit
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
import plotly.graph_objects as go
import pytest

from qiskit_noise_learning.analysis import Fit
from qiskit_noise_learning.data import AveragedData, ModelData, ObservableData


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
        {"model_prediction": False},
        {"exponential_fit": False},
        {"observable_type": None},
        {"observable_type": "raw", "exponential_fit": False, "model_prediction": False},
    ],
)
def test_plot_toggles_drop_layers_without_error(fit_with_data, kwargs):
    assert isinstance(fit_with_data.plot_qubit_pair_decays([(0, 1)], **kwargs), go.Figure)


def test_plot_without_model_raises():
    fit = Fit(model=None)
    with pytest.raises(ValueError, match="model"):
        fit.plot_qubit_pair_decays([(0, 1)])
