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

from qiskit_noise_learning.visualizations import (
    exponential_fit_curves_layer,
    model_curves_layer,
    observable_points_layer,
    standard_decay_layers,
)


@pytest.fixture()
def observable_data(make_cz_path, make_observable_data):
    return make_observable_data([(make_cz_path("XI"), 1.0, 0.9, [0, 1, 2])])


@pytest.fixture()
def averaged_data(make_cz_path, make_averaged_data):
    return make_averaged_data([(make_cz_path("XI"), -1, 0.8)])


# --------------------------------------------------------------------------------------------------
# Default styling: exponential fit dashed, model solid (locks the swap made this session)
# --------------------------------------------------------------------------------------------------


def test_exponential_fit_layer_proxy_is_dashed(averaged_data):
    layer = exponential_fit_curves_layer(averaged_data)
    assert layer.proxy["line"]["dash"] == "dash"


def test_model_layer_proxy_is_solid(make_cz_path, make_fidelity_model_data):
    model, model_data = make_fidelity_model_data([make_cz_path("XI")])
    layer = model_curves_layer(model, model_data)
    assert layer.proxy["line"]["dash"] == "solid"


# --------------------------------------------------------------------------------------------------
# Layer.paths
# --------------------------------------------------------------------------------------------------


def test_observable_layer_carries_its_paths(make_cz_path, observable_data):
    layer = observable_points_layer(observable_data)
    assert list(layer.paths) == [make_cz_path("XI")]


def test_model_layer_carries_no_paths(make_cz_path, make_fidelity_model_data):
    model, model_data = make_fidelity_model_data([make_cz_path("XI")])
    layer = model_curves_layer(model, model_data)
    assert layer.paths == ()


# --------------------------------------------------------------------------------------------------
# standard_decay_layers composition
# --------------------------------------------------------------------------------------------------


def test_standard_layers_raw_observable_only(observable_data):
    layers = standard_decay_layers(observable_data=observable_data, observable_type="raw")
    assert [layer.name for layer in layers] == ["Observable points"]


def test_standard_layers_means_observable_only(observable_data):
    layers = standard_decay_layers(observable_data=observable_data, observable_type="means")
    assert [layer.name for layer in layers] == ["Observable means"]


def test_standard_layers_both_observable(observable_data):
    layers = standard_decay_layers(observable_data=observable_data, observable_type="both")
    assert [layer.name for layer in layers] == ["Observable points", "Observable means"]


def test_standard_layers_full_stack(
    observable_data, averaged_data, make_cz_path, make_fidelity_model_data
):
    model, model_data = make_fidelity_model_data([make_cz_path("XI")])
    layers = standard_decay_layers(
        observable_data=observable_data,
        observable_type="raw",
        averaged_data=averaged_data,
        model=model,
        model_data=model_data,
    )
    assert [layer.name for layer in layers] == ["Observable points", "Exponential fit", "Model"]


def test_standard_layers_empty_without_data():
    assert standard_decay_layers() == []


def test_standard_layers_model_needs_both_model_and_data(make_cz_path, make_fidelity_model_data):
    model, _ = make_fidelity_model_data([make_cz_path("XI")])
    # Model without model_data contributes no layer.
    assert standard_decay_layers(model=model) == []


def test_standard_layers_invalid_observable_type_raises(observable_data):
    with pytest.raises(ValueError, match="observable_type"):
        standard_decay_layers(observable_data=observable_data, observable_type="nonsense")
