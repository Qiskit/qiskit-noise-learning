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
from matplotlib.figure import Figure

from qiskit_noise_learning.visualizations.interactive_figure import TokenMap
from qiskit_noise_learning.visualizations.path_data.layers import (
    RenderContext,
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


def _context(paths, **overrides):
    """A render context over ``paths``, with everything an orchestrator would resolve defaulted."""
    fields = {
        "ax": Figure().subplots(),
        "cell": "c0",
        "colors": {},
        "labels": {},
        "groups": {path: index for index, path in enumerate(paths)},
        "fragment_depths": np.array([0.0, 1.0, 2.0]),
        "paths": list(paths),
        "path_tokens": TokenMap("p"),
        "layer_tokens": TokenMap("l"),
    }
    return RenderContext(**{**fields, **overrides})


# --------------------------------------------------------------------------------------------------
# Styling
# --------------------------------------------------------------------------------------------------


def test_curve_layers_draw_the_style_their_proxies_advertise(
    averaged_data, make_cz_path, make_fidelity_model_data
):
    # A legend handle drawn in a style the data is not drawn in tells the reader the opposite of the
    # truth, which is worse than either style being wrong on its own.
    path = make_cz_path("XI")
    model, model_data = make_fidelity_model_data([path])
    for layer in (
        exponential_fit_curves_layer(averaged_data),
        model_curves_layer(model, model_data),
    ):
        context = _context([path])
        layer.render(context)
        (line,) = context.ax.lines
        assert line.get_linestyle() == layer.proxy["linestyle"]


def test_observable_layer_proxy_draws_markers_without_a_line(observable_data):
    layer = observable_points_layer(observable_data)
    assert layer.proxy["marker"] == "o"
    assert layer.proxy["linestyle"] == "none"


# --------------------------------------------------------------------------------------------------
# Layer.key and gid allocation
# --------------------------------------------------------------------------------------------------


def test_layer_key_matches_the_gids_its_render_tags_with(make_cz_path, observable_data):
    # The orchestrator builds the series-legend entry from ``key`` and the browser matches it to the
    # artists by token, so a layer that tagged under some other key would be untoggleable.
    path = make_cz_path("XI")
    layer = observable_points_layer(observable_data)
    context = _context([path])
    traces = layer.render(context)

    layer_token = context.layer_tokens.token(layer.key)
    assert traces
    assert all(gid.split("|")[3] == layer_token for gid in traces)


def test_layer_drawing_nothing_allocates_no_token(make_cz_path, observable_data):
    # A layer with no data for any path in the figure must not reach the legend: an entry that
    # switches nothing is a control the reader can only be misled by.
    layer = observable_points_layer(observable_data)
    context = _context([make_cz_path("XI", spam=False)])
    assert layer.render(context) == {}
    assert len(context.layer_tokens) == 0
    assert len(context.path_tokens) == 0


def test_paths_sharing_a_group_share_a_path_token(make_cz_path, make_observable_data):
    first, second = make_cz_path("XI"), make_cz_path("XX")
    observable_data = make_observable_data([(first, 1.0, 0.9, [0]), (second, 1.0, 0.8, [0])])
    context = _context([first, second], groups={first: "shared", second: "shared"})
    traces = observable_points_layer(observable_data).render(context)

    assert len(context.path_tokens) == 1
    assert len({gid.split("|")[2] for gid in traces}) == 1


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
