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

from qiskit_noise_learning.sequences import Path
from qiskit_noise_learning.visualizations.path_data.layers import (
    observable_points_layer,
    standard_decay_layers,
)
from qiskit_noise_learning.visualizations.path_data.orchestrators import (
    path_labels,
    plot_path_grid_overlay,
    plot_path_overlay,
    plot_qubit_pair_decays,
)


def _subplot_titles(fig):
    """The subplot-title annotation texts of a grid figure (those without an axis reference)."""
    return [ann.text for ann in fig.layout.annotations if ann.xref == "paper"]


# --------------------------------------------------------------------------------------------------
# path_labels
# --------------------------------------------------------------------------------------------------


def test_path_labels_are_math_delimited(make_cz_path, gate_set_cz):
    p = make_cz_path("XI")
    labels = path_labels([p], gate_set_cz)
    assert labels[p].startswith("$") and labels[p].endswith("$")


# --------------------------------------------------------------------------------------------------
# plot_path_overlay
# --------------------------------------------------------------------------------------------------


def test_overlay_returns_figure(make_cz_path, make_observable_data):
    obs = make_observable_data([(make_cz_path("XI"), 1.0, 0.9, [0, 1, 2])])
    layers = standard_decay_layers(observable_data=obs, observable_type="raw")
    fig = plot_path_overlay(layers)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 1


def test_overlay_default_index_labels_without_gate_set(make_cz_path, make_observable_data):
    obs = make_observable_data([(make_cz_path("XI"), 1.0, 0.9, [0])])
    layers = [observable_points_layer(obs)]
    fig = plot_path_overlay(layers)
    # No gate set -> labels fall back to positional index strings.
    assert any(trace.name == "0" for trace in fig.data)


# --------------------------------------------------------------------------------------------------
# plot_path_grid_overlay: group_title
# --------------------------------------------------------------------------------------------------


def test_grid_default_subplot_title_is_str_key(make_cz_path, make_observable_data):
    p = make_cz_path("XI")
    obs = make_observable_data([(p, 1.0, 0.9, [0, 1])])
    fig = plot_path_grid_overlay({(0, 1): [p]}, [observable_points_layer(obs)])
    assert "(0, 1)" in _subplot_titles(fig)


def test_grid_group_title_callable(make_cz_path, make_observable_data):
    p = make_cz_path("XI")
    obs = make_observable_data([(p, 1.0, 0.9, [0, 1])])
    fig = plot_path_grid_overlay(
        {(0, 1): [p]}, [observable_points_layer(obs)], group_title=lambda key: f"pair-{key}"
    )
    assert "pair-(0, 1)" in _subplot_titles(fig)


# --------------------------------------------------------------------------------------------------
# plot_qubit_pair_decays
# --------------------------------------------------------------------------------------------------


def test_pair_decays_requires_gate_set(make_cz_path, make_averaged_data):
    averaged = make_averaged_data([(make_cz_path("XI"), -1, 0.8)])
    with pytest.raises(ValueError, match="gate_set"):
        plot_qubit_pair_decays([(0, 1)], averaged_data=averaged)


def test_pair_decays_returns_figure(make_cz_path, make_averaged_data, gate_set_cz):
    averaged = make_averaged_data([(make_cz_path("XI"), -1, 0.8)])
    fig = plot_qubit_pair_decays([(0, 1)], averaged_data=averaged, gate_set=gate_set_cz)
    assert isinstance(fig, go.Figure)


def test_pair_decays_subplot_title_uses_placeholders(make_cz_path, make_averaged_data, gate_set_cz):
    averaged = make_averaged_data([(make_cz_path("XI"), -1, 0.8)])
    fig = plot_qubit_pair_decays([(0, 1)], averaged_data=averaged, gate_set=gate_set_cz)
    assert "(i, j) = (0, 1)" in _subplot_titles(fig)


def test_pair_decays_custom_placeholders(make_cz_path, make_averaged_data, gate_set_cz):
    averaged = make_averaged_data([(make_cz_path("XI"), -1, 0.8)])
    fig = plot_qubit_pair_decays(
        [(0, 1)], averaged_data=averaged, gate_set=gate_set_cz, placeholders=("a", "b")
    )
    assert "(a, b) = (0, 1)" in _subplot_titles(fig)


def test_pair_decays_filters_non_decay_paths_for_model_prediction(
    make_cz_path, make_averaged_data, make_fidelity_model_data
):
    # A real decay path plus a non-decay (empty repeatable) SPAM-like path sharing the pair.
    decay = make_cz_path("XI")
    non_decay = Path(
        start_fragment=decay.start_fragment, repeatable_fragment=[], end_fragment=decay.end_fragment
    )
    averaged = make_averaged_data([(decay, -1, 0.8), (non_decay, -1, 0.9)])
    model, model_data = make_fidelity_model_data([decay])

    # Without filtering the non-decay path, model_curves would raise; a clean render proves the
    # non-decay path was dropped before reaching the model-curve layer.
    fig = plot_qubit_pair_decays(
        [(0, 1)],
        averaged_data=averaged,
        model=model,
        model_data=model_data,
        gate_set=model.gate_set,
    )
    assert isinstance(fig, go.Figure)


def test_pair_decays_assigns_empty_start_fragment_path_via_transition(
    make_cz_path, make_averaged_data, gate_set_cz
):
    # A decay path with no start fragment still acts on the CZ's qubits through its transition
    # Paulis, so it is assigned to pair (0, 1) and dropped from an unrelated pair.
    p = make_cz_path("XI", spam=False)
    assert not p.start_fragment
    averaged = make_averaged_data([(p, -1, 0.8)])
    on_pair = plot_qubit_pair_decays([(0, 1)], averaged_data=averaged, gate_set=gate_set_cz)
    off_pair = plot_qubit_pair_decays([(2, 3)], averaged_data=averaged, gate_set=gate_set_cz)
    assert len(on_pair.data) > 0
    assert len(off_pair.data) == 0


def test_pair_decays_model_only_with_explicit_paths(make_cz_path, make_fidelity_model_data):
    # No observable/averaged data: an explicit ``paths`` lets the model curve be drawn on its own.
    p = make_cz_path("XI")
    model, model_data = make_fidelity_model_data([p])
    fig = plot_qubit_pair_decays(
        [(0, 1)], model=model, model_data=model_data, paths=[p], gate_set=model.gate_set
    )
    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0
