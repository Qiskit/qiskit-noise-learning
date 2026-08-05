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

import json
import re

import pytest

from qiskit_noise_learning.sequences import Path
from qiskit_noise_learning.visualizations.interactive_figure import InteractiveFigure
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


def _subplot_titles(figure):
    """The subplot titles of a figure, in axes order."""
    return [ax.get_title() for ax in figure.figure.axes]


def _svg_ids(figure, prefix):
    """The ``id`` attributes of the rendered SVG that start with ``prefix``, in document order.

    Read from the SVG rather than from the artists, since the ``id`` is what the browser matches on
    and an artist whose ``gid`` never made it through the backend would toggle nothing.
    """
    return [
        found for found in re.findall(r'id="([^"]+)"', figure.to_svg()) if found.startswith(prefix)
    ]


def _sidecar(figure):
    """The figure's JSON sidecar, as the browser receives it."""
    payload = re.search(
        r'class="qnl-figure-data">(.*?)</script>', figure.to_html(), re.DOTALL
    ).group(1)
    return json.loads(payload)


def _key_tokens(figure, dimension):
    """The tokens the given legend offers, taken from the rendered SVG."""
    return {
        found.split("|")[2]
        for found in _svg_ids(figure, "key|")
        if found.split("|")[1] == dimension
    }


def _trace_tokens(figure):
    """The ``(path_token, layer_token)`` pairs the rendered data artists carry."""
    return {tuple(found.split("|")[2:4]) for found in _svg_ids(figure, "trace|")}


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


def test_overlay_returns_interactive_figure(make_cz_path, make_observable_data):
    obs = make_observable_data([(make_cz_path("XI"), 1.0, 0.9, [0, 1, 2])])
    layers = standard_decay_layers(observable_data=obs, observable_type="raw")
    figure = plot_path_overlay(layers)
    assert isinstance(figure, InteractiveFigure)
    assert figure.figure.axes[0].has_data()


def test_overlay_default_index_labels_without_gate_set(make_cz_path, make_observable_data):
    obs = make_observable_data([(make_cz_path("XI"), 1.0, 0.9, [0])])
    figure = plot_path_overlay([observable_points_layer(obs)])
    # No gate set -> labels fall back to positional index strings.
    assert {trace["label"] for trace in _sidecar(figure)["traces"].values()} == {"0"}


def test_overlay_into_a_supplied_axes_still_gets_its_legends(make_cz_path, make_observable_data):
    # Without them the caller would have curves that can be hidden and no control to restore them.
    from matplotlib.figure import Figure

    obs = make_observable_data([(make_cz_path("XI"), 1.0, 0.9, [0])])
    ax = Figure().subplots()
    figure = plot_path_overlay([observable_points_layer(obs)], ax=ax)
    assert figure.figure is ax.figure
    assert len(ax.figure.legends) == 2


def test_overlay_axis_labels(make_cz_path, make_observable_data):
    obs = make_observable_data([(make_cz_path("XI"), 1.0, 0.9, [0])])
    ax = plot_path_overlay([observable_points_layer(obs)]).figure.axes[0]
    assert ax.get_xlabel() == "fragment_depth"
    assert ax.get_ylabel() == "observable"


# --------------------------------------------------------------------------------------------------
# plot_path_grid_overlay
# --------------------------------------------------------------------------------------------------


def test_grid_default_subplot_title_is_str_key(make_cz_path, make_observable_data):
    p = make_cz_path("XI")
    obs = make_observable_data([(p, 1.0, 0.9, [0, 1])])
    figure = plot_path_grid_overlay({(0, 1): [p]}, [observable_points_layer(obs)])
    assert "(0, 1)" in _subplot_titles(figure)


def test_grid_group_title_callable(make_cz_path, make_observable_data):
    p = make_cz_path("XI")
    obs = make_observable_data([(p, 1.0, 0.9, [0, 1])])
    figure = plot_path_grid_overlay(
        {(0, 1): [p]}, [observable_points_layer(obs)], group_title=lambda key: f"pair-{key}"
    )
    assert "pair-(0, 1)" in _subplot_titles(figure)


def test_grid_blanks_cells_past_the_last_group(make_cz_path, make_observable_data):
    p = make_cz_path("XI")
    obs = make_observable_data([(p, 1.0, 0.9, [0, 1])])
    figure = plot_path_grid_overlay(
        {(0, 1): [p], (1, 2): [p]}, [observable_points_layer(obs)], num_cols=3
    )
    # Three columns for two groups: the spare frame is switched off rather than left looking empty.
    assert [ax.get_visible() and ax.axison for ax in figure.figure.axes] == [True, True, False]


def test_grid_series_key_shares_one_entry_across_cells(make_cz_path, make_observable_data):
    first, second = make_cz_path("XI"), make_cz_path("XX")
    obs = make_observable_data([(first, 1.0, 0.9, [0]), (second, 1.0, 0.8, [0])])
    figure = plot_path_grid_overlay(
        {"a": [first], "b": [second]},
        [observable_points_layer(obs)],
        series_key=lambda path, key: "together",
    )
    # One key across both cells -> one path-legend entry, and one click reaches both subplots.
    assert len(_key_tokens(figure, "path")) == 1
    assert len({tokens[0] for tokens in _trace_tokens(figure)}) == 1


def test_grid_paths_are_numbered_globally_without_a_gate_set(make_cz_path, make_observable_data):
    # Cell-local numbering would give two unrelated paths the same label, hence one legend entry.
    first, second = make_cz_path("XI"), make_cz_path("XX")
    obs = make_observable_data([(first, 1.0, 0.9, [0]), (second, 1.0, 0.8, [0])])
    figure = plot_path_grid_overlay({"a": [first], "b": [second]}, [observable_points_layer(obs)])
    assert {trace["label"] for trace in _sidecar(figure)["traces"].values()} == {"0", "1"}
    assert len(_key_tokens(figure, "path")) == 2


# --------------------------------------------------------------------------------------------------
# Structural invariants of the interactive output
# --------------------------------------------------------------------------------------------------


@pytest.fixture()
def pair_grid(make_cz_path, make_observable_data, make_averaged_data, make_fidelity_model_data):
    """A multi-cell, multi-layer, multi-path figure -- the shape the invariants have to hold on."""
    paths = [make_cz_path("XI"), make_cz_path("XX")]
    obs = make_observable_data([(path, 1.0, 0.9, [0, 1, 2]) for path in paths])
    averaged = make_averaged_data(
        [(path, fragment_depth, 0.8) for path in paths for fragment_depth in (-1, 0, 1)]
    )
    model, model_data = make_fidelity_model_data(paths)
    return plot_qubit_pair_decays(
        [(0, 1), (1, 2)],
        observable_data=obs,
        observable_type="both",
        averaged_data=averaged,
        model=model,
        model_data=model_data,
        gate_set=model.gate_set,
    )


def test_trace_ids_are_unique(pair_grid):
    # They become SVG ``id`` attributes; a duplicate is invalid markup and hides the wrong artist.
    ids = _svg_ids(pair_grid, "trace|")
    assert len(ids) > 1
    assert len(set(ids)) == len(ids)


def test_every_trace_resolves_to_both_legends(pair_grid):
    path_tokens, layer_tokens = _key_tokens(pair_grid, "path"), _key_tokens(pair_grid, "layer")
    assert path_tokens and layer_tokens
    for path_token, layer_token in _trace_tokens(pair_grid):
        assert path_token in path_tokens
        assert layer_token in layer_tokens


def test_no_orphan_legend_entries(pair_grid):
    # An entry with nothing behind it is a control that appears to do something and does not.
    drawn = _trace_tokens(pair_grid)
    assert _key_tokens(pair_grid, "path") == {tokens[0] for tokens in drawn}
    assert _key_tokens(pair_grid, "layer") == {tokens[1] for tokens in drawn}


def test_both_legend_parts_are_clickable(pair_grid):
    # The handle and the label are separate elements; tagging both is what makes either one work.
    parts = {tuple(found.split("|")[1:]) for found in _svg_ids(pair_grid, "key|")}
    for dimension, token, part in list(parts):
        assert (dimension, token, "handle" if part == "text" else "text") in parts


def test_sidecar_covers_the_drawn_traces(pair_grid):
    sidecar = _sidecar(pair_grid)
    assert set(sidecar["cells"]) == {"c0", "c1"}
    for gid, trace in sidecar["traces"].items():
        assert gid in _svg_ids(pair_grid, "trace|")
        assert trace["points"]


def test_legends_do_not_overlap_the_subplots_or_each_other(pair_grid):
    # Both sit outside the axes, which is the only reason a legend of a dozen formulas can be shown
    # at all. Nothing in a rendered image reports this, so it is checked here.
    figure = pair_grid.figure
    figure.draw_without_rendering()
    legends = [legend.get_window_extent() for legend in figure.legends]
    assert len(legends) == 2
    assert not legends[0].overlaps(legends[1])
    for ax in figure.axes:
        if ax.axison:
            for legend in legends:
                assert not legend.overlaps(ax.get_window_extent())


# --------------------------------------------------------------------------------------------------
# plot_qubit_pair_decays
# --------------------------------------------------------------------------------------------------


def test_pair_decays_requires_gate_set(make_cz_path, make_averaged_data):
    averaged = make_averaged_data([(make_cz_path("XI"), -1, 0.8)])
    with pytest.raises(ValueError, match="gate_set"):
        plot_qubit_pair_decays([(0, 1)], averaged_data=averaged)


def test_pair_decays_returns_interactive_figure(make_cz_path, make_averaged_data, gate_set_cz):
    averaged = make_averaged_data([(make_cz_path("XI"), -1, 0.8)])
    figure = plot_qubit_pair_decays([(0, 1)], averaged_data=averaged, gate_set=gate_set_cz)
    assert isinstance(figure, InteractiveFigure)


def test_pair_decays_subplot_title_uses_placeholders(make_cz_path, make_averaged_data, gate_set_cz):
    averaged = make_averaged_data([(make_cz_path("XI"), -1, 0.8)])
    figure = plot_qubit_pair_decays([(0, 1)], averaged_data=averaged, gate_set=gate_set_cz)
    assert "(i, j) = (0, 1)" in _subplot_titles(figure)


def test_pair_decays_custom_placeholders(make_cz_path, make_averaged_data, gate_set_cz):
    averaged = make_averaged_data([(make_cz_path("XI"), -1, 0.8)])
    figure = plot_qubit_pair_decays(
        [(0, 1)], averaged_data=averaged, gate_set=gate_set_cz, placeholders=("a", "b")
    )
    assert "(a, b) = (0, 1)" in _subplot_titles(figure)


def test_pair_decays_pair_order_does_not_matter(make_cz_path, make_averaged_data, gate_set_cz):
    # A pair names a subplot and is unordered: reversing it is the same subplot, drawn the same way.
    averaged = make_averaged_data([(make_cz_path("XI"), -1, 0.8)])
    forwards = plot_qubit_pair_decays([(0, 1)], averaged_data=averaged, gate_set=gate_set_cz)
    backwards = plot_qubit_pair_decays([(1, 0)], averaged_data=averaged, gate_set=gate_set_cz)
    assert _subplot_titles(backwards) == _subplot_titles(forwards)
    assert len(_sidecar(backwards)["cells"]) == len(_sidecar(forwards)["cells"])


@pytest.mark.parametrize("pairs", [[(0, 1), (0, 1)], [(0, 1), (1, 0)]])
def test_pair_decays_rejects_a_repeated_pair(pairs, make_cz_path, make_averaged_data, gate_set_cz):
    # Two requests for one subplot would silently produce fewer subplots than the caller asked for,
    # which is indistinguishable from a data problem.
    averaged = make_averaged_data([(make_cz_path("XI"), -1, 0.8)])
    with pytest.raises(ValueError, match="Duplicate qubit pair"):
        plot_qubit_pair_decays(pairs, averaged_data=averaged, gate_set=gate_set_cz)


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
    figure = plot_qubit_pair_decays(
        [(0, 1)],
        averaged_data=averaged,
        model=model,
        model_data=model_data,
        gate_set=model.gate_set,
    )
    assert isinstance(figure, InteractiveFigure)


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
    assert _sidecar(on_pair)["traces"]
    assert _sidecar(off_pair)["traces"] == {}


def test_pair_decays_model_only_with_explicit_paths(make_cz_path, make_fidelity_model_data):
    # No observable/averaged data: an explicit ``paths`` lets the model curve be drawn on its own.
    p = make_cz_path("XI")
    model, model_data = make_fidelity_model_data([p])
    figure = plot_qubit_pair_decays(
        [(0, 1)], model=model, model_data=model_data, paths=[p], gate_set=model.gate_set
    )
    assert _sidecar(figure)["traces"]
