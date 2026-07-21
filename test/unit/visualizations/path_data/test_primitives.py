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
import plotly.graph_objects as go

from qiskit_noise_learning.visualizations.path_data.primitives import (
    PointSeries,
    _default_fragment_depths,
    plot_path_decay_curves,
    plot_path_scatters,
)

# --------------------------------------------------------------------------------------------------
# PointSeries
# --------------------------------------------------------------------------------------------------


def test_point_series_fields():
    series = PointSeries(xs=np.array([0.0, 1.0]), ys=np.array([1.0, 0.5]))
    assert list(series.xs) == [0.0, 1.0]
    assert list(series.ys) == [1.0, 0.5]
    assert series.stds is None


# --------------------------------------------------------------------------------------------------
# _default_fragment_depths
# --------------------------------------------------------------------------------------------------


def test_default_fragment_depths_empty_fallback():
    fragment_depths = _default_fragment_depths()
    assert fragment_depths[0] == 0.0
    assert fragment_depths[-1] == 10.0
    assert len(fragment_depths) == 100


def test_default_fragment_depths_respects_num():
    assert len(_default_fragment_depths(num=25)) == 25


def test_default_fragment_depths_max_across_dicts():
    d1 = {"a": PointSeries(xs=np.array([0.0, 4.0]), ys=np.array([1.0, 0.5]))}
    d2 = {"b": PointSeries(xs=np.array([0.0, 9.0]), ys=np.array([1.0, 0.5]))}
    fragment_depths = _default_fragment_depths(d1, d2)
    assert fragment_depths[-1] == 9.0


def test_default_fragment_depths_ignores_empty_series():
    d = {"a": PointSeries(xs=np.array([]), ys=np.array([]))}
    assert _default_fragment_depths(d)[-1] == 10.0


# --------------------------------------------------------------------------------------------------
# plot_path_scatters
# --------------------------------------------------------------------------------------------------


def test_scatters_returns_figure_one_trace_per_path():
    points = {
        "a": PointSeries(xs=np.array([0.0, 1.0]), ys=np.array([1.0, 0.5])),
        "b": PointSeries(xs=np.array([0.0]), ys=np.array([0.9])),
    }
    fig = plot_path_scatters(points)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2


def test_scatters_error_bars_only_when_stds_set():
    points = {
        "with": PointSeries(xs=np.array([0.0]), ys=np.array([1.0]), stds=np.array([0.1])),
        "without": PointSeries(xs=np.array([0.0]), ys=np.array([1.0])),
    }
    fig = plot_path_scatters(points, labels={"with": "with", "without": "without"})
    by_name = {trace.name: trace for trace in fig.data}
    assert by_name["with"].error_y.array is not None
    assert by_name["without"].error_y.array is None


def test_scatters_legend_follows_labels():
    points = {"a": PointSeries(xs=np.array([0.0]), ys=np.array([1.0]))}
    fig = plot_path_scatters(points, labels={"a": "series-a"})
    assert fig.data[0].name == "series-a"
    assert fig.data[0].showlegend is True
    # No label -> omitted from legend.
    fig2 = plot_path_scatters(points)
    assert fig2.data[0].showlegend is False


# --------------------------------------------------------------------------------------------------
# plot_path_decay_curves
# --------------------------------------------------------------------------------------------------


def test_decay_curve_values():
    fragment_depths = np.array([0.0, 1.0, 2.0])
    fig = plot_path_decay_curves(
        bases={"a": 0.5}, intercepts={"a": 0.9}, fragment_depths=fragment_depths
    )
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    np.testing.assert_allclose(fig.data[0].y, 0.9 * 0.5**fragment_depths)


def test_decay_curve_line_kwargs_passthrough():
    fig = plot_path_decay_curves(
        bases={"a": 0.5}, intercepts={"a": 1.0}, fragment_depths=[0, 1], line_kwargs={"dash": "dot"}
    )
    assert fig.data[0].line.dash == "dot"
