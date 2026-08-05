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

from qiskit_noise_learning.visualizations.path_data.primitives import (
    PointSeries,
    _default_fragment_depths,
    plot_path_decay_curves,
    plot_path_scatters,
)


@pytest.fixture
def ax():
    """A bare axes to draw into, with no pyplot state behind it."""
    return Figure().subplots()


def _gid(path, index):
    return f"trace|c0|{path}|l0|{index}"


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


def test_scatters_draws_one_series_per_path(ax):
    points = {
        "a": PointSeries(xs=np.array([0.0, 1.0]), ys=np.array([1.0, 0.5])),
        "b": PointSeries(xs=np.array([0.0]), ys=np.array([0.9])),
    }
    traces = plot_path_scatters(points, ax, gid=_gid)
    assert set(traces) == {_gid("a", 0), _gid("b", 0)}
    assert traces[_gid("a", 0)]["points"] == [[0.0, 1.0], [1.0, 0.5]]


def test_scatters_error_bars_only_when_stds_set(ax):
    points = {
        "with": PointSeries(xs=np.array([0.0]), ys=np.array([1.0]), stds=np.array([0.1])),
        "without": PointSeries(xs=np.array([0.0]), ys=np.array([1.0])),
    }
    traces = plot_path_scatters(points, ax, gid=_gid)
    assert traces[_gid("with", 0)]["stds"] == [0.1]
    assert "stds" not in traces[_gid("without", 0)]


def test_scatters_hover_label_follows_labels(ax):
    points = {"a": PointSeries(xs=np.array([0.0]), ys=np.array([1.0]))}
    traces = plot_path_scatters(points, ax, labels={"a": "series-a"}, gid=_gid)
    assert traces[_gid("a", 0)]["label"] == "series-a"
    # No label -> an empty one rather than a missing key, so the browser need not special-case it.
    traces = plot_path_scatters(points, ax, gid=_gid)
    assert traces[_gid("a", 0)]["label"] == ""


def test_scatters_tags_every_artist_uniquely(ax):
    # Error bars make a series several artists; each becomes an SVG ``id``, so each needs its own.
    points = {"a": PointSeries(xs=np.array([0.0]), ys=np.array([1.0]), stds=np.array([0.1]))}
    plot_path_scatters(points, ax, gid=_gid)
    gids = [artist.get_gid() for artist in ax.get_children() if artist.get_gid()]
    assert len(gids) > 1
    assert len(set(gids)) == len(gids)


def test_scatters_without_gid_draws_but_reports_nothing(ax):
    points = {"a": PointSeries(xs=np.array([0.0]), ys=np.array([1.0]))}
    assert plot_path_scatters(points, ax) == {}
    assert ax.has_data()


def test_scatters_marker_kwargs_override_defaults(ax):
    points = {"a": PointSeries(xs=np.array([0.0]), ys=np.array([1.0]))}
    plot_path_scatters(points, ax, colors={"a": "red"}, marker_kwargs={"marker": "s"})
    (line,) = ax.get_lines()
    assert line.get_marker() == "s"
    assert line.get_color() == "red"


# --------------------------------------------------------------------------------------------------
# plot_path_decay_curves
# --------------------------------------------------------------------------------------------------


def test_decay_curve_values(ax):
    fragment_depths = np.array([0.0, 1.0, 2.0])
    traces = plot_path_decay_curves({"a": 0.5}, {"a": 0.9}, fragment_depths, ax, gid=_gid)
    (line,) = ax.get_lines()
    np.testing.assert_allclose(line.get_ydata(), 0.9 * 0.5**fragment_depths)
    ys = [point[1] for point in traces[_gid("a", 0)]["points"]]
    np.testing.assert_allclose(ys, 0.9 * 0.5**fragment_depths)


def test_decay_curve_line_kwargs_passthrough(ax):
    plot_path_decay_curves({"a": 0.5}, {"a": 1.0}, [0, 1], ax, line_kwargs={"linestyle": ":"})
    (line,) = ax.get_lines()
    assert line.get_linestyle() == ":"


def test_decay_curve_carries_no_stds(ax):
    # A fitted curve is exact at every point it is evaluated at; there is nothing to spread.
    traces = plot_path_decay_curves({"a": 0.5}, {"a": 1.0}, [0, 1], ax, gid=_gid)
    assert "stds" not in traces[_gid("a", 0)]
