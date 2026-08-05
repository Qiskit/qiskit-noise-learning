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

"""Core path-referenced plotting primitives and the PointSeries container.

Each primitive draws one kind of decay data into a single :class:`~matplotlib.axes.Axes` and returns
the hover data for what it drew, keyed by the ``gid`` each artist was tagged with.  Returning that
rather than the axes is what lets a caller assemble an :class:`~.InteractiveFigure`: the ``gid``\\ s
are how the browser finds the artists, and this mapping is how it knows what they mean.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from ...optionals import HAS_MATPLOTLIB
from ...sequences import Path

if TYPE_CHECKING:
    from matplotlib.artist import Artist
    from matplotlib.axes import Axes

#: Marker and line properties every series starts from, before per-path color and caller
#: overrides are merged over them.
_SCATTER_DEFAULTS = {"marker": "o", "linestyle": "none", "markersize": 4, "capsize": 2}
_CURVE_DEFAULTS = {"linestyle": "-", "linewidth": 1.5}

#: How many points along a fitted curve carry a hover readout.  The curve is *drawn* through every
#: fragment depth it is evaluated at -- that is what makes it look smooth -- but its readout has
#: nothing to say that varies from point to point: there is no measurement there, only the fit the
#: curve was generated from.  Sampling the readout at a handful of anchors instead keeps the pointer
#: always near one while dropping a payload that, over a grid of subplots, comes to outweigh the
#: drawing it annotates.
_CURVE_READOUT_ANCHORS = 5


@dataclass(frozen=True, eq=False)
class PointSeries:
    """A series of 2d points.

    Args:
        xs: A 1d array of x values, parallel to ``ys``.
        ys: A 1d array of y values, parallel to ``xs``.
        stds: An optional 1d array of standard deviations aligned with ``xs``.
    """

    xs: np.ndarray
    ys: np.ndarray
    stds: np.ndarray | None = None


def _trace_data(
    label: str | None, xs: np.ndarray, ys: np.ndarray, stds: np.ndarray | None = None
) -> dict[str, Any]:
    """The hover-readout entry for one series, in the shape :class:`~.InteractiveFigure` expects."""
    entry: dict[str, Any] = {
        "label": label or "",
        "points": [[float(x), float(y)] for x, y in zip(xs, ys)],
    }
    if stds is not None:
        entry["stds"] = [float(std) for std in stds]
    return entry


def _tag(artists: Sequence["Artist"], gid: Callable[[int], str] | None) -> None:
    """Tag each artist with its ``gid``, numbering them so no two share one.

    A single series is often several artists -- a marker line plus the segments and caps of its
    error bars -- and matplotlib writes each ``gid`` into the SVG as an ``id``, which must be
    unique.
    """
    if gid is None:
        return
    for index, artist in enumerate(artists):
        artist.set_gid(gid(index))


def _anchor_indices(count: int, wanted: int = _CURVE_READOUT_ANCHORS) -> np.ndarray:
    """Up to ``wanted`` positions spread evenly over a series of ``count`` points.

    A short series keeps all of its points rather than being thinned to fewer than it has, so the
    only series this changes are the dense ones it exists for.
    """
    if count <= 0:
        return np.empty(0, dtype=int)
    return np.unique(np.linspace(0, count - 1, min(wanted, count)).round().astype(int))


def _curve_readout(label: str | None, base: float, intercept: float) -> str:
    """What hovering a fitted decay curve reports.

    The fit itself, rather than the curve's height where the pointer happens to be: an analytic
    curve passes through no measurement, so its coordinates say only where the reader put their
    mouse.
    """
    lines = [] if not label else [label]
    lines.append(f"fit: base = {base:.6g}, intercept = {intercept:.6g}")
    return "\n".join(lines)


def _errorbar_artists(container: Any) -> list["Artist"]:
    """Every drawn artist of an errorbar, as a flat list.

    matplotlib returns these as a nested ``(data_line, caplines, barlinecols)``, and which parts are
    present depends on whether there were error bars to draw at all.
    """
    data_line, caplines, barlinecols = container.lines
    artists: list[Artist] = [] if data_line is None else [data_line]
    artists.extend(caplines)
    artists.extend(barlinecols)
    return artists


@HAS_MATPLOTLIB.require_in_call
def plot_path_scatters(
    points: Mapping[Path, PointSeries],
    ax: "Axes",
    *,
    colors: Mapping[Path, str | None] | None = None,
    labels: Mapping[Path, str | None] | None = None,
    gid: Callable[[Path, int], str] | None = None,
    marker_kwargs: Mapping[str, object] | None = None,
) -> dict[str, dict[str, Any]]:
    """Path-referenced scatter plot of point series.

    Args:
        points: A mapping from path to its :class:`PointSeries`. A series with ``stds`` set is drawn
            with symmetric error bars.
        ax: The axes to draw into.
        colors: An optional mapping from path to a matplotlib color for its markers. A path mapped
            to ``None``, or absent, takes the axes' next cycle color.
        labels: An optional mapping from path to the label its hover readout shows. Legends are
            built by the orchestrators from the resolved key sets, so a label here affects only the
            readout.
        gid: An optional callable ``(path, artist_index) -> gid`` naming each artist drawn, from
            :func:`~.trace_gid`. Without it the artists are untagged and the returned mapping is
            empty, since there is then nothing for the browser to key the hover data to.
        marker_kwargs: Optional matplotlib properties for the drawn series (e.g. ``marker``,
            ``markersize``, ``alpha``), merged over the per-path color and the scatter defaults.

    Returns:
        The hover data for the drawn series, keyed by the ``gid`` of the artist it belongs to.

    Raises:
        ImportError: If ``matplotlib`` is not installed.
    """
    extra = dict(marker_kwargs or {})
    traces: dict[str, dict[str, Any]] = {}
    for path, series in points.items():
        xs = np.asarray(series.xs, dtype=float)
        ys = np.asarray(series.ys, dtype=float)
        stds = None if series.stds is None else np.asarray(series.stds, dtype=float)
        color = colors.get(path) if colors else None

        container = ax.errorbar(xs, ys, yerr=stds, **{**_SCATTER_DEFAULTS, "color": color, **extra})
        artists = _errorbar_artists(container)
        _tag(artists, None if gid is None else lambda index, path=path: gid(path, index))
        if gid is not None and artists:
            label = labels.get(path) if labels else None
            traces[gid(path, 0)] = _trace_data(label, xs, ys, stds)

    return traces


@HAS_MATPLOTLIB.require_in_call
def plot_path_decay_curves(
    bases: Mapping[Path, float],
    intercepts: Mapping[Path, float],
    fragment_depths: Sequence[float] | np.ndarray,
    ax: "Axes",
    *,
    colors: Mapping[Path, str | None] | None = None,
    labels: Mapping[Path, str | None] | None = None,
    gid: Callable[[Path, int], str] | None = None,
    line_kwargs: Mapping[str, object] | None = None,
) -> dict[str, dict[str, Any]]:
    """Plot smooth exponential decay curves ``intercept * base**fragment_depth`` for each path.

    Args:
        bases: A mapping from path to its per-repetition decay base (fidelity).
        intercepts: A mapping from path to its ``fragment_depth=0`` intercept (the SPAM prefactor).
            Must contain every path in ``bases``.
        fragment_depths: The fragment-depth values (x) at which to evaluate the curve.
        ax: The axes to draw into.
        colors: An optional mapping from path to a matplotlib color for its line. A path mapped to
            ``None``, or absent, takes the axes' next cycle color.
        labels: An optional mapping from path to the label its hover readout shows. Legends are
            built by the orchestrators from the resolved key sets, so a label here affects only the
            readout.
        gid: An optional callable ``(path, artist_index) -> gid`` naming each artist drawn, from
            :func:`~.trace_gid`. Without it the artists are untagged and the returned mapping is
            empty, since there is then nothing for the browser to key the hover data to.
        line_kwargs: Optional matplotlib properties for the drawn lines (e.g. ``linestyle``,
            ``linewidth``, ``alpha``), merged over the per-path color and the curve defaults.

    Returns:
        The hover data for the drawn curves, keyed by the ``gid`` of the artist it belongs to. A
        curve is drawn at every value in ``fragment_depths`` but reads out at only a few points
        along it, each reporting the fit the curve came from rather than the curve's height there.

    Raises:
        ImportError: If ``matplotlib`` is not installed.
    """
    extra = dict(line_kwargs or {})
    fragment_depths_arr = np.asarray(fragment_depths, dtype=float)
    traces: dict[str, dict[str, Any]] = {}
    for path, base in bases.items():
        ys = intercepts[path] * base**fragment_depths_arr
        color = colors.get(path) if colors else None

        lines = ax.plot(fragment_depths_arr, ys, **{**_CURVE_DEFAULTS, "color": color, **extra})
        _tag(lines, None if gid is None else lambda index, path=path: gid(path, index))
        if gid is not None and lines:
            label = labels.get(path) if labels else None
            anchors = _anchor_indices(fragment_depths_arr.size)
            entry = _trace_data(label, fragment_depths_arr[anchors], ys[anchors])
            entry["texts"] = [_curve_readout(label, base, intercepts[path])] * anchors.size
            traces[gid(path, 0)] = entry

    return traces


def _default_fragment_depths(
    *point_dicts: Mapping[Path, PointSeries], num: int = 100
) -> np.ndarray:
    """A dense fragment-depth range from ``0`` to the largest fragment depth across the given point
    mappings."""
    max_fragment_depth = 0.0
    found = False
    for points in point_dicts:
        for series in points.values():
            if series.xs.size:
                max_fragment_depth = max(max_fragment_depth, float(np.max(series.xs)))
                found = True
    if not found:
        max_fragment_depth = 10.0
    return np.linspace(0.0, max_fragment_depth, num)
