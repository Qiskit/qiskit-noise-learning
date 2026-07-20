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

"""Core path-referenced plotting primitives and the PointSeries container."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ...optionals import HAS_PLOTLY
from ...sequences import Path

if TYPE_CHECKING:
    import plotly.graph_objects as go


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


@HAS_PLOTLY.require_in_call
def plot_path_scatters(
    points: Mapping[Path, PointSeries],
    *,
    fig: "go.Figure | None" = None,
    colors: Mapping[Path, str] | None = None,
    labels: Mapping[Path, str] | None = None,
    groups: Mapping[Path, str] | None = None,
    marker_kwargs: Mapping[str, object] | None = None,
    row: int | None = None,
    col: int | None = None,
) -> "go.Figure":
    """Path-referenced scatter plot of point series.

    Args:
        points: A mapping from path to its :class:`PointSeries`. A series with ``stds`` set is drawn
            with symmetric error bars.
        fig: An existing figure to add traces to. If ``None``, a new figure is created.
        colors: An optional mapping from path to a plotly color string for its markers.
        labels: An optional mapping from path to a legend label (the trace ``name``). Paths without
            a label are omitted from the legend.
        groups: An optional mapping from path to a ``legendgroup`` key, controlling which traces
            share a legend entry and toggle together. Defaults to ``labels`` (each label is its own
            group); pass this to make the grouping identity differ from the displayed label.
        marker_kwargs: Optional properties for the traces' ``marker`` (e.g. ``symbol``, ``size``,
            ``opacity``), merged over the per-path color.
        row: The subplot row to add traces to (1-indexed), for figures created with subplots.
        col: The subplot column to add traces to (1-indexed), for figures created with subplots.

    Returns:
        The figure with the added traces.
    """
    import plotly.graph_objects as go

    if fig is None:
        fig = go.Figure()

    marker_extra = marker_kwargs or {}
    for path, series in points.items():
        label = labels.get(path) if labels else None
        group = groups.get(path) if groups else label
        color = colors.get(path) if colors else None
        error_y = None
        if series.stds is not None:
            error_y = {"type": "data", "array": np.asarray(series.stds), "visible": True}
        fig.add_trace(
            go.Scatter(
                x=np.asarray(series.xs),
                y=np.asarray(series.ys),
                mode="markers",
                name=label,
                legendgroup=group,
                showlegend=label is not None,
                marker={"color": color, **marker_extra},
                error_y=error_y,
            ),
            row=row,
            col=col,
        )

    return fig


@HAS_PLOTLY.require_in_call
def plot_path_decay_curves(
    bases: Mapping[Path, float],
    intercepts: Mapping[Path, float],
    depths: Sequence[float] | np.ndarray,
    *,
    fig: "go.Figure | None" = None,
    colors: Mapping[Path, str] | None = None,
    labels: Mapping[Path, str] | None = None,
    groups: Mapping[Path, str] | None = None,
    line_kwargs: Mapping[str, object] | None = None,
    row: int | None = None,
    col: int | None = None,
) -> "go.Figure":
    """Plot smooth exponential decay curves ``intercept * base**depth`` for each path.

    Args:
        bases: A mapping from path to its per-repetition decay base (fidelity).
        intercepts: A mapping from path to its ``depth=0`` intercept (the SPAM prefactor). Must
            contain every path in ``bases``.
        depths: The depth values (x) at which to evaluate the curve.
        fig: An existing figure to add traces to. If ``None``, a new figure is created.
        colors: An optional mapping from path to a plotly color string for its line.
        labels: An optional mapping from path to a legend label (the trace ``name``). Paths without
            a label are omitted from the legend.
        groups: An optional mapping from path to a ``legendgroup`` key, controlling which traces
            share a legend entry and toggle together. Defaults to ``labels`` (each label is its own
            group); pass this to make the grouping identity differ from the displayed label.
        line_kwargs: Optional properties for the traces' ``line`` (e.g. ``dash``, ``width``), merged
            over the per-path color.
        row: The subplot row to add traces to (1-indexed), for figures created with subplots.
        col: The subplot column to add traces to (1-indexed), for figures created with subplots.

    Returns:
        The figure with the added traces.
    """
    import plotly.graph_objects as go

    if fig is None:
        fig = go.Figure()

    line_extra = line_kwargs or {}
    depths_arr = np.asarray(depths, dtype=float)
    for path, base in bases.items():
        label = labels.get(path) if labels else None
        group = groups.get(path) if groups else label
        color = colors.get(path) if colors else None
        y = intercepts[path] * base**depths_arr
        fig.add_trace(
            go.Scatter(
                x=depths_arr,
                y=y,
                mode="lines",
                name=label,
                legendgroup=group,
                showlegend=label is not None,
                line={"color": color, **line_extra},
            ),
            row=row,
            col=col,
        )

    return fig


def _default_depths(*point_dicts: Mapping[Path, PointSeries], num: int = 100) -> np.ndarray:
    """A dense depth range from ``0`` to the largest depth across the given point mappings."""
    max_depth = 0.0
    found = False
    for points in point_dicts:
        for series in points.values():
            if series.xs.size:
                max_depth = max(max_depth, float(np.max(series.xs)))
                found = True
    if not found:
        max_depth = 10.0
    return np.linspace(0.0, max_depth, num)
