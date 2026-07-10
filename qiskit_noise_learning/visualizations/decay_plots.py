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

"""Decay-curve plotting for observable, averaged, and model data."""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

from ..optionals import HAS_PLOTLY
from .fidelity_math_labels import path_math_label

if TYPE_CHECKING:
    import plotly.graph_objects as go

    from ..data import AveragedData, ModelData, ObservableData
    from ..gate_sets import GateSet
    from ..math import LinearMap
    from ..sequences import Path


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


# Default per-layer marker symbols / line dashes, shared by the per-type plotters and by the layer
# metadata used to build the symbol legend (so an overridden symbol/dash is reflected there too).
_OBSERVABLE_POINTS_SYMBOL = "circle"
_AVERAGED_POINTS_SYMBOL = "x"
_FIT_DASH = "solid"
_MODEL_DASH = "dash"

# Neutral color for the symbol-legend proxy traces (symbols there denote layer type, not path).
_SYMBOL_LEGEND_COLOR = "rgb(120,120,120)"


# --------------------------------------------------------------------------------------------------
# Core plotting primitives
# --------------------------------------------------------------------------------------------------


@HAS_PLOTLY.require_in_call
def plot_path_scatters(
    points: Mapping[Path, PointSeries],
    *,
    fig: go.Figure | None = None,
    colors: Mapping[Path, str] | None = None,
    labels: Mapping[Path, str] | None = None,
    groups: Mapping[Path, str] | None = None,
    marker_kwargs: Mapping[str, object] | None = None,
    row: int | None = None,
    col: int | None = None,
) -> go.Figure:
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
    fig: go.Figure | None = None,
    colors: Mapping[Path, str] | None = None,
    labels: Mapping[Path, str] | None = None,
    groups: Mapping[Path, str] | None = None,
    line_kwargs: Mapping[str, object] | None = None,
    row: int | None = None,
    col: int | None = None,
) -> go.Figure:
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


# --------------------------------------------------------------------------------------------------
# Adapters: extract inputs to core plotters from data classes
# --------------------------------------------------------------------------------------------------


def averaged_data_curves(
    averaged_data: AveragedData,
    paths: Iterable[Path] | None = None,
) -> tuple[dict[Path, float], dict[Path, float]]:
    """Extract exponential decay parameters from averaged data.

    Extracts the ``base`` and ``intercept`` for entries with ``depth == -1``, for use in
    :func:`plot_path_decay_curves`. The ``intercept`` is drawn from ``metadata["spam_fidelity"]``,
    and defaults to ``1.0`` if absent.

    Args:
        averaged_data: The averaged data to extract from.
        paths: Optional paths to restrict extraction to. Defaults to all paths in the data.

    Returns:
        A ``(bases, intercepts)`` pair of mappings from path to float.
    """
    dataset = averaged_data.dataset
    wanted = set(paths) if paths is not None else None
    mask = dataset["depth"].data == -1
    entry_paths = dataset["unbound_path"].data[mask]
    observables = dataset["observables"].data[mask]
    metadata = dataset["metadata"].data[mask]

    bases: dict[Path, float] = {}
    intercepts: dict[Path, float] = {}
    for path, fidelity, meta in zip(entry_paths, observables, metadata):
        if wanted is not None and path not in wanted:
            continue
        bases[path] = float(fidelity)
        intercepts[path] = float(meta["spam_fidelity"]) if meta and "spam_fidelity" in meta else 1.0

    return bases, intercepts


def averaged_data_points(
    averaged_data: AveragedData,
    paths: Iterable[Path] | None = None,
) -> dict[Path, PointSeries]:
    """Extract point series data for the bound paths in an average data instance.

    Args:
        averaged_data: The averaged data to extract from.
        paths: Optional paths to restrict extraction to. Defaults to all paths in the data.

    Returns:
        A mapping from path to its :class:`PointSeries` (with ``stds`` populated).
    """
    dataset = averaged_data.dataset
    wanted = set(paths) if paths is not None else None
    mask = dataset["depth"].data >= 0
    entry_paths = dataset["unbound_path"].data[mask]
    depths = dataset["depth"].data[mask]
    observables = dataset["observables"].data[mask]
    stds = dataset["std"].data[mask]

    result: dict[Path, PointSeries] = {}
    for path in dict.fromkeys(entry_paths):
        if wanted is not None and path not in wanted:
            continue
        path_mask = np.array([other == path for other in entry_paths])
        result[path] = PointSeries(
            xs=depths[path_mask].astype(float),
            ys=observables[path_mask].astype(float),
            stds=stds[path_mask].astype(float),
        )

    return result


def observable_data_points(
    observable_data: ObservableData,
    paths: Iterable[Path] | None = None,
) -> dict[Path, PointSeries]:
    """Extract raw per-randomization decay points from observable data.

    Flattens the ``randomization`` axis (dropping ``nan`` raggedness padding) so each retained
    randomization becomes its own point.

    Args:
        observable_data: The observable data to extract from.
        paths: Optional paths to restrict extraction to. Defaults to all paths in the data.

    Returns:
        A mapping from path to its :class:`PointSeries` (with ``stds`` unset).
    """
    dataset = observable_data.dataset
    wanted = set(paths) if paths is not None else None
    entry_paths = dataset["unbound_path"].data
    depths = dataset["depth"].data
    observables = dataset["observables"].data

    result: dict[Path, PointSeries] = {}
    for path in dict.fromkeys(entry_paths):
        if wanted is not None and path not in wanted:
            continue
        row_mask = np.array([other == path for other in entry_paths])

        flat_depths: list[float] = []
        flat_values: list[float] = []
        for depth, row_values in zip(depths[row_mask], observables[row_mask]):
            valid = row_values[~np.isnan(row_values)]
            flat_depths.extend([float(depth)] * valid.size)
            flat_values.extend(float(value) for value in valid)

        result[path] = PointSeries(
            xs=np.array(flat_depths, dtype=float),
            ys=np.array(flat_values, dtype=float),
        )

    return result


def model_curves(
    model: LinearMap,
    model_data: ModelData,
    paths: Iterable[Path],
) -> tuple[dict[Path, float], dict[Path, float]]:
    """Compute model-predicted decay curve parameters for a set of paths.

    For each path, the base is the product of the fidelities in the repeatable fragment, and the
    intercept is the product of the fidelities in the start and end fragments.

    Args:
        model: A fidelity model, i.e. a :class:`~.LinearMap` whose output space is a
            :class:`~.LogFidelitySpace`.
        model_data: The fitted model parameters.
        paths: The paths to predict decays for.

    Returns:
        A ``(bases, intercepts)`` pair of mappings from path to float.

    Raises:
        ValueError: If ``model`` is not a fidelity model.
    """
    from ..models import LogPathMap, is_fidelity_model

    if not is_fidelity_model(model):
        raise ValueError(
            "model must be a fidelity model (its output space must be a LogFidelitySpace)."
        )

    paths = list(paths)
    rates = dict(
        zip(model_data.dataset["parameter"].data, model_data.dataset["parameter_values"].data)
    )
    path_map = LogPathMap(model.output_space) @ model

    unbound = [path.without_depth() for path in paths]
    depth_zero = [path.bind_at(0) for path in paths]
    log_fidelities = path_map.projected_output(unbound, rates)
    log_intercepts = path_map.projected_output(depth_zero, rates)

    bases = {path: float(np.exp(-log_fidelities[u])) for path, u in zip(paths, unbound)}
    intercepts = {path: float(np.exp(-log_intercepts[z])) for path, z in zip(paths, depth_zero)}
    return bases, intercepts


# --------------------------------------------------------------------------------------------------
# Labels and coordination helpers
# --------------------------------------------------------------------------------------------------


def path_labels(
    paths: Iterable[Path],
    gate_set: GateSet,
    *,
    style: Literal["transition", "formula"] = "transition",
    noise_site: Mapping[str, Literal["before", "after"]] | None = None,
    repeatable_only: bool = True,
) -> dict[Path, str]:
    """Build math-mode LaTeX legend labels for paths via :func:`~.path_math_label`.

    Args:
        paths: The paths to label.
        gate_set: The gate set the paths' fidelity indices belong to.
        style: The :func:`~.path_math_label` style, ``"formula"`` or ``"transition"``.
        noise_site: An optional noise-site mapping forwarded to :func:`~.path_math_label`.
        repeatable_only: Whether to label only the repeatable fragment (the decaying part).

    Returns:
        A mapping from path to a ``$...$``-delimited LaTeX label.
    """
    return {
        path: "$"
        + path_math_label(
            gate_set,
            path,
            style=style,
            noise_site=noise_site,
            repeatable_only=repeatable_only,
        )
        + "$"
        for path in paths
    }


def _palette() -> list[str]:
    """Color palette for assigning default colors."""
    import plotly.colors as pc

    return pc.qualitative.Plotly


def _colors_by_group(
    groups: Mapping[Path, Hashable | None],
    overrides: Mapping[Path, str] | None,
) -> dict[Path, str | None]:
    """Assign a color per path, keyed by group so paths in the same group share a color."""
    palette = _palette()
    group_to_color: dict[Hashable, str] = {}
    result: dict[Path, str | None] = {}
    for path, group in groups.items():
        if overrides and path in overrides:
            result[path] = overrides[path]
        elif group is None:
            result[path] = None
        else:
            if group not in group_to_color:
                group_to_color[group] = palette[len(group_to_color) % len(palette)]
            result[path] = group_to_color[group]
    return result


def _dedupe_legend(fig: go.Figure) -> None:
    """Show only the first trace of each ``legendgroup`` in the legend."""
    seen: set[str] = set()
    for trace in fig.data:
        group = trace.legendgroup
        if group is None:
            continue
        if group in seen:
            trace.showlegend = False
        else:
            seen.add(group)
            trace.showlegend = True


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


def _resolve_gate_set(gate_set: GateSet | None, model: LinearMap | None) -> GateSet | None:
    """The gate set to label with: the explicit one, else the model's."""
    if gate_set is not None:
        return gate_set
    if model is not None and hasattr(model, "gate_set"):
        return model.gate_set
    return None


def _dataset_paths(*datas: AveragedData | ObservableData | None) -> list[Path]:
    """The unique unbound paths across the given data objects, in first-seen order."""
    seen: dict[Path, None] = {}
    for data in datas:
        if data is not None:
            for path in data.dataset["unbound_path"].data:
                seen.setdefault(path, None)
    return list(seen)


def _dataset_max_depth(*datas: AveragedData | ObservableData | None) -> int | None:
    """The largest non-negative ``depth`` across the given data objects, or ``None`` if none."""
    max_depth: int | None = None
    for data in datas:
        if data is not None:
            depths = data.dataset["depth"].data
            nonneg = depths[depths >= 0]
            if nonneg.size:
                found = int(nonneg.max())
                max_depth = found if max_depth is None else max(max_depth, found)
    return max_depth


# --------------------------------------------------------------------------------------------------
# Per-data-type plotters
# --------------------------------------------------------------------------------------------------


@HAS_PLOTLY.require_in_call
def plot_observable_points(
    observable_data: ObservableData,
    *,
    fig: go.Figure | None = None,
    colors: Mapping[Path, str] | None = None,
    labels: Mapping[Path, str] | None = None,
    groups: Mapping[Path, str] | None = None,
    marker_kwargs: Mapping[str, object] | None = None,
    paths: Iterable[Path] | None = None,
    row: int | None = None,
    col: int | None = None,
) -> go.Figure:
    """Scatter the raw per-randomization observable points, one series per path.

    Args:
        observable_data: The raw observable data.
        fig: An existing figure to add traces to. If ``None``, a new figure is created.
        colors: Optional per-path marker colors, forwarded to :func:`plot_path_scatters`.
        labels: Optional per-path legend labels, forwarded to :func:`plot_path_scatters`.
        groups: Optional per-path ``legendgroup`` keys, forwarded to :func:`plot_path_scatters`.
        marker_kwargs: Optional ``marker`` overrides; defaults to a filled-circle ``symbol``.
        paths: Optional paths to restrict to. Defaults to all paths in the data.
        row: The subplot row to add traces to (1-indexed).
        col: The subplot column to add traces to (1-indexed).

    Returns:
        The figure with the added traces.
    """
    return plot_path_scatters(
        observable_data_points(observable_data, paths),
        fig=fig,
        colors=colors,
        labels=labels,
        groups=groups,
        marker_kwargs={"symbol": _OBSERVABLE_POINTS_SYMBOL, **(marker_kwargs or {})},
        row=row,
        col=col,
    )


@HAS_PLOTLY.require_in_call
def plot_averaged_points(
    averaged_data: AveragedData,
    *,
    fig: go.Figure | None = None,
    colors: Mapping[Path, str] | None = None,
    labels: Mapping[Path, str] | None = None,
    groups: Mapping[Path, str] | None = None,
    marker_kwargs: Mapping[str, object] | None = None,
    paths: Iterable[Path] | None = None,
    row: int | None = None,
    col: int | None = None,
) -> go.Figure:
    """Scatter the averaged observable points (with error bars), one series per path.

    Args:
        averaged_data: The averaged data.
        fig: An existing figure to add traces to. If ``None``, a new figure is created.
        colors: Optional per-path marker colors, forwarded to :func:`plot_path_scatters`.
        labels: Optional per-path legend labels, forwarded to :func:`plot_path_scatters`.
        groups: Optional per-path ``legendgroup`` keys, forwarded to :func:`plot_path_scatters`.
        marker_kwargs: Optional ``marker`` overrides; defaults to an ``x`` symbol.
        paths: Optional paths to restrict to. Defaults to all paths in the data.
        row: The subplot row to add traces to (1-indexed).
        col: The subplot column to add traces to (1-indexed).

    Returns:
        The figure with the added traces.
    """
    return plot_path_scatters(
        averaged_data_points(averaged_data, paths),
        fig=fig,
        colors=colors,
        labels=labels,
        groups=groups,
        marker_kwargs={"symbol": _AVERAGED_POINTS_SYMBOL, **(marker_kwargs or {})},
        row=row,
        col=col,
    )


@HAS_PLOTLY.require_in_call
def plot_fit_curves(
    averaged_data: AveragedData,
    *,
    fig: go.Figure | None = None,
    colors: Mapping[Path, str] | None = None,
    labels: Mapping[Path, str] | None = None,
    groups: Mapping[Path, str] | None = None,
    line_kwargs: Mapping[str, object] | None = None,
    depths: Sequence[float] | np.ndarray | None = None,
    paths: Iterable[Path] | None = None,
    row: int | None = None,
    col: int | None = None,
) -> go.Figure:
    """Plot the fitted decay curves from averaged data, one series per path.

    Args:
        averaged_data: The averaged data (its ``depth == -1`` entries hold the fitted parameters).
        fig: An existing figure to add traces to. If ``None``, a new figure is created.
        colors: Optional per-path line colors, forwarded to :func:`plot_path_decay_curves`.
        labels: Optional per-path legend labels, forwarded to :func:`plot_path_decay_curves`.
        groups: Optional per-path ``legendgroup`` keys, forwarded to :func:`plot_path_decay_curves`.
        line_kwargs: Optional ``line`` overrides; defaults to a solid ``dash``.
        depths: The depth range for the curves. Defaults to a range derived from the averaged data's
            points, or ``0``–``10`` when it has none.
        paths: Optional paths to restrict to. Defaults to all paths in the data.
        row: The subplot row to add traces to (1-indexed).
        col: The subplot column to add traces to (1-indexed).

    Returns:
        The figure with the added traces.
    """
    bases, intercepts = averaged_data_curves(averaged_data, paths)
    if depths is None:
        depths = _default_depths(averaged_data_points(averaged_data, paths))
    return plot_path_decay_curves(
        bases,
        intercepts,
        depths,
        fig=fig,
        colors=colors,
        labels=labels,
        groups=groups,
        line_kwargs={"dash": _FIT_DASH, **(line_kwargs or {})},
        row=row,
        col=col,
    )


@HAS_PLOTLY.require_in_call
def plot_model_curves(
    model: LinearMap,
    model_data: ModelData,
    *,
    paths: Iterable[Path] | None = None,
    fig: go.Figure | None = None,
    colors: Mapping[Path, str] | None = None,
    labels: Mapping[Path, str] | None = None,
    groups: Mapping[Path, str] | None = None,
    line_kwargs: Mapping[str, object] | None = None,
    depths: Sequence[float] | np.ndarray | None = None,
    row: int | None = None,
    col: int | None = None,
) -> go.Figure:
    """Plot model-predicted decay curves, one series per path.

    Args:
        model: A fidelity model (a :class:`~.LinearMap` with a :class:`~.LogFidelitySpace` output).
        model_data: The fitted model parameters.
        paths: The paths to predict decays for. If ``None``, no curves are drawn (the model has no
            paths of its own to enumerate).
        fig: An existing figure to add traces to. If ``None``, a new figure is created.
        colors: Optional per-path line colors, forwarded to :func:`plot_path_decay_curves`.
        labels: Optional per-path legend labels, forwarded to :func:`plot_path_decay_curves`.
        groups: Optional per-path ``legendgroup`` keys, forwarded to :func:`plot_path_decay_curves`.
        line_kwargs: Optional ``line`` overrides; defaults to a dashed ``dash``.
        depths: The depth range for the curves. Defaults to ``0``–``10``.
        row: The subplot row to add traces to (1-indexed).
        col: The subplot column to add traces to (1-indexed).

    Returns:
        The figure with the added traces.
    """
    import plotly.graph_objects as go

    if fig is None:
        fig = go.Figure()
    if paths is None:
        return fig

    bases, intercepts = model_curves(model, model_data, paths)
    if depths is None:
        depths = _default_depths()
    return plot_path_decay_curves(
        bases,
        intercepts,
        depths,
        fig=fig,
        colors=colors,
        labels=labels,
        groups=groups,
        line_kwargs={"dash": _MODEL_DASH, **(line_kwargs or {})},
        row=row,
        col=col,
    )


@HAS_PLOTLY.require_in_call
def plot_observable_means(
    observable_data: ObservableData,
    *,
    fig: go.Figure | None = None,
    colors: Mapping[Path, str] | None = None,
    labels: Mapping[Path, str] | None = None,
    groups: Mapping[Path, str] | None = None,
    marker_kwargs: Mapping[str, object] | None = None,
    paths: Iterable[Path] | None = None,
    row: int | None = None,
    col: int | None = None,
) -> go.Figure:
    """Scatter the per-depth *means* of raw observable data, one series per path.

    Averages the observable data over randomizations (the same computation as the
    :class:`~.AverageObservables` stage) and renders the result with :func:`plot_averaged_points`,
    so each path shows one error-barred point per depth rather than the raw per-randomization cloud.

    Args:
        observable_data: The raw observable data to average.
        fig: An existing figure to add traces to. If ``None``, a new figure is created.
        colors: Optional per-path marker colors, forwarded to :func:`plot_averaged_points`.
        labels: Optional per-path legend labels, forwarded to :func:`plot_averaged_points`.
        groups: Optional per-path ``legendgroup`` keys, forwarded to :func:`plot_averaged_points`.
        marker_kwargs: Optional ``marker`` overrides; defaults to an ``x`` symbol.
        paths: Optional paths to restrict to. Defaults to all paths in the data.
        row: The subplot row to add traces to (1-indexed).
        col: The subplot column to add traces to (1-indexed).

    Returns:
        The figure with the added traces.
    """
    from ..analysis.average_observables import average_observables

    averaged_data = average_observables(observable_data, set(paths) if paths is not None else None)
    return plot_averaged_points(
        averaged_data,
        fig=fig,
        colors=colors,
        labels=labels,
        groups=groups,
        marker_kwargs=marker_kwargs,
        paths=paths,
        row=row,
        col=col,
    )


# --------------------------------------------------------------------------------------------------
# Layers: coordinated render callables for the orchestrators
# --------------------------------------------------------------------------------------------------
#
# A layer closes over its data (and per-layer style) and is invoked by an orchestrator as
# ``layer(*, fig, colors, labels, groups, depths, paths, row, col)``. Because ``paths`` is injected
# at call time, the same layer works unchanged on a single axes and in every grid cell. Layers
# swallow (via ``**_``) any coordination kwargs their underlying plotter does not use.

Layer = Callable[..., "go.Figure"]


def _tag(render: Layer, name: str, proxy: dict) -> Layer:
    """Attach symbol-legend metadata (display name + proxy trace style) to a layer callable."""
    render.legend_name = name
    render.legend_proxy = proxy
    return render


def observable_points_layer(
    observable_data: ObservableData, *, marker_kwargs: Mapping[str, object] | None = None
) -> Layer:
    """A layer drawing raw observable scatter points (see :func:`plot_observable_points`)."""

    def render(*, fig, colors, labels, groups, paths, row, col, **_):
        return plot_observable_points(
            observable_data,
            fig=fig,
            colors=colors,
            labels=labels,
            groups=groups,
            marker_kwargs=marker_kwargs,
            paths=paths,
            row=row,
            col=col,
        )

    return _tag(
        render,
        "Observable points",
        {
            "mode": "markers",
            "marker": {"symbol": _OBSERVABLE_POINTS_SYMBOL, **(marker_kwargs or {})},
        },
    )


def observable_means_layer(
    observable_data: ObservableData, *, marker_kwargs: Mapping[str, object] | None = None
) -> Layer:
    """A layer drawing averaged observable points (see :func:`plot_observable_means`)."""

    def render(*, fig, colors, labels, groups, paths, row, col, **_):
        return plot_observable_means(
            observable_data,
            fig=fig,
            colors=colors,
            labels=labels,
            groups=groups,
            marker_kwargs=marker_kwargs,
            paths=paths,
            row=row,
            col=col,
        )

    return _tag(
        render,
        "Observable means",
        {"mode": "markers", "marker": {"symbol": _AVERAGED_POINTS_SYMBOL, **(marker_kwargs or {})}},
    )


def averaged_points_layer(
    averaged_data: AveragedData, *, marker_kwargs: Mapping[str, object] | None = None
) -> Layer:
    """A layer drawing averaged points with error bars (see :func:`plot_averaged_points`)."""

    def render(*, fig, colors, labels, groups, paths, row, col, **_):
        return plot_averaged_points(
            averaged_data,
            fig=fig,
            colors=colors,
            labels=labels,
            groups=groups,
            marker_kwargs=marker_kwargs,
            paths=paths,
            row=row,
            col=col,
        )

    return _tag(
        render,
        "Averaged points",
        {"mode": "markers", "marker": {"symbol": _AVERAGED_POINTS_SYMBOL, **(marker_kwargs or {})}},
    )


def fit_curves_layer(
    averaged_data: AveragedData, *, line_kwargs: Mapping[str, object] | None = None
) -> Layer:
    """A layer drawing fitted decay curves (see :func:`plot_fit_curves`)."""

    def render(*, fig, colors, labels, groups, depths, paths, row, col, **_):
        return plot_fit_curves(
            averaged_data,
            fig=fig,
            colors=colors,
            labels=labels,
            groups=groups,
            line_kwargs=line_kwargs,
            depths=depths,
            paths=paths,
            row=row,
            col=col,
        )

    return _tag(
        render,
        "Fit",
        {"mode": "lines", "line": {"dash": _FIT_DASH, **(line_kwargs or {})}},
    )


def model_curves_layer(
    model: LinearMap, model_data: ModelData, *, line_kwargs: Mapping[str, object] | None = None
) -> Layer:
    """A layer drawing model-predicted decay curves (see :func:`plot_model_curves`)."""

    def render(*, fig, colors, labels, groups, depths, paths, row, col, **_):
        return plot_model_curves(
            model,
            model_data,
            paths=paths,
            fig=fig,
            colors=colors,
            labels=labels,
            groups=groups,
            line_kwargs=line_kwargs,
            depths=depths,
            row=row,
            col=col,
        )

    return _tag(
        render,
        "Model",
        {"mode": "lines", "line": {"dash": _MODEL_DASH, **(line_kwargs or {})}},
    )


# --------------------------------------------------------------------------------------------------
# Orchestrators: lay out a list of layers with shared coordination
# --------------------------------------------------------------------------------------------------


def _add_symbol_legend(fig: go.Figure, layers: Iterable[Layer]) -> None:
    """Add a second legend (``legend2``) mapping each layer's symbol/dash to its name.

    Emits one neutral-colored proxy trace per distinct layer that carries metadata; a symbol
    legend is only added when at least two distinct layer types are present. No-op otherwise.

    This legend is a static reference key: its clicks are disabled (``itemclick=False``). Plotly
    ties a trace to a single ``legendgroup``/legend, and the data traces use that to group and
    toggle by path in the main legend, so the symbol legend cannot also toggle whole layers.
    """
    import plotly.graph_objects as go

    entries: list[tuple[str, dict]] = []
    seen: set[str] = set()
    for layer in layers:
        name = getattr(layer, "legend_name", None)
        proxy = getattr(layer, "legend_proxy", None)
        if name is None or proxy is None or name in seen:
            continue
        seen.add(name)
        entries.append((name, proxy))

    if len(entries) < 2:
        return

    for name, proxy in entries:
        trace = {
            "x": [None],
            "y": [None],
            "mode": proxy.get("mode", "markers"),
            "name": name,
            "legend": "legend2",
            "showlegend": True,
            "hoverinfo": "skip",
        }
        if "marker" in proxy:
            trace["marker"] = {**proxy["marker"], "color": _SYMBOL_LEGEND_COLOR}
        if "line" in proxy:
            trace["line"] = {**proxy["line"], "color": _SYMBOL_LEGEND_COLOR}
        fig.add_trace(go.Scatter(**trace))

    fig.update_layout(
        # Path legend: vertical, top-right (plotly default position). Series legend: a horizontal
        # strip along the top so it never stacks on top of the (variable-height) path legend.
        legend={"title_text": "Path", "y": 1.0, "yanchor": "top"},
        legend2={
            "title_text": "Series",
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0.0,
            "itemclick": False,
            "itemdoubleclick": False,
        },
    )


@HAS_PLOTLY.require_in_call
def plot_path_overlay(
    layers: Iterable[Layer],
    paths: Iterable[Path],
    *,
    gate_set: GateSet | None = None,
    colors: Mapping[Path, str] | None = None,
    labels: Mapping[Path, str] | None = None,
    groups: Mapping[Path, str] | None = None,
    label_style: str = "formula",
    depths: Sequence[float] | np.ndarray | None = None,
    title: str | None = None,
    fig: go.Figure | None = None,
    row: int | None = None,
    col: int | None = None,
) -> go.Figure:
    """Overlay an arbitrary list of decay layers on a single axes, with shared coordination.

    Each path is its own series: one color and one deduplicated legend entry shared across every
    layer. Labels default to :func:`path_labels` when a gate set is available, else index strings.

    When ``fig`` is ``None`` this creates the figure and finalizes it (dedupe legend, symbol legend,
    axis titles). When an existing ``fig`` is passed (e.g. a subplot cell) it only adds traces and
    leaves finalization to the caller — how :func:`plot_path_grid_overlay` renders each cell.

    Args:
        layers: The layers to draw (e.g. from :func:`averaged_points_layer`,
            :func:`model_curves_layer`), each invoked with the shared coordination.
        paths: The paths to plot (opaque layers cannot be introspected, so this is explicit).
        gate_set: The gate set used to build default labels.
        colors: Optional per-path color overrides; each path is otherwise assigned its own color.
        labels: Optional per-path legend labels.
        groups: Optional per-path ``legendgroup``/color-identity keys. Defaults to a per-path
            identity (each path its own color and legend entry).
        label_style: The :func:`~.path_math_label` style for default labels.
        depths: The depth range passed to curve layers. Defaults to ``0``–``10``.
        title: An optional figure title.
        fig: An existing figure to add traces to. If ``None``, a new figure is created + finalized.
        row: The subplot row to add traces to (1-indexed).
        col: The subplot column to add traces to (1-indexed).

    Returns:
        The figure with the overlaid layers.
    """
    import plotly.graph_objects as go

    if is_new_fig := fig is None:
        fig = go.Figure()

    layers = list(layers)
    path_list = list(paths)
    if labels is None:
        if gate_set is not None:
            labels = path_labels(path_list, gate_set, style=label_style)
        else:
            labels = {path: str(index) for index, path in enumerate(path_list)}

    identity = groups if groups is not None else {p: str(i) for i, p in enumerate(path_list)}
    color_map = _colors_by_group(identity, colors)
    if depths is None:
        depths = _default_depths()

    for layer in layers:
        layer(
            fig=fig,
            colors=color_map,
            labels=labels,
            groups=identity,
            depths=depths,
            paths=path_list,
            row=row,
            col=col,
        )

    # Finalize only when we own the figure; a caller passing ``fig`` (e.g. a grid cell) finalizes.
    if is_new_fig:
        _dedupe_legend(fig)
        _add_symbol_legend(fig, layers)
        fig.update_layout(xaxis_title="depth", yaxis_title="observable")
        if title is not None:
            fig.update_layout(title_text=title)
    return fig


@HAS_PLOTLY.require_in_call
def plot_path_grid_overlay(
    groups: Mapping[Hashable, Sequence[Path]],
    layers: Iterable[Layer],
    *,
    num_cols: int = 3,
    gate_set: GateSet | None = None,
    label: Callable[[Path, Hashable], str] | None = None,
    series_key: Callable[[Path, Hashable], Hashable] | None = None,
    colors: Mapping[Hashable, str] | None = None,
    label_style: str = "formula",
    depths: Sequence[float] | np.ndarray | None = None,
    title: str | None = None,
) -> go.Figure:
    """Lay out an arbitrary list of decay layers across a grid of subplots (one per group).

    Subplot membership (``groups``) and series identity (``series_key``) are independent: color and
    the deduplicated legend entry are keyed by ``series_key``'s value, so paths that resolve to the
    same key share a color and one legend entry across the whole grid, while ``label`` controls only
    displayed text. The same ``layers`` are drawn in every cell, restricted to that cell's paths.

    Args:
        groups: A mapping from a group key (subplot title) to the paths drawn in that subplot.
        layers: The layers to draw in each cell.
        num_cols: The number of subplot columns; rows are derived from the group count.
        gate_set: The gate set for default labels.
        label: A callable ``(path, group_key) -> str`` giving each path's displayed label. Defaults
            to a group-independent :func:`~.path_math_label` (formula, repeatable only).
        series_key: A callable ``(path, group_key) -> Hashable`` giving each path's series identity
            (its color and shared legend entry). Defaults to the displayed ``label``.
        colors: Optional overrides mapping a series key to a color.
        label_style: The :func:`~.path_math_label` style for the default label.
        depths: The depth range passed to curve layers. Defaults to ``0``–``10``.
        title: An optional figure title.

    Returns:
        The subplot-grid figure.
    """
    from plotly.subplots import make_subplots

    layers = list(layers)
    group_items = list(groups.items())
    num_rows = max(1, -(-len(group_items) // num_cols))  # ceil division
    fig = make_subplots(
        rows=num_rows, cols=num_cols, subplot_titles=[str(key) for key, _ in group_items]
    )
    resolved_gate_set = _resolve_gate_set(gate_set, None)
    if depths is None:
        depths = _default_depths()

    palette = _palette()
    series_to_color: dict[Hashable, str] = {}

    def _label_for(path: Path, key: Hashable) -> str | None:
        if label is not None:
            return label(path, key)
        if resolved_gate_set is not None:
            return (
                "$"
                + path_math_label(resolved_gate_set, path, style=label_style, repeatable_only=True)
                + "$"
            )
        return None

    def _color_for(series_val: Hashable | None) -> str | None:
        if series_val is None:
            return None
        if colors and series_val in colors:
            return colors[series_val]
        if series_val not in series_to_color:
            series_to_color[series_val] = palette[len(series_to_color) % len(palette)]
        return series_to_color[series_val]

    for index, (key, group_paths) in enumerate(group_items):
        grid_row = index // num_cols + 1
        grid_col = index % num_cols + 1
        group_paths = list(group_paths)

        cell_labels = {path: _label_for(path, key) for path in group_paths}
        cell_series = {
            path: (series_key(path, key) if series_key is not None else cell_labels[path])
            for path in group_paths
        }
        cell_colors = {path: _color_for(cell_series[path]) for path in group_paths}
        cell_groups = {
            path: None if cell_series[path] is None else str(cell_series[path])
            for path in group_paths
        }

        # Render this cell through the single-axes overlay (passing our resolved coordination and an
        # existing ``fig`` so it only adds traces; we finalize the whole grid once below).
        plot_path_overlay(
            layers,
            group_paths,
            colors=cell_colors,
            labels=cell_labels,
            groups=cell_groups,
            depths=depths,
            fig=fig,
            row=grid_row,
            col=grid_col,
        )

    _dedupe_legend(fig)
    _add_symbol_legend(fig, layers)
    fig.update_xaxes(title_text="depth")
    fig.update_yaxes(title_text="observable")
    if title is not None:
        fig.update_layout(title_text=title)
    return fig


# --------------------------------------------------------------------------------------------------
# Batteries-included overlays (the standard four-layer decay plot)
# --------------------------------------------------------------------------------------------------


def standard_decay_layers(
    *,
    observable_data: ObservableData | None = None,
    observable_type: Literal["raw", "means", "both"] = "raw",
    observable_marker_kwargs: Mapping[str, object] | None = None,
    means_marker_kwargs: Mapping[str, object] | None = None,
    averaged_data: AveragedData | None = None,
    averaged_marker_kwargs: Mapping[str, object] | None = None,
    averaged_line_kwargs: Mapping[str, object] | None = None,
    model: LinearMap | None = None,
    model_data: ModelData | None = None,
    model_line_kwargs: Mapping[str, object] | None = None,
) -> list[Layer]:
    """Build the standard decay layer stack for the supplied data sources.

    Includes observable-scatter layer(s) (if ``observable_data``), averaged-points and fit-curve
    layers (if ``averaged_data``), and a model-curve layer (if ``model`` and ``model_data``). Pass
    the result to :func:`plot_path_overlay` or :func:`plot_path_grid_overlay`.

    Args:
        observable_data: Optional raw observable data.
        observable_type: Which observable layer(s) to draw from ``observable_data`` — ``"raw"`` (raw
            per-randomization scatter), ``"means"`` (per-depth means with error bars, averaged via
            :class:`~.AverageObservables`), or ``"both"``. The raw and means layers are styled
            independently (defaulting to a ``circle`` and an ``x`` symbol respectively).
        observable_marker_kwargs: Optional ``marker`` overrides for the raw observable scatter.
        means_marker_kwargs: Optional ``marker`` overrides for the observable-means scatter.
        averaged_data: Optional averaged data (averaged points + fitted curve).
        averaged_marker_kwargs: Optional ``marker`` overrides for the averaged points.
        averaged_line_kwargs: Optional ``line`` overrides for the fitted curve.
        model: Optional fidelity model for predicted curves (requires ``model_data``).
        model_data: Optional fitted parameters for predicted curves (requires ``model``).
        model_line_kwargs: Optional ``line`` overrides for the model curve.

    Returns:
        The list of layers, in draw order.

    Raises:
        ValueError: If ``observable_type`` is not ``"raw"``, ``"means"``, or ``"both"``.
    """
    if observable_type not in ("raw", "means", "both"):
        raise ValueError(
            f"Invalid observable_type: {observable_type!r}. Must be 'raw', 'means', or 'both'."
        )

    layers: list[Layer] = []
    if observable_data is not None:
        if observable_type in ("raw", "both"):
            layers.append(
                observable_points_layer(observable_data, marker_kwargs=observable_marker_kwargs)
            )
        if observable_type in ("means", "both"):
            layers.append(
                observable_means_layer(observable_data, marker_kwargs=means_marker_kwargs)
            )
    if averaged_data is not None:
        layers.append(averaged_points_layer(averaged_data, marker_kwargs=averaged_marker_kwargs))
        layers.append(fit_curves_layer(averaged_data, line_kwargs=averaged_line_kwargs))
    if model is not None and model_data is not None:
        layers.append(model_curves_layer(model, model_data, line_kwargs=model_line_kwargs))
    return layers


@HAS_PLOTLY.require_in_call
def plot_2_qubit_decays(
    pairs: Sequence[tuple[int, int]],
    *,
    observable_data: ObservableData | None = None,
    observable_type: Literal["raw", "means", "both"] = "raw",
    observable_marker_kwargs: Mapping[str, object] | None = None,
    means_marker_kwargs: Mapping[str, object] | None = None,
    averaged_data: AveragedData | None = None,
    averaged_marker_kwargs: Mapping[str, object] | None = None,
    averaged_line_kwargs: Mapping[str, object] | None = None,
    model: LinearMap | None = None,
    model_data: ModelData | None = None,
    model_line_kwargs: Mapping[str, object] | None = None,
    gate_set: GateSet | None = None,
    num_cols: int = 3,
    colors: Mapping[Hashable, str] | None = None,
    label_style: str = "formula",
    noise_site: Mapping[str, str] | None = None,
    placeholders: tuple[str, str] = ("i", "j"),
    depths: Sequence[float] | np.ndarray | None = None,
    title: str | None = None,
) -> go.Figure:
    """Grid of fidelity decays over qubit pairs, one subplot per pair, with shared labels.

    Each subplot shows the decays for the paths acting on that pair (assigned by the rule
    ``set(pair) >= path.start_fragment[0].out_bit_indices``). Series labels are **canonicalized**:
    the pair's qubits are relabeled to ``placeholders`` (min qubit -> ``"i"``, max -> ``"j"``), so a
    given Pauli fidelity (e.g. ``X_{i} X_{j}``) shares a color and a single legend entry across
    every pair it appears in. Subplot titles show the actual pair.

    Args:
        pairs: The qubit pairs to plot, one subplot each.
        observable_data: Optional raw observable data for scatter points.
        observable_type: Which observable layer(s) to draw from ``observable_data`` — ``"raw"``,
            ``"means"``, or ``"both"`` (see :func:`standard_decay_layers`).
        observable_marker_kwargs: Optional ``marker`` properties for the raw observable points.
        means_marker_kwargs: Optional ``marker`` properties for the observable-means points.
        averaged_data: Optional averaged data for averaged points and fitted curves.
        averaged_marker_kwargs: Optional ``marker`` properties for the averaged points.
        averaged_line_kwargs: Optional ``line`` properties for the fitted curves.
        model: Optional fidelity model for predicted curves (requires ``model_data``).
        model_data: Optional fitted parameters for predicted curves (requires ``model``).
        model_line_kwargs: Optional ``line`` properties for the model curves.
        gate_set: The gate set used to build labels. Defaults to the model's gate set; required
            (here or via the model) since labels are always drawn.
        num_cols: The number of subplot columns; rows are derived from the pair count.
        colors: Optional overrides mapping a (canonicalized) series label to a color.
        label_style: The :func:`~.path_math_label` style for the series labels.
        noise_site: An optional noise-site mapping forwarded to :func:`~.path_math_label` (with
            ``style="formula"`` this yields the compact ``f^{gate}_{pauli}`` label).
        placeholders: The two display symbols for the pair's (min, max) qubit indices.
        depths: The depth range for the curves. Defaults to ``0`` through the largest observed
            depth.
        title: An optional figure title.

    Returns:
        The subplot-grid figure.

    Raises:
        ValueError: If no gate set is available (neither ``gate_set`` nor a model with one).
    """
    resolved_gate_set = _resolve_gate_set(gate_set, model)
    if resolved_gate_set is None:
        raise ValueError("A gate_set (or a model carrying one) is required to label the decays.")

    paths = _dataset_paths(observable_data, averaged_data)
    groups: dict[Hashable, list[Path]] = {}
    for pair in pairs:
        pair_set = set(pair)
        groups[tuple(pair)] = [
            path
            for path in paths
            if path.start_fragment and pair_set.issuperset(path.start_fragment[0].out_bit_indices)
        ]

    def _label(path: Path, pair: Hashable) -> str:
        low, high = sorted(pair)
        qubit_labels = {low: placeholders[0], high: placeholders[1]}
        return (
            "$"
            + path_math_label(
                resolved_gate_set,
                path,
                style=label_style,
                noise_site=noise_site,
                repeatable_only=True,
                qubit_labels=qubit_labels,
            )
            + "$"
        )

    layers = standard_decay_layers(
        observable_data=observable_data,
        observable_type=observable_type,
        observable_marker_kwargs=observable_marker_kwargs,
        means_marker_kwargs=means_marker_kwargs,
        averaged_data=averaged_data,
        averaged_marker_kwargs=averaged_marker_kwargs,
        averaged_line_kwargs=averaged_line_kwargs,
        model=model,
        model_data=model_data,
        model_line_kwargs=model_line_kwargs,
    )
    return plot_path_grid_overlay(
        groups,
        layers,
        num_cols=num_cols,
        gate_set=resolved_gate_set,
        label=_label,
        colors=colors,
        depths=depths,
        title=title,
    )
