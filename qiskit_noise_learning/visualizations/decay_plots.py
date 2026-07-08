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


# --------------------------------------------------------------------------------------------------
# Core plotting primitives
# --------------------------------------------------------------------------------------------------


@HAS_PLOTLY.require_in_call
def plot_path_scatter(
    points: Mapping[Path, PointSeries],
    *,
    fig: go.Figure | None = None,
    colors: Mapping[Path, str] | None = None,
    labels: Mapping[Path, str] | None = None,
    row: int | None = None,
    col: int | None = None,
) -> go.Figure:
    """Path-referenced scatter plot of point series.

    Args:
        points: A mapping from path to its :class:`PointSeries`. A series with ``stds`` set is drawn
            with symmetric error bars.
        fig: An existing figure to add traces to. If ``None``, a new figure is created.
        colors: An optional mapping from path to a plotly color string for its markers.
        labels: An optional mapping from path to a legend label. A path's label also serves as its
            ``legendgroup``; paths without a label are omitted from the legend.
        row: The subplot row to add traces to (1-indexed), for figures created with subplots.
        col: The subplot column to add traces to (1-indexed), for figures created with subplots.

    Returns:
        The figure with the added traces.
    """
    import plotly.graph_objects as go

    if fig is None:
        fig = go.Figure()

    for path, series in points.items():
        label = labels.get(path) if labels else None
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
                legendgroup=label,
                showlegend=label is not None,
                marker={"color": color},
                error_y=error_y,
            ),
            row=row,
            col=col,
        )

    return fig


@HAS_PLOTLY.require_in_call
def plot_decay_curves(
    bases: Mapping[Path, float],
    intercepts: Mapping[Path, float],
    depths: Sequence[float] | np.ndarray,
    *,
    fig: go.Figure | None = None,
    colors: Mapping[Path, str] | None = None,
    labels: Mapping[Path, str] | None = None,
    dash: str = "solid",
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
        labels: An optional mapping from path to a legend label, also used as the ``legendgroup``.
        dash: The plotly line dash style (e.g. ``"solid"``, ``"dash"``, ``"dot"``).
        row: The subplot row to add traces to (1-indexed), for figures created with subplots.
        col: The subplot column to add traces to (1-indexed), for figures created with subplots.

    Returns:
        The figure with the added traces.
    """
    import plotly.graph_objects as go

    if fig is None:
        fig = go.Figure()

    depths_arr = np.asarray(depths, dtype=float)
    for path, base in bases.items():
        label = labels.get(path) if labels else None
        color = colors.get(path) if colors else None
        y = intercepts[path] * base**depths_arr
        fig.add_trace(
            go.Scatter(
                x=depths_arr,
                y=y,
                mode="lines",
                name=label,
                legendgroup=label,
                showlegend=label is not None,
                line={"color": color, "dash": dash},
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
) -> tuple[dict[Path, float], dict[Path, float]]:
    """Extract exponential decay parameters from averaged data.

    Extracts the ``base`` and ``intercept`` for entries with ``depth == -1``, for use in
    :func:`plot_decay_curves`. The ``intercept`` is drawn from ``metadata["spam_fidelity"]``, and
    defaults to ``1.0`` if absent.

    Args:
        averaged_data: The averaged data to extract from.

    Returns:
        A ``(bases, intercepts)`` pair of mappings from path to float.
    """
    dataset = averaged_data.dataset
    mask = dataset["depth"].data == -1
    paths = dataset["unbound_path"].data[mask]
    observables = dataset["observables"].data[mask]
    metadata = dataset["metadata"].data[mask]

    bases: dict[Path, float] = {}
    intercepts: dict[Path, float] = {}
    for path, fidelity, meta in zip(paths, observables, metadata):
        bases[path] = float(fidelity)
        intercepts[path] = float(meta["spam_fidelity"]) if meta and "spam_fidelity" in meta else 1.0

    return bases, intercepts


def averaged_data_points(averaged_data: AveragedData) -> dict[Path, PointSeries]:
    """Extract point series data for the bound paths in an average data instance.

    Args:
        averaged_data: The averaged data to extract from.

    Returns:
        A mapping from path to its :class:`PointSeries` (with ``stds`` populated).
    """
    dataset = averaged_data.dataset
    mask = dataset["depth"].data >= 0
    paths = dataset["unbound_path"].data[mask]
    depths = dataset["depth"].data[mask]
    observables = dataset["observables"].data[mask]
    stds = dataset["std"].data[mask]

    result: dict[Path, PointSeries] = {}
    for path in dict.fromkeys(paths):
        path_mask = np.array([other == path for other in paths])
        result[path] = PointSeries(
            xs=depths[path_mask].astype(float),
            ys=observables[path_mask].astype(float),
            stds=stds[path_mask].astype(float),
        )

    return result


def observable_data_points(observable_data: ObservableData) -> dict[Path, PointSeries]:
    """Extract raw per-randomization decay points from observable data.

    Flattens the ``randomization`` axis (dropping ``nan`` raggedness padding) so each retained
    randomization becomes its own point.

    Args:
        observable_data: The observable data to extract from.

    Returns:
        A mapping from path to its :class:`PointSeries` (with ``stds`` unset).
    """
    dataset = observable_data.dataset
    paths = dataset["unbound_path"].data
    depths = dataset["depth"].data
    observables = dataset["observables"].data

    result: dict[Path, PointSeries] = {}
    for path in dict.fromkeys(paths):
        row_mask = np.array([other == path for other in paths])

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
    """The qualitative color palette used to assign per-series colors."""
    import plotly.colors as pc

    return pc.qualitative.Plotly


def _colors_by_label(
    labels: Mapping[Path, str | None],
    overrides: Mapping[Path, str] | None,
) -> dict[Path, str | None]:
    """Assign a color per path, keyed by label so paths sharing a label share a color."""
    palette = _palette()
    label_to_color: dict[str, str] = {}
    result: dict[Path, str | None] = {}
    for path, label in labels.items():
        if overrides and path in overrides:
            result[path] = overrides[path]
        elif label is None:
            result[path] = None
        else:
            if label not in label_to_color:
                label_to_color[label] = palette[len(label_to_color) % len(palette)]
            result[path] = label_to_color[label]
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


# --------------------------------------------------------------------------------------------------
# Top-level overlays
# --------------------------------------------------------------------------------------------------


@HAS_PLOTLY.require_in_call
def plot_decays(
    *,
    observable_data: ObservableData | None = None,
    averaged_data: AveragedData | None = None,
    model: LinearMap | None = None,
    model_data: ModelData | None = None,
    paths: Iterable[Path] | None = None,
    gate_set: GateSet | None = None,
    fig: go.Figure | None = None,
    colors: Mapping[Path, str] | None = None,
    labels: Mapping[Path, str] | None = None,
    label_style: str = "formula",
    depths: Sequence[float] | np.ndarray | None = None,
    row: int | None = None,
    col: int | None = None,
) -> go.Figure:
    """Overlay decay points and fitted / model-predicted decay curves on a single axes.

    Draws, per unbound path: raw scatter points (from ``observable_data``), averaged points with
    error bars (from ``averaged_data``), the fitted decay curve (solid, from ``averaged_data``), and
    the model-predicted decay curve (dashed, from ``model`` + ``model_data``). Points and both
    curves for a path share a color and a single legend entry.

    Args:
        observable_data: Optional raw observable data for scatter points.
        averaged_data: Optional averaged data for averaged points and the fitted curve.
        model: Optional fidelity model for the predicted curve (requires ``model_data``).
        model_data: Optional fitted parameters for the predicted curve (requires ``model``).
        paths: The paths to plot. Defaults to the union of paths present in the supplied data.
        gate_set: The gate set used to build default labels. Defaults to the model's gate set.
        fig: An existing figure to add traces to. If ``None``, a new figure is created.
        colors: Optional per-path color overrides. Paths sharing a label otherwise share a color.
        labels: Optional per-path legend labels. Defaults to :func:`path_labels` when a gate set is
            available, otherwise to index strings.
        label_style: The :func:`~.path_math_label` style for default labels.
        depths: The depth range for the curves. Defaults to ``0`` through the largest observed
            depth.
        row: The subplot row to add traces to (1-indexed).
        col: The subplot column to add traces to (1-indexed).

    Returns:
        The figure with the overlaid traces.
    """
    import plotly.graph_objects as go

    created = fig is None
    if fig is None:
        fig = go.Figure()

    observable_points = (
        observable_data_points(observable_data) if observable_data is not None else {}
    )
    averaged_points = averaged_data_points(averaged_data) if averaged_data is not None else {}
    fit_fidelities, fit_intercepts = (
        averaged_data_curves(averaged_data) if averaged_data is not None else ({}, {})
    )

    if paths is not None:
        path_list = list(paths)
    else:
        path_list = list(dict.fromkeys([*observable_points, *averaged_points, *fit_fidelities]))

    model_fidelities: dict[Path, float] = {}
    model_intercepts: dict[Path, float] = {}
    if model is not None and model_data is not None:
        model_fidelities, model_intercepts = model_curves(model, model_data, path_list)

    if labels is None:
        resolved_gate_set = _resolve_gate_set(gate_set, model)
        if resolved_gate_set is not None:
            labels = path_labels(path_list, resolved_gate_set, style=label_style)
        else:
            labels = {path: str(index) for index, path in enumerate(path_list)}

    color_map = _colors_by_label({path: labels.get(path) for path in path_list}, colors)

    if depths is None:
        depths = _default_depths(observable_points, averaged_points)

    plot_path_scatter(observable_points, fig=fig, colors=color_map, labels=labels, row=row, col=col)
    plot_path_scatter(averaged_points, fig=fig, colors=color_map, labels=labels, row=row, col=col)
    if fit_fidelities:
        plot_decay_curves(
            fit_fidelities,
            fit_intercepts,
            depths,
            fig=fig,
            colors=color_map,
            labels=labels,
            dash="solid",
            row=row,
            col=col,
        )
    if model_fidelities:
        plot_decay_curves(
            model_fidelities,
            model_intercepts,
            depths,
            fig=fig,
            colors=color_map,
            labels=labels,
            dash="dash",
            row=row,
            col=col,
        )

    _dedupe_legend(fig)
    if created:
        fig.update_layout(xaxis_title="depth", yaxis_title="observable")

    return fig


@HAS_PLOTLY.require_in_call
def plot_decay_grid(
    groups: Mapping[Hashable, Sequence[Path]],
    *,
    observable_data: ObservableData | None = None,
    averaged_data: AveragedData | None = None,
    model: LinearMap | None = None,
    model_data: ModelData | None = None,
    gate_set: GateSet | None = None,
    num_cols: int = 3,
    label: Callable[[Path, Hashable], str] | None = None,
    colors: Mapping[str, str] | None = None,
    label_style: str = "formula",
    depths: Sequence[float] | np.ndarray | None = None,
) -> go.Figure:
    """Draw a grid of decay overlays, one subplot per group.

    Subplot membership and series identity are independent: ``groups`` decides which subplot a path
    is drawn in (a path may appear in several), while the ``label`` callable's returned string
    decides a path's color and legend entry. Labels equal across subplots share a color and collapse
    to a single legend entry, giving one coherent legend for the whole grid.

    Args:
        groups: A mapping from a group key (used as the subplot title) to the paths in that subplot.
        observable_data: Optional raw observable data for scatter points.
        averaged_data: Optional averaged data for averaged points and fitted curves.
        model: Optional fidelity model for predicted curves (requires ``model_data``).
        model_data: Optional fitted parameters for predicted curves (requires ``model``).
        gate_set: The gate set for default labels. Defaults to the model's gate set.
        num_cols: The number of subplot columns; rows are derived from the group count.
        label: A callable ``(path, group_key) -> str`` giving each path's label within a subplot.
            Defaults to a group-independent :func:`~.path_math_label` (formula, repeatable only).
        colors: Optional overrides mapping a label string to a color.
        label_style: The :func:`~.path_math_label` style for the default label.
        depths: The depth range for the curves. Defaults to ``0`` through the largest observed
            depth.

    Returns:
        The subplot-grid figure.
    """
    from plotly.subplots import make_subplots

    group_items = list(groups.items())
    num_rows = max(1, -(-len(group_items) // num_cols))  # ceil division
    fig = make_subplots(
        rows=num_rows,
        cols=num_cols,
        subplot_titles=[str(key) for key, _ in group_items],
    )

    observable_points = (
        observable_data_points(observable_data) if observable_data is not None else {}
    )
    averaged_points = averaged_data_points(averaged_data) if averaged_data is not None else {}
    fit_fidelities, fit_intercepts = (
        averaged_data_curves(averaged_data) if averaged_data is not None else ({}, {})
    )
    resolved_gate_set = _resolve_gate_set(gate_set, model)

    if depths is None:
        depths = _default_depths(observable_points, averaged_points)

    palette = _palette()
    label_to_color: dict[str, str] = {}

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

    def _color_for(label_str: str | None) -> str | None:
        if label_str is None:
            return None
        if colors and label_str in colors:
            return colors[label_str]
        if label_str not in label_to_color:
            label_to_color[label_str] = palette[len(label_to_color) % len(palette)]
        return label_to_color[label_str]

    for index, (key, group_paths) in enumerate(group_items):
        grid_row = index // num_cols + 1
        grid_col = index % num_cols + 1
        group_paths = list(group_paths)

        cell_labels = {path: _label_for(path, key) for path in group_paths}
        cell_colors = {path: _color_for(cell_labels[path]) for path in group_paths}

        plot_path_scatter(
            {path: observable_points[path] for path in group_paths if path in observable_points},
            fig=fig,
            colors=cell_colors,
            labels=cell_labels,
            row=grid_row,
            col=grid_col,
        )
        plot_path_scatter(
            {path: averaged_points[path] for path in group_paths if path in averaged_points},
            fig=fig,
            colors=cell_colors,
            labels=cell_labels,
            row=grid_row,
            col=grid_col,
        )

        cell_fit = {path: fit_fidelities[path] for path in group_paths if path in fit_fidelities}
        if cell_fit:
            plot_decay_curves(
                cell_fit,
                {path: fit_intercepts[path] for path in cell_fit},
                depths,
                fig=fig,
                colors=cell_colors,
                labels=cell_labels,
                dash="solid",
                row=grid_row,
                col=grid_col,
            )

        if model is not None and model_data is not None and group_paths:
            group_fidelities, group_intercepts = model_curves(model, model_data, group_paths)
            plot_decay_curves(
                group_fidelities,
                group_intercepts,
                depths,
                fig=fig,
                colors=cell_colors,
                labels=cell_labels,
                dash="dash",
                row=grid_row,
                col=grid_col,
            )

    _dedupe_legend(fig)
    fig.update_xaxes(title_text="depth")
    fig.update_yaxes(title_text="observable")
    return fig
