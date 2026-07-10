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

"""Orchestrators: lay out layers on a single axes or a subplot grid, plus shared coordination."""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Literal

import numpy as np

from ...optionals import HAS_PLOTLY
from ..fidelity_math_labels import path_math_label
from .data_adapters import _dataset_paths
from .layers import Layer, standard_decay_layers
from .primitives import _default_depths

if TYPE_CHECKING:
    import plotly.graph_objects as go

    from ...data import AveragedData, ModelData, ObservableData
    from ...gate_sets import GateSet
    from ...math import LinearMap
    from ...sequences import Path


_SYMBOL_LEGEND_COLOR = "rgb(120,120,120)"


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


def _resolve_gate_set(gate_set: GateSet | None, model: LinearMap | None) -> GateSet | None:
    """The gate set to label with: the explicit one, else the model's."""
    if gate_set is not None:
        return gate_set
    if model is not None and hasattr(model, "gate_set"):
        return model.gate_set
    return None


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
        if layer.name is None or layer.proxy is None or layer.name in seen:
            continue
        seen.add(layer.name)
        entries.append((layer.name, layer.proxy))

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


@HAS_PLOTLY.require_in_call
def plot_path_overlay(
    layers: Iterable[Layer],
    paths: Iterable[Path] | None = None,
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
        paths: The paths to plot. Defaults to the union of the paths each layer contributes.
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
    if paths is not None:
        path_list = list(paths)
    else:
        path_list = list(dict.fromkeys(path for layer in layers for path in layer.paths))
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
        layer.render(
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
