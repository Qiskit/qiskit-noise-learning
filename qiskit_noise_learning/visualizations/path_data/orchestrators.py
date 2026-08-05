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

from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from itertools import chain
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from ...data import AveragedData, ModelData, ObservableData
from ...gate_sets import GateSet
from ...math import LinearMap
from ...optionals import HAS_MATPLOTLIB
from ...sequences import Path
from ..fidelity_math_labels import path_math_label
from ..interactive_figure import InteractiveFigure, TokenMap, cell_token, tag_legend
from .data_adapters import _dataset_paths, averaged_data_points, observable_data_points
from .layers import Layer, RenderContext, standard_decay_layers
from .primitives import _default_fragment_depths

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


# The dimensions every figure in this module resolves visibility along, one legend each: which path
# a mark belongs to, and which kind of mark it is. The order is the order the tokens appear in a
# trace's ``gid``, which is what ``RenderContext.gid_factory`` builds them in.
_DIMENSIONS = ("path", "layer")
_PATH_DIMENSION, _LAYER_DIMENSION = _DIMENSIONS

# Neutral color for the series legend's proxy handles: those entries mean a marker or a dash, not a
# path, and taking a path's color would suggest otherwise.
_SERIES_LEGEND_COLOR = "0.35"

# Layout, in inches. Both legends sit outside the axes and figures are saved with
# ``bbox_inches="tight"``, so the canvas only has to hold the subplots -- whatever the legends need
# is added to the crop. The path legend's font is small because its labels are (LaTeX) formulas and
# there is one per path.
_CELL_WIDTH = 3.4
_CELL_HEIGHT = 2.6
_SINGLE_WIDTH = 6.0
_SINGLE_HEIGHT = 4.0
_CELL_TITLE_FONT_SIZE = 10
_PATH_LEGEND_FONT_SIZE = 9
_PATH_LEGEND_LABEL_SPACING = 0.3


def _palette() -> list[str]:
    """Color palette for assigning default colors."""
    import matplotlib as mpl

    return list(mpl.rcParams["axes.prop_cycle"].by_key().get("color", ["C0"]))


def _path_qubits(path: Path) -> set[int]:
    """The qubit indices a path acts on: the support of every transition Pauli in its fragments."""
    qubits: set[int] = set()
    for fidelity_index in chain(path.start_fragment, path.repeatable_fragment, path.end_fragment):
        in_pauli, out_pauli = fidelity_index.transition
        qubits.update(int(i) for i in in_pauli.indices)
        qubits.update(int(i) for i in out_pauli.indices)
    return qubits


def _resolve_identity(
    paths: Sequence[Path], groups: Mapping[Path, Hashable | None] | None
) -> dict[Path, Hashable]:
    """Each path's series key -- what it shares a color and a path-legend entry with.

    A key of ``None`` used to mean "leave this path out of the legend". It cannot any more: every
    artist's ``gid`` names a series key and every key gets an entry, and a path without one could be
    switched off and never brought back. Such paths become their own series instead.
    """
    resolved: dict[Path, Hashable] = {}
    for path in paths:
        group = groups.get(path) if groups else None
        resolved[path] = path if group is None else group
    return resolved


def _colors_by_group(
    groups: Mapping[Path, Hashable],
    overrides: Mapping[Path, str] | None,
) -> dict[Path, str | None]:
    """Assign a color per path, keyed by group so paths in the same group share a color."""
    palette = _palette()
    group_to_color: dict[Hashable, str] = {}
    result: dict[Path, str | None] = {}
    for path, group in groups.items():
        if overrides and path in overrides:
            result[path] = overrides[path]
        else:
            if group not in group_to_color:
                group_to_color[group] = palette[len(group_to_color) % len(palette)]
            result[path] = group_to_color[group]
    return result


def _by_group(
    identity: Mapping[Path, Hashable], per_path: Mapping[Path, Any]
) -> dict[Hashable, Any]:
    """Collapse a per-path mapping to one value per series key, taking the first of each."""
    result: dict[Hashable, Any] = {}
    for path, group in identity.items():
        if group not in result:
            result[group] = per_path.get(path)
    return result


def _resolve_gate_set(gate_set: GateSet | None, model: LinearMap | None) -> GateSet | None:
    """The gate set to label with: the explicit one, else the fidelity model's."""
    from ...models import is_fidelity_model

    if gate_set is not None:
        return gate_set
    if model is not None and is_fidelity_model(model):
        return model.output_space.gate_set
    return None


def _layer_meta(layers: Sequence[Layer]) -> dict[Hashable, tuple[str, dict[str, object] | None]]:
    """Each layer's display name and legend-handle style, keyed as its render calls key its gids."""
    meta: dict[Hashable, tuple[str, dict[str, object] | None]] = {}
    for index, layer in enumerate(layers):
        key = layer.key if layer.key is not None else layer.name
        name = layer.name if layer.name is not None else f"Layer {index + 1}"
        meta.setdefault(key, (name, layer.proxy))
    return meta


def _draw_cell(
    ax: "Axes",
    cell: str,
    layers: Sequence[Layer],
    paths: Sequence[Path],
    colors: Mapping[Path, str | None],
    labels: Mapping[Path, str | None],
    identity: Mapping[Path, Hashable],
    fragment_depths: np.ndarray,
    path_tokens: TokenMap,
    layer_tokens: TokenMap,
) -> dict[str, dict[str, Any]]:
    """Draw every layer into one axes, returning the merged hover data for what they drew.

    The two token maps are the figure's, not the cell's: a series key appearing in several cells
    must resolve to one legend entry, which is what makes a single click reach every cell at once.
    """
    context = RenderContext(
        ax=ax,
        cell=cell,
        colors=colors,
        labels=labels,
        groups=identity,
        fragment_depths=fragment_depths,
        paths=paths,
        path_tokens=path_tokens,
        layer_tokens=layer_tokens,
    )
    traces: dict[str, dict[str, Any]] = {}
    for layer in layers:
        traces.update(layer.render(context))
    return traces


def _add_legends(
    fig: "Figure",
    path_entries: Sequence[tuple[str, str, str | None]],
    layer_entries: Sequence[tuple[str, str, dict[str, object] | None]],
) -> None:
    """Add the path and series legends, tagged so the browser can drive them.

    Both are placed outside the axes -- the paths in a column to the right, the series in a strip
    above -- so neither can cover data, and ``bbox_inches="tight"`` grows the saved image to fit
    however many entries there turn out to be.
    """
    from matplotlib.lines import Line2D

    if path_entries:
        legend = fig.legend(
            [Line2D([], [], color=color, linewidth=2) for _, _, color in path_entries],
            [label for _, label, _ in path_entries],
            loc="center left",
            bbox_to_anchor=(1.0, 0.5),
            title="Path",
            fontsize=_PATH_LEGEND_FONT_SIZE,
            labelspacing=_PATH_LEGEND_LABEL_SPACING,
            frameon=False,
        )
        tag_legend(legend, _PATH_DIMENSION, [token for token, _, _ in path_entries])

    if layer_entries:
        handles = [
            Line2D([], [], **{"color": _SERIES_LEGEND_COLOR, **(proxy or {"linestyle": "none"})})
            for _, _, proxy in layer_entries
        ]
        legend = fig.legend(
            handles,
            [name for _, name, _ in layer_entries],
            loc="lower left",
            bbox_to_anchor=(0.0, 1.0),
            ncols=len(layer_entries),
            title="Series",
            frameon=False,
        )
        tag_legend(legend, _LAYER_DIMENSION, [token for token, _, _ in layer_entries])


def _legend_entries(
    path_tokens: TokenMap,
    group_labels: Mapping[Hashable, str | None],
    group_colors: Mapping[Hashable, str | None],
    layer_tokens: TokenMap,
    layer_meta: Mapping[Hashable, tuple[str, dict[str, object] | None]],
) -> tuple[list[tuple[str, str, str | None]], list[tuple[str, str, dict[str, object] | None]]]:
    """Build both legends' entries from the tokens that were actually allocated while drawing.

    Only keys that reached an artist have tokens, so a layer or a path that turned out to draw
    nothing -- a model curve with no prediction for any path in the figure, say -- contributes no
    entry, rather than a control that does nothing.
    """
    paths = [
        (token, group_labels.get(group) or token, group_colors.get(group))
        for group, token in path_tokens.items()
    ]
    layers = [
        (token, *layer_meta.get(key, (str(key), None))) for key, token in layer_tokens.items()
    ]
    return paths, layers


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


@HAS_MATPLOTLIB.require_in_call
def plot_path_overlay(
    layers: Iterable[Layer],
    paths: Iterable[Path] | None = None,
    *,
    gate_set: GateSet | None = None,
    colors: Mapping[Path, str] | None = None,
    labels: Mapping[Path, str] | None = None,
    groups: Mapping[Path, str] | None = None,
    label_style: str = "formula",
    fragment_depths: Sequence[float] | np.ndarray | None = None,
    title: str | None = None,
    ax: "Axes | None" = None,
) -> InteractiveFigure:
    """Overlay an arbitrary list of decay layers on a single axes, with shared coordination.

    Each path is its own series: one color and one legend entry shared across every layer. Labels
    default to :func:`path_labels` when a gate set is available, else index strings.

    Args:
        layers: The layers to draw (e.g. from :func:`exponential_fit_curves_layer`,
            :func:`model_curves_layer`), each invoked with the shared coordination.
        paths: The paths to plot. Defaults to the union of the paths each layer contributes.
        gate_set: The gate set used to build default labels.
        colors: Optional per-path color overrides; each path is otherwise assigned its own color.
        labels: Optional per-path legend labels.
        groups: Optional per-path series-identity keys (a path's color and shared legend entry).
            Defaults to a per-path identity (each path its own color and legend entry).
        label_style: The :func:`~.path_math_label` style for default labels.
        fragment_depths: The fragment-depth range passed to curve layers. Defaults to ``0``–``10``.
        title: An optional figure title.
        ax: An existing axes to draw into. If ``None``, a figure with a single axes is created. Note
            that the legends are added either way, to the figure the axes belongs to, since a figure
            whose curves can be hidden but not restored is worse than no figure at all.

    Returns:
        The figure with the overlaid layers.
    """
    from matplotlib.figure import Figure

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

    identity = _resolve_identity(path_list, groups)
    color_map = _colors_by_group(identity, colors)
    if fragment_depths is None:
        fragment_depths = _default_fragment_depths()

    if ax is None:
        fig = Figure(figsize=(_SINGLE_WIDTH, _SINGLE_HEIGHT), layout="constrained")
        ax = fig.subplots()
    else:
        fig = ax.figure

    path_tokens, layer_tokens = TokenMap("p"), TokenMap("l")
    cell = cell_token(0)
    traces = _draw_cell(
        ax,
        cell,
        layers,
        path_list,
        color_map,
        labels,
        identity,
        np.asarray(fragment_depths, dtype=float),
        path_tokens,
        layer_tokens,
    )

    ax.set_xlabel("fragment_depth")
    ax.set_ylabel("observable")
    if title is not None:
        fig.suptitle(title)
    _add_legends(
        fig,
        *_legend_entries(
            path_tokens,
            _by_group(identity, labels),
            _by_group(identity, color_map),
            layer_tokens,
            _layer_meta(layers),
        ),
    )
    return InteractiveFigure(fig, dimensions=_DIMENSIONS, cells={cell: ax}, traces=traces)


@HAS_MATPLOTLIB.require_in_call
def plot_path_grid_overlay(
    groups: Mapping[Hashable, Sequence[Path]],
    layers: Iterable[Layer],
    *,
    num_cols: int = 3,
    gate_set: GateSet | None = None,
    label: Callable[[Path, Hashable], str] | None = None,
    group_title: Callable[[Hashable], str] | None = None,
    series_key: Callable[[Path, Hashable], Hashable] | None = None,
    colors: Mapping[Hashable, str] | None = None,
    label_style: str = "formula",
    fragment_depths: Sequence[float] | np.ndarray | None = None,
    title: str | None = None,
) -> InteractiveFigure:
    """Lay out an arbitrary list of decay layers across a grid of subplots (one per group).

    Subplot membership (``groups``) and series identity (``series_key``) are independent: color and
    the shared legend entry are keyed by ``series_key``'s value, so paths that resolve to the same
    key share a color and one legend entry across the whole grid — and one click on that entry hides
    them in every cell at once — while ``label`` controls only displayed text. The same ``layers``
    are drawn in every cell, restricted to that cell's paths.

    Args:
        groups: A mapping from a group key (subplot title) to the paths drawn in that subplot.
        layers: The layers to draw in each cell.
        num_cols: The number of subplot columns; rows are derived from the group count.
        gate_set: The gate set for default labels.
        label: A callable ``(path, group_key) -> str`` giving each path's displayed label. Defaults
            to a group-independent :func:`~.path_math_label` (formula, repeatable only), or to a
            positional index when there is no gate set to build one from.
        group_title: A callable ``(group_key) -> str`` giving each subplot's title. Defaults to
            ``str(group_key)``.
        series_key: A callable ``(path, group_key) -> Hashable`` giving each path's series identity
            (its color and shared legend entry). Defaults to the displayed ``label``.
        colors: Optional overrides mapping a series key to a color.
        label_style: The :func:`~.path_math_label` style for the default label.
        fragment_depths: The fragment-depth range passed to curve layers. Defaults to ``0``–``10``.
        title: An optional figure title.

    Returns:
        The subplot-grid figure.
    """
    from matplotlib.figure import Figure

    layers = list(layers)
    group_items = list(groups.items())
    num_rows = max(1, -(-len(group_items) // num_cols))
    fig = Figure(figsize=(_CELL_WIDTH * num_cols, _CELL_HEIGHT * num_rows), layout="constrained")
    axes = [ax for row in fig.subplots(num_rows, num_cols, squeeze=False) for ax in row]

    resolved_gate_set = _resolve_gate_set(gate_set, None)
    if fragment_depths is None:
        fragment_depths = _default_fragment_depths()
    fragment_depths = np.asarray(fragment_depths, dtype=float)

    palette = _palette()
    series_to_color: dict[Hashable, str] = {}
    # Numbers the paths across the whole grid, for the fallback label. Cell-local numbering would
    # make two unrelated paths in different cells share a label, hence a color and a legend entry.
    path_numbers: dict[Path, int] = {}

    def _label_for(path: Path, key: Hashable) -> str:
        if label is not None:
            return label(path, key)
        if resolved_gate_set is not None:
            return (
                "$"
                + path_math_label(resolved_gate_set, path, style=label_style, repeatable_only=True)
                + "$"
            )
        return str(path_numbers.setdefault(path, len(path_numbers)))

    def _color_for(series_val: Hashable) -> str:
        if colors and series_val in colors:
            return colors[series_val]
        if series_val not in series_to_color:
            series_to_color[series_val] = palette[len(series_to_color) % len(palette)]
        return series_to_color[series_val]

    path_tokens, layer_tokens = TokenMap("p"), TokenMap("l")
    cells: dict[str, Axes] = {}
    traces: dict[str, dict[str, Any]] = {}
    group_labels: dict[Hashable, str] = {}
    group_colors: dict[Hashable, str] = {}

    for index, (key, group_paths) in enumerate(group_items):
        ax = axes[index]
        cell = cell_token(index)
        group_paths = list(group_paths)

        cell_labels = {path: _label_for(path, key) for path in group_paths}
        cell_identity = {
            path: (series_key(path, key) if series_key is not None else cell_labels[path])
            for path in group_paths
        }
        cell_identity = _resolve_identity(group_paths, cell_identity)
        cell_colors = {path: _color_for(cell_identity[path]) for path in group_paths}
        for path in group_paths:
            group_labels.setdefault(cell_identity[path], cell_labels[path])
            group_colors.setdefault(cell_identity[path], cell_colors[path])

        traces.update(
            _draw_cell(
                ax,
                cell,
                layers,
                group_paths,
                cell_colors,
                cell_labels,
                cell_identity,
                fragment_depths,
                path_tokens,
                layer_tokens,
            )
        )
        ax.set_title((group_title or str)(key), fontsize=_CELL_TITLE_FONT_SIZE)
        cells[cell] = ax

    # Cells past the last group have no data; leaving their frames drawn would read as an empty
    # result rather than as no result at all.
    for ax in axes[len(group_items) :]:
        ax.set_axis_off()

    fig.supxlabel("fragment_depth")
    fig.supylabel("observable")
    if title is not None:
        fig.suptitle(title)
    _add_legends(
        fig,
        *_legend_entries(
            path_tokens, group_labels, group_colors, layer_tokens, _layer_meta(layers)
        ),
    )
    return InteractiveFigure(fig, dimensions=_DIMENSIONS, cells=cells, traces=traces)


@HAS_MATPLOTLIB.require_in_call
def plot_qubit_pair_decays(
    pairs: Sequence[tuple[int, int]],
    *,
    observable_data: ObservableData | None = None,
    observable_type: Literal["raw", "means", "both"] = "raw",
    observable_marker_kwargs: Mapping[str, object] | None = None,
    means_marker_kwargs: Mapping[str, object] | None = None,
    averaged_data: AveragedData | None = None,
    exponential_fit_line_kwargs: Mapping[str, object] | None = None,
    model: LinearMap | None = None,
    model_data: ModelData | None = None,
    model_line_kwargs: Mapping[str, object] | None = None,
    gate_set: GateSet | None = None,
    num_cols: int = 3,
    colors: Mapping[Hashable, str] | None = None,
    label_style: str = "formula",
    noise_site: Mapping[str, str] | None = None,
    placeholders: tuple[str, str] = ("i", "j"),
    paths: Iterable[Path] | None = None,
    fragment_depths: Sequence[float] | np.ndarray | None = None,
    title: str | None = None,
) -> InteractiveFigure:
    """Grid of fidelity decays over qubit pairs, one subplot per pair, with shared labels.

    Each subplot shows the decays for the paths acting on that pair, where a path acts on the qubits
    that its transition Paulis are supported on (a path is assigned to ``pair`` when ``pair`` is a
    superset of that support). Series labels are **canonicalized**: the pair's qubits are relabeled
    to ``placeholders`` (min qubit -> ``"i"``, max -> ``"j"``), so a given Pauli fidelity (e.g.
    ``X_{i} X_{j}``) shares a color and a single legend entry across every pair it appears in.
    Subplot titles show the actual pair.

    Args:
        pairs: The qubit pairs to plot, one subplot each.
        observable_data: Optional raw observable data for scatter points.
        observable_type: Which observable layer(s) to draw from ``observable_data`` — ``"raw"``,
            ``"means"``, or ``"both"`` (see :func:`standard_decay_layers`).
        observable_marker_kwargs: Optional marker properties for the raw observable points.
        means_marker_kwargs: Optional marker properties for the observable-means points.
        averaged_data: Optional averaged data supplying the exponential-fit decay curve.
        exponential_fit_line_kwargs: Optional line properties for the exponential-fit curve.
        model: Optional fidelity model for predicted curves (requires ``model_data``).
        model_data: Optional fitted parameters for predicted curves (requires ``model``).
        model_line_kwargs: Optional line properties for the model curves.
        gate_set: The gate set used to build labels. Defaults to the model's gate set; required
            (here or via the model) since labels are always drawn.
        num_cols: The number of subplot columns; rows are derived from the pair count.
        colors: Optional overrides mapping a (canonicalized) series label to a color.
        label_style: The :func:`~.path_math_label` style for the series labels.
        noise_site: An optional noise-site mapping forwarded to :func:`~.path_math_label` (with
            ``style="formula"`` this yields the compact ``f^{gate}_{pauli}`` label).
        placeholders: The two display symbols for the pair's (min, max) qubit indices.
        paths: The set of paths to draw across all layers. Defaults to the decay paths found in
            ``observable_data``/``averaged_data``. Supply this to plot decays that cannot be derived
            from empirical data — most notably model curves with no observable or averaged data
            present. When given, it scopes every layer (each layer still only draws a path for which
            its own data source has an entry); non-decay paths are dropped.
        fragment_depths: The fragment-depth range for the curves. Defaults to ``0`` through the
            largest fragment depth in the empirical data present (observable and averaged-data
            points), or ``0``–``10`` when there is none.
        title: An optional figure title.

    Returns:
        The subplot-grid figure.

    Raises:
        ValueError: If no gate set is available (neither ``gate_set`` nor a model with one).
    """
    resolved_gate_set = _resolve_gate_set(gate_set, model)
    if resolved_gate_set is None:
        raise ValueError("A gate_set (or a model carrying one) is required to label the decays.")

    # Resolve the path set (explicit wins, else derived from the empirical data) and restrict it to
    # decay paths (unbound, non-empty repeatable fragment).
    if paths is None:
        candidate_paths = _dataset_paths(observable_data, averaged_data)
    else:
        candidate_paths = list(paths)
    paths = [path for path in candidate_paths if path.is_unbound and path.repeatable_fragment]

    # Default the fragment-depth range to span the empirical data actually present, so the fitted
    # and model curves extend across the observed fragment depths rather than the generic 0-10
    # fallback.
    if fragment_depths is None:
        empirical_points: list[dict] = []
        if observable_data is not None:
            empirical_points.append(observable_data_points(observable_data, paths))
        if averaged_data is not None:
            empirical_points.append(averaged_data_points(averaged_data, paths))
        fragment_depths = _default_fragment_depths(*empirical_points)

    groups: dict[Hashable, list[Path]] = {}
    for pair in pairs:
        pair_set = set(pair)
        groups[tuple(pair)] = [path for path in paths if pair_set.issuperset(_path_qubits(path))]

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

    def _group_title(pair: Hashable) -> str:
        low, high = sorted(pair)
        return f"({placeholders[0]}, {placeholders[1]}) = ({low}, {high})"

    layers = standard_decay_layers(
        observable_data=observable_data,
        observable_type=observable_type,
        observable_marker_kwargs=observable_marker_kwargs,
        means_marker_kwargs=means_marker_kwargs,
        averaged_data=averaged_data,
        exponential_fit_line_kwargs=exponential_fit_line_kwargs,
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
        group_title=_group_title,
        colors=colors,
        fragment_depths=fragment_depths,
        title=title,
    )
