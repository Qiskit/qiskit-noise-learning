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

"""Layers: self-describing, coordinated render units, and the standard decay stack.

A :class:`Layer` bundles a render callable (which draws one kind of decay data into a figure given
shared coordination) with the metadata an orchestrator needs: its symbol-legend ``name``/``proxy``
and the ``paths`` it contributes for path resolution. The ``*_layer`` builders construct these from
a data source; :func:`standard_decay_layers` assembles the observable/exponential-fit/model stack.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np

from ...data import AveragedData, ModelData, ObservableData
from ...math import LinearMap
from ...sequences import Path
from .data_adapters import (
    _dataset_paths,
    averaged_data_points,
    exponential_fit_curves,
    observable_data_points,
)
from .primitives import plot_path_decay_curves, plot_path_scatters

if TYPE_CHECKING:
    import plotly.graph_objects as go


# Default per-layer marker symbols / line dashes, reflected in both the rendered traces and the
# symbol-legend proxies.
_OBSERVABLE_POINTS_SYMBOL = "circle"
_AVERAGED_POINTS_SYMBOL = "x"
_FIT_DASH = "dash"
_MODEL_DASH = "solid"


@dataclass(frozen=True, eq=False)
class RenderContext:
    """The shared coordination an orchestrator injects into each :class:`Layer`'s render call.

    An orchestrator resolves the per-path color/label/group identities, the depth range, and the
    target subplot cell once, then passes this bundle to every layer so their traces line up.

    Args:
        fig: The figure to add traces to.
        colors: A mapping from path to its plotly color string (``None`` for the plotly default).
        labels: A mapping from path to its legend label.
        groups: A mapping from path to its ``legendgroup`` key (``None`` to omit from the legend).
        depths: The depth values (x) at which curve layers evaluate their decays.
        paths: The paths to draw in this cell.
        row: The subplot row to add traces to (1-indexed), or ``None`` for a single-axes figure.
        col: The subplot column to add traces to (1-indexed), or ``None`` for a single-axes figure.
    """

    fig: "go.Figure"
    colors: Mapping[Path, str | None]
    labels: Mapping[Path, str | None]
    groups: Mapping[Path, str | None]
    depths: np.ndarray
    paths: Sequence[Path]
    row: int | None = None
    col: int | None = None


@dataclass(frozen=True, eq=False)
class Layer:
    """A coordinated render unit for an overlay.

    Args:
        render: A callable ``(context) -> go.Figure`` which draws this layer's traces into
            ``context.fig`` using the injected :class:`RenderContext` coordination.
        name: The symbol-legend display name, or ``None`` to omit the layer from the symbol legend.
        proxy: The symbol-legend proxy style (a ``{"mode": ..., "marker"|"line": {...}}`` dict), or
            ``None``.
        paths: The paths this layer contributes when an orchestrator resolves the plotted path set
            (empty for layers, like the model curve, that carry no paths of their own).
    """

    render: Callable[[RenderContext], "go.Figure"]
    name: str | None = None
    proxy: dict[str, object] | None = None
    paths: tuple[Path, ...] = field(default_factory=tuple)


def observable_points_layer(
    observable_data: ObservableData, *, marker_kwargs: Mapping[str, object] | None = None
) -> Layer:
    """A layer scattering raw per-randomization observable points (default ``circle`` marker)."""
    marker = {"symbol": _OBSERVABLE_POINTS_SYMBOL, **(marker_kwargs or {})}

    def render(ctx: RenderContext) -> "go.Figure":
        return plot_path_scatters(
            observable_data_points(observable_data, ctx.paths),
            fig=ctx.fig,
            colors=ctx.colors,
            labels=ctx.labels,
            groups=ctx.groups,
            marker_kwargs=marker,
            row=ctx.row,
            col=ctx.col,
        )

    return Layer(
        render,
        name="Observable points",
        proxy={"mode": "markers", "marker": marker},
        paths=tuple(_dataset_paths(observable_data)),
    )


def observable_means_layer(
    observable_data: ObservableData, *, marker_kwargs: Mapping[str, object] | None = None
) -> Layer:
    """A layer scattering per-depth means of raw observable data (default ``x`` marker).

    Averages over randomizations via :class:`~.AverageObservables`, so each path shows one
    error-barred point per depth rather than the raw per-randomization cloud.
    """
    marker = {"symbol": _AVERAGED_POINTS_SYMBOL, **(marker_kwargs or {})}

    def render(ctx: RenderContext) -> "go.Figure":
        from ...analysis.average_observables import average_observables

        averaged = average_observables(
            observable_data, set(ctx.paths) if ctx.paths is not None else None
        )
        return plot_path_scatters(
            averaged_data_points(averaged, ctx.paths),
            fig=ctx.fig,
            colors=ctx.colors,
            labels=ctx.labels,
            groups=ctx.groups,
            marker_kwargs=marker,
            row=ctx.row,
            col=ctx.col,
        )

    return Layer(
        render,
        name="Observable means",
        proxy={"mode": "markers", "marker": marker},
        paths=tuple(_dataset_paths(observable_data)),
    )


def exponential_fit_curves_layer(
    averaged_data: AveragedData, *, line_kwargs: Mapping[str, object] | None = None
) -> Layer:
    """A layer drawing the exponential-fit decay curves from averaged data (default solid line)."""
    line = {"dash": _FIT_DASH, **(line_kwargs or {})}

    def render(ctx: RenderContext) -> "go.Figure":
        bases, intercepts = exponential_fit_curves(averaged_data, ctx.paths)
        return plot_path_decay_curves(
            bases,
            intercepts,
            ctx.depths,
            fig=ctx.fig,
            colors=ctx.colors,
            labels=ctx.labels,
            groups=ctx.groups,
            line_kwargs=line,
            row=ctx.row,
            col=ctx.col,
        )

    return Layer(
        render,
        name="Exponential fit",
        proxy={"mode": "lines", "line": line},
        paths=tuple(_dataset_paths(averaged_data)),
    )


def model_curves_layer(
    model: LinearMap, model_data: ModelData, *, line_kwargs: Mapping[str, object] | None = None
) -> Layer:
    """A layer drawing model-predicted decay curves (default dashed ``dash``).

    Carries no paths of its own — it renders whatever paths the orchestrator resolves from the other
    layers (or the caller supplies explicitly).
    """
    line = {"dash": _MODEL_DASH, **(line_kwargs or {})}

    def render(ctx: RenderContext) -> "go.Figure":
        from ...analysis.utils import predicted_path_decays

        bases, intercepts = predicted_path_decays(model, model_data, ctx.paths)
        return plot_path_decay_curves(
            bases,
            intercepts,
            ctx.depths,
            fig=ctx.fig,
            colors=ctx.colors,
            labels=ctx.labels,
            groups=ctx.groups,
            line_kwargs=line,
            row=ctx.row,
            col=ctx.col,
        )

    return Layer(render, name="Model", proxy={"mode": "lines", "line": line})


def standard_decay_layers(
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
) -> list[Layer]:
    """Build the standard decay layer stack for the supplied data sources.

    Includes observable-scatter layer(s) (if ``observable_data``), the exponential-fit decay curve
    (if ``averaged_data``), and a model-curve layer (if ``model`` and ``model_data``). Pass the
    result to :func:`~.plot_path_overlay` or :func:`~.plot_path_grid_overlay`.

    Args:
        observable_data: Optional raw observable data.
        observable_type: Which observable layer(s) to draw from ``observable_data`` — ``"raw"`` (raw
            per-randomization scatter), ``"means"`` (per-depth means with error bars, averaged via
            :class:`~.AverageObservables`), or ``"both"``. The raw and means layers are styled
            independently (defaulting to a ``circle`` and an ``x`` symbol respectively).
        observable_marker_kwargs: Optional ``marker`` overrides for the raw observable scatter.
        means_marker_kwargs: Optional ``marker`` overrides for the observable-means scatter.
        averaged_data: Optional averaged data supplying the exponential-fit decay curve.
        exponential_fit_line_kwargs: Optional ``line`` overrides for the exponential-fit curve.
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
        layers.append(
            exponential_fit_curves_layer(averaged_data, line_kwargs=exponential_fit_line_kwargs)
        )
    if model is not None and model_data is not None:
        layers.append(model_curves_layer(model, model_data, line_kwargs=model_line_kwargs))
    return layers
