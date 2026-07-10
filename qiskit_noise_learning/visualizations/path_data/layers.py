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
a data source; :func:`standard_decay_layers` assembles the observable/averaged/fit/model stack.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from .data_adapters import (
    _dataset_paths,
    averaged_data_curves,
    averaged_data_points,
    model_curves,
    observable_data_points,
)
from .primitives import _default_depths, plot_path_decay_curves, plot_path_scatters

if TYPE_CHECKING:
    import plotly.graph_objects as go

    from ...data import AveragedData, ModelData, ObservableData
    from ...math import LinearMap
    from ...sequences import Path


# Default per-layer marker symbols / line dashes, reflected in both the rendered traces and the
# symbol-legend proxies.
_OBSERVABLE_POINTS_SYMBOL = "circle"
_AVERAGED_POINTS_SYMBOL = "x"
_FIT_DASH = "solid"
_MODEL_DASH = "dash"


@dataclass(frozen=True, eq=False)
class Layer:
    """A coordinated render unit for an overlay.

    Args:
        render: A callable ``(*, fig, colors, labels, groups, depths, paths, row, col)`` returning a
            figure, which draws this layer's traces into ``fig`` using the injected coordination.
        name: The symbol-legend display name, or ``None`` to omit the layer from the symbol legend.
        proxy: The symbol-legend proxy style (a ``{"mode": ..., "marker"|"line": {...}}`` dict), or
            ``None``.
        paths: The paths this layer contributes when an orchestrator resolves the plotted path set
            (empty for layers, like the model curve, that carry no paths of their own).
    """

    render: Callable[..., go.Figure]
    name: str | None = None
    proxy: dict | None = None
    paths: tuple[Path, ...] = field(default_factory=tuple)


def observable_points_layer(
    observable_data: ObservableData, *, marker_kwargs: Mapping[str, object] | None = None
) -> Layer:
    """A layer scattering raw per-randomization observable points (default ``circle`` marker)."""
    marker = {"symbol": _OBSERVABLE_POINTS_SYMBOL, **(marker_kwargs or {})}

    def render(*, fig, colors, labels, groups, paths, row, col, **_):
        return plot_path_scatters(
            observable_data_points(observable_data, paths),
            fig=fig,
            colors=colors,
            labels=labels,
            groups=groups,
            marker_kwargs=marker,
            row=row,
            col=col,
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

    def render(*, fig, colors, labels, groups, paths, row, col, **_):
        from ...analysis.average_observables import average_observables

        averaged = average_observables(observable_data, set(paths) if paths is not None else None)
        return plot_path_scatters(
            averaged_data_points(averaged, paths),
            fig=fig,
            colors=colors,
            labels=labels,
            groups=groups,
            marker_kwargs=marker,
            row=row,
            col=col,
        )

    return Layer(
        render,
        name="Observable means",
        proxy={"mode": "markers", "marker": marker},
        paths=tuple(_dataset_paths(observable_data)),
    )


def averaged_points_layer(
    averaged_data: AveragedData, *, marker_kwargs: Mapping[str, object] | None = None
) -> Layer:
    """A layer scattering averaged points with error bars (default ``x`` marker)."""
    marker = {"symbol": _AVERAGED_POINTS_SYMBOL, **(marker_kwargs or {})}

    def render(*, fig, colors, labels, groups, paths, row, col, **_):
        return plot_path_scatters(
            averaged_data_points(averaged_data, paths),
            fig=fig,
            colors=colors,
            labels=labels,
            groups=groups,
            marker_kwargs=marker,
            row=row,
            col=col,
        )

    return Layer(
        render,
        name="Averaged points",
        proxy={"mode": "markers", "marker": marker},
        paths=tuple(_dataset_paths(averaged_data)),
    )


def fit_curves_layer(
    averaged_data: AveragedData, *, line_kwargs: Mapping[str, object] | None = None
) -> Layer:
    """A layer drawing the fitted decay curves from averaged data (default solid ``dash``)."""
    line = {"dash": _FIT_DASH, **(line_kwargs or {})}

    def render(*, fig, colors, labels, groups, depths, paths, row, col, **_):
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
            line_kwargs=line,
            row=row,
            col=col,
        )

    return Layer(
        render,
        name="Fit",
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

    def render(*, fig, colors, labels, groups, depths, paths, row, col, **_):
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
            line_kwargs=line,
            row=row,
            col=col,
        )

    return Layer(render, name="Model", proxy={"mode": "lines", "line": line})


def standard_decay_layers(
    *,
    observable_data: ObservableData | None = None,
    observable_type: Literal["raw", "means", "both"] = "raw",
    observable_marker_kwargs: Mapping[str, object] | None = None,
    means_marker_kwargs: Mapping[str, object] | None = None,
    averaged_data: AveragedData | None = None,
    averaged_points: bool = True,
    averaged_marker_kwargs: Mapping[str, object] | None = None,
    averaged_line_kwargs: Mapping[str, object] | None = None,
    model: LinearMap | None = None,
    model_data: ModelData | None = None,
    model_line_kwargs: Mapping[str, object] | None = None,
) -> list[Layer]:
    """Build the standard decay layer stack for the supplied data sources.

    Includes observable-scatter layer(s) (if ``observable_data``), averaged-points and fit-curve
    layers (if ``averaged_data``), and a model-curve layer (if ``model`` and ``model_data``). Pass
    the result to :func:`~.plot_path_overlay` or :func:`~.plot_path_grid_overlay`.

    Args:
        observable_data: Optional raw observable data.
        observable_type: Which observable layer(s) to draw from ``observable_data`` — ``"raw"`` (raw
            per-randomization scatter), ``"means"`` (per-depth means with error bars, averaged via
            :class:`~.AverageObservables`), or ``"both"``. The raw and means layers are styled
            independently (defaulting to a ``circle`` and an ``x`` symbol respectively).
        observable_marker_kwargs: Optional ``marker`` overrides for the raw observable scatter.
        means_marker_kwargs: Optional ``marker`` overrides for the observable-means scatter.
        averaged_data: Optional averaged data (averaged points + fitted curve).
        averaged_points: Whether to include the averaged-points layer. The fitted curve is always
            drawn when ``averaged_data`` is given; set this ``False`` to draw only the curve (e.g.
            when empirical points are already supplied via ``observable_data``).
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
        if averaged_points:
            layers.append(
                averaged_points_layer(averaged_data, marker_kwargs=averaged_marker_kwargs)
            )
        layers.append(fit_curves_layer(averaged_data, line_kwargs=averaged_line_kwargs))
    if model is not None and model_data is not None:
        layers.append(model_curves_layer(model, model_data, line_kwargs=model_line_kwargs))
    return layers
