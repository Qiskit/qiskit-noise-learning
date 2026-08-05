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

A :class:`Layer` bundles a render callable (which draws one kind of decay data into an axes given
shared coordination) with the metadata an orchestrator needs: its series-legend ``name``/``proxy``,
the ``key`` its legend entry and its artists' ``gid``\\ s are built from, and the ``paths`` it
contributes for path resolution. The ``*_layer`` builders construct these from a data source;
:func:`standard_decay_layers` assembles the observable/exponential-fit/model stack.
"""

from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from ...data import AveragedData, ModelData, ObservableData
from ...math import LinearMap
from ...sequences import Path
from ..interactive_figure import TokenMap, trace_gid
from .data_adapters import (
    _dataset_paths,
    averaged_data_points,
    exponential_fit_curves,
    observable_data_points,
)
from .primitives import plot_path_decay_curves, plot_path_scatters

if TYPE_CHECKING:
    from matplotlib.axes import Axes


#: The marker symbol or line style each standard layer draws with, so that a reader can tell
#: the layers apart in a static export, where the legends cannot be clicked.
_OBSERVABLE_POINTS_MARKER = "o"
_AVERAGED_POINTS_MARKER = "x"
_FIT_LINESTYLE = "--"
_MODEL_LINESTYLE = "-"


@dataclass(frozen=True, eq=False)
class RenderContext:
    """The shared coordination an orchestrator injects into each :class:`Layer`'s render call.

    An orchestrator resolves the per-path color/label/group identities, the fragment-depth range,
    and the target subplot cell once, then passes this bundle to every layer so their artists line
    up.

    Args:
        ax: The axes to draw into.
        cell: The token naming that axes within its figure, from :func:`~.cell_token`. Part of every
            ``gid``, since the same path and layer are drawn in every cell of a grid and their
            artists still need distinct SVG ids.
        colors: A mapping from path to its matplotlib color (``None`` for the axes' cycle color).
        labels: A mapping from path to the label its hover readout shows.
        groups: A mapping from path to its series identity -- the key it shares a color and a path
            legend entry with. Paths with equal keys toggle together.
        fragment_depths: The fragment-depth values (x) at which curve layers evaluate their decays.
        paths: The paths to draw in this cell.
        path_tokens: The figure-wide token allocation for ``groups`` values. Shared across cells, so
            one path legend entry reaches every cell its key appears in.
        layer_tokens: The figure-wide token allocation for layer keys, likewise shared.
    """

    ax: "Axes"
    cell: str
    colors: Mapping[Path, str | None]
    labels: Mapping[Path, str | None]
    groups: Mapping[Path, Hashable]
    fragment_depths: np.ndarray
    paths: Sequence[Path]
    path_tokens: TokenMap
    layer_tokens: TokenMap

    def gid_factory(self, layer_key: Hashable) -> Callable[[Path, int], str]:
        """Return the ``gid`` builder a layer tags its artists with.

        Args:
            layer_key: The layer's identity, which must be the :attr:`Layer.key` the orchestrator
                will build that layer's series-legend entry from -- the two are matched by value,
                and a layer that asks under some other key gets an entry that switches nothing.

        Returns:
            A callable ``(path, artist_index) -> gid`` to pass to a primitive's ``gid`` argument.
        """

        def gid(path: Path, index: int) -> str:
            # Both tokens are allocated here rather than up front, so that a key only enters the
            # legend once something has actually been drawn under it. A layer with no data for any
            # path in this figure would otherwise contribute an entry that switches nothing.
            path_token = self.path_tokens.token(self.groups[path])
            layer_token = self.layer_tokens.token(layer_key)
            # In the order the orchestrators declare ``DIMENSIONS``, which is how the browser knows
            # which of the two tokens belongs to which legend.
            return trace_gid(self.cell, (path_token, layer_token), index)

        return gid


@dataclass(frozen=True, eq=False)
class Layer:
    """A coordinated render unit for an overlay.

    Args:
        render: A callable ``(context) -> dict`` which draws this layer's artists into
            ``context.ax`` using the injected :class:`RenderContext` coordination, and returns the
            hover data for what it drew (as the primitives do).
        name: The series-legend display name, or ``None`` for a name derived from the layer's
            position in the stack.
        proxy: The series-legend proxy style, as matplotlib :class:`~matplotlib.lines.Line2D`
            properties (e.g. ``{"marker": "o", "linestyle": "none"}``), or ``None`` for an entry
            whose handle draws nothing.
        paths: The paths this layer contributes when an orchestrator resolves the plotted path set
            (empty for layers, like the model curve, that carry no paths of their own).
        key: This layer's identity, distinguishing its series-legend entry and its artists'
            ``gid``\\ s from other layers'. Defaults to ``name``. Two layers sharing a key share one
            legend entry and toggle as one.
    """

    render: Callable[[RenderContext], dict[str, dict[str, Any]]]
    name: str | None = None
    proxy: dict[str, object] | None = None
    paths: tuple[Path, ...] = field(default_factory=tuple)
    key: Hashable | None = None


def observable_points_layer(
    observable_data: ObservableData, *, marker_kwargs: Mapping[str, object] | None = None
) -> Layer:
    """A layer scattering raw per-randomization observable points (default ``o`` marker).

    Args:
        observable_data: The raw observable data to scatter.
        marker_kwargs: Optional matplotlib properties for the markers (e.g. ``marker``,
            ``markersize``, ``alpha``).

    Returns:
        The layer.
    """
    key = "Observable points"
    style = {"marker": _OBSERVABLE_POINTS_MARKER, **(marker_kwargs or {})}

    def render(ctx: RenderContext) -> dict[str, dict[str, Any]]:
        return plot_path_scatters(
            observable_data_points(observable_data, ctx.paths),
            ctx.ax,
            colors=ctx.colors,
            labels=ctx.labels,
            gid=ctx.gid_factory(key),
            marker_kwargs=style,
        )

    return Layer(
        render,
        name=key,
        key=key,
        proxy={**style, "linestyle": "none"},
        paths=tuple(_dataset_paths(observable_data)),
    )


def observable_means_layer(
    observable_data: ObservableData, *, marker_kwargs: Mapping[str, object] | None = None
) -> Layer:
    """A layer scattering per-fragment-depth means of raw observable data (default ``x`` marker).

    Averages over randomizations via :class:`~.AverageObservables`, so each path shows one
    error-barred point per fragment depth rather than the raw per-randomization cloud.

    Args:
        observable_data: The raw observable data to average and scatter.
        marker_kwargs: Optional matplotlib properties for the markers (e.g. ``marker``,
            ``markersize``, ``alpha``).

    Returns:
        The layer.
    """
    key = "Observable means"
    style = {"marker": _AVERAGED_POINTS_MARKER, **(marker_kwargs or {})}

    def render(ctx: RenderContext) -> dict[str, dict[str, Any]]:
        from ...analysis.average_observables import average_observables

        averaged = average_observables(observable_data, set(ctx.paths))
        return plot_path_scatters(
            averaged_data_points(averaged, ctx.paths),
            ctx.ax,
            colors=ctx.colors,
            labels=ctx.labels,
            gid=ctx.gid_factory(key),
            marker_kwargs=style,
        )

    return Layer(
        render,
        name=key,
        key=key,
        proxy={**style, "linestyle": "none"},
        paths=tuple(_dataset_paths(observable_data)),
    )


def exponential_fit_curves_layer(
    averaged_data: AveragedData, *, line_kwargs: Mapping[str, object] | None = None
) -> Layer:
    """A layer drawing the exponential-fit decay curves from averaged data (default dashed line).

    Args:
        averaged_data: The averaged data whose fitted decays to draw.
        line_kwargs: Optional matplotlib properties for the lines (e.g. ``linestyle``,
            ``linewidth``, ``alpha``).

    Returns:
        The layer.
    """
    key = "Exponential fit"
    style = {"linestyle": _FIT_LINESTYLE, **(line_kwargs or {})}

    def render(ctx: RenderContext) -> dict[str, dict[str, Any]]:
        bases, intercepts = exponential_fit_curves(averaged_data, ctx.paths)
        return plot_path_decay_curves(
            bases,
            intercepts,
            ctx.fragment_depths,
            ctx.ax,
            colors=ctx.colors,
            labels=ctx.labels,
            gid=ctx.gid_factory(key),
            line_kwargs=style,
        )

    return Layer(
        render,
        name=key,
        key=key,
        proxy=dict(style),
        paths=tuple(_dataset_paths(averaged_data)),
    )


def model_curves_layer(
    model: LinearMap, model_data: ModelData, *, line_kwargs: Mapping[str, object] | None = None
) -> Layer:
    """A layer drawing model-predicted decay curves (default solid line).

    Carries no paths of its own -- it renders whatever paths the orchestrator resolves from the
    other layers (or the caller supplies explicitly).

    Args:
        model: The fidelity model to predict decays with.
        model_data: The fitted parameters to predict decays from.
        line_kwargs: Optional matplotlib properties for the lines (e.g. ``linestyle``,
            ``linewidth``, ``alpha``).

    Returns:
        The layer.
    """
    key = "Model"
    style = {"linestyle": _MODEL_LINESTYLE, **(line_kwargs or {})}

    def render(ctx: RenderContext) -> dict[str, dict[str, Any]]:
        from ...analysis.utils import predicted_path_decays

        bases, intercepts = predicted_path_decays(model, model_data, ctx.paths)
        return plot_path_decay_curves(
            bases,
            intercepts,
            ctx.fragment_depths,
            ctx.ax,
            colors=ctx.colors,
            labels=ctx.labels,
            gid=ctx.gid_factory(key),
            line_kwargs=style,
        )

    return Layer(render, name=key, key=key, proxy=dict(style))


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
        observable_type: Which observable layer(s) to draw from ``observable_data`` -- ``"raw"``
            (raw per-randomization scatter), ``"means"`` (per-fragment-depth means with error bars,
            averaged via :class:`~.AverageObservables`), or ``"both"``. The raw and means layers are
            styled independently (defaulting to an ``o`` and an ``x`` marker respectively).
        observable_marker_kwargs: Optional marker overrides for the raw observable scatter.
        means_marker_kwargs: Optional marker overrides for the observable-means scatter.
        averaged_data: Optional averaged data supplying the exponential-fit decay curve.
        exponential_fit_line_kwargs: Optional line overrides for the exponential-fit curve.
        model: Optional fidelity model for predicted curves (requires ``model_data``).
        model_data: Optional fitted parameters for predicted curves (requires ``model``).
        model_line_kwargs: Optional line overrides for the model curve.

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
