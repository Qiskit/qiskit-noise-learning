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

"""Matplotlib figures that toggle their own artists in the browser.

A figure is drawn entirely by matplotlib -- mathematics included -- and then annotated so that a
small script can hide and show parts of it.  Two consequences are worth knowing:

* Mathematics is typeset when the figure is built, by matplotlib's mathtext, so nothing about the
  page a figure lands on can affect it and the interactive and static renderings cannot disagree.
  The cost is mathtext's narrower macro coverage.
* Interactivity is attached, not drawn.  matplotlib writes an artist's
  :meth:`~matplotlib.artist.Artist.set_gid` value into the SVG as an ``id``, which is all the script
  needs to find an artist and hide it.  Legends are ordinary matplotlib legends, so they are part of
  the image and export unchanged.

An artist names one key per dimension it can be hidden by, and is shown only while none of those
keys is switched off, which is what lets the legends act independently.
"""

import io
import json
import re
import uuid
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..optionals import HAS_MATPLOTLIB

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.legend import Legend

#: Separator between the fields of a ``gid``.  Chosen because it cannot occur in the fields
#: themselves: every field is either a fixed word or a token from a :class:`TokenMap`.
_SEP = "|"

_JAVASCRIPT_PATH = Path(__file__).parent / "_static" / "qnl_interactive.js"

#: Smallest size, in CSS pixels, the reader can shrink a figure to by dragging it.  Small enough to
#: get a wide figure out of the way, large enough that it can always be dragged back.
_MIN_FIGURE_PX = (160, 120)

#: How many of each unit a browser puts in an inch.  A point is 1/72 inch and a CSS pixel is 1/96,
#: so a figure matplotlib sizes in points asks a browser for a third more pixels than it has points
#: -- which is most of why an untouched matplotlib SVG overflows the column it is dropped into.
_CSS_UNITS_PER_INCH = {"pt": 72.0, "px": 96.0, "in": 1.0}

_SVG_ROOT_PATTERN = re.compile(r"<svg\b[^>]*>")
_SVG_LENGTH_PATTERN = re.compile(r'\s(width|height)="([0-9.]+)(pt|px|in)?"')

#: Fields of a trace ``gid`` that are not dimension tokens: the word ``trace``, the cell, and the
#: artist's number.
_TRACE_GID_FIXED_FIELDS = 3


class TokenMap:
    """Short, XML-safe names for the keys that ``gid``\\ s are built from.

    A ``gid`` becomes an SVG ``id`` verbatim, so it must be unique in the document and cannot
    contain arbitrary text.  Keys satisfy neither constraint: one may be any text at all, and two
    distinct keys may share a display label.  Numbering them sidesteps both, and costs nothing in
    legibility, since the reader never sees a token -- labels travel separately, in the JSON
    sidecar, where they have no syntax to satisfy.

    Args:
        prefix: Letter placed before each number, distinguishing tokens from different maps at a
            glance when reading the generated SVG.
    """

    def __init__(self, prefix: str):
        self._prefix = prefix
        self._tokens: dict[Any, str] = {}

    def token(self, key: Any) -> str:
        """Return the token for ``key``, allocating one if this is the first time it is seen.

        Args:
            key: Any hashable key.  Keys are compared by equality, so two equal keys share a token.

        Returns:
            The token for ``key``.
        """
        if key not in self._tokens:
            self._tokens[key] = f"{self._prefix}{len(self._tokens)}"
        return self._tokens[key]

    def __contains__(self, key: Any) -> bool:
        return key in self._tokens

    def __len__(self) -> int:
        return len(self._tokens)

    def __iter__(self) -> Iterator[Any]:
        return iter(self._tokens)

    def items(self) -> Iterator[tuple[Any, str]]:
        """Iterate over ``(key, token)`` pairs in allocation order."""
        return iter(self._tokens.items())


def trace_gid(cell: str, tokens: Sequence[str], index: int = 0) -> str:
    """Return the ``gid`` for one data artist.

    The ``gid`` names the artist's key in each of its figure's dimensions, rather than assigning the
    artist to a single group, which is what lets the legends act independently.

    Args:
        cell: Token of the subplot the artist is drawn in, from :func:`cell_token`.
        tokens: The artist's key in each dimension, in the order the figure declares its
            :attr:`~InteractiveFigure.dimensions` -- the two orders being the same is how the
            browser reads a key back out of an ``id``.
        index: Distinguishes artists that share every key.  A single series is often several
            artists -- a line plus its error bars -- and each needs a distinct SVG ``id``.

    Returns:
        The ``gid`` to pass to :meth:`~matplotlib.artist.Artist.set_gid`.
    """
    return _SEP.join(["trace", cell, *tokens, str(index)])


def legend_gid(dimension: str, token: str, part: str) -> str:
    """Return the ``gid`` for one legend entry.

    Args:
        dimension: Which of its figure's :attr:`~InteractiveFigure.dimensions` this entry switches.
        token: Token of the key this entry switches.
        part: ``"handle"`` for the sample line or marker, ``"text"`` for the label.  Both are
            tagged, so that clicking either one works.

    Returns:
        The ``gid`` to pass to :meth:`~matplotlib.artist.Artist.set_gid`.
    """
    return _SEP.join(["key", dimension, token, part])


def tag_legend(legend: "Legend", dimension: str, tokens: Sequence[str]) -> None:
    """Tag a legend's entries so the browser can tell which key each one switches.

    Both parts of every entry are tagged, in the order the entries were passed to the legend.  It is
    the legend's own artists that are tagged, not the proxies handed to it: a legend draws its own
    copies of the handles it is given, so tagging the originals would name artists that are never in
    the figure.

    Args:
        legend: The legend to tag, as returned by :meth:`~matplotlib.figure.Figure.legend`.
        dimension: Which of its figure's :attr:`~InteractiveFigure.dimensions` this legend switches.
        tokens: The token each entry switches, parallel to the handles the legend was built from.
    """
    for token, handle, text in zip(tokens, legend.legend_handles, legend.get_texts()):
        handle.set_gid(legend_gid(dimension, token, "handle"))
        text.set_gid(legend_gid(dimension, token, "text"))


def cell_token(index: int) -> str:
    """Return the token naming the subplot at ``index`` in a figure's subplots."""
    return f"c{index}"


def axes_gid(cell: str) -> str:
    """Return the ``gid`` for a subplot's background patch.

    Tagging the patch gives the browser something whose on-screen rectangle *is* the plotting area,
    from which it derives screen positions for the hover readout.  Deriving them there rather than
    here keeps the readout correct however the SVG is later cropped or scaled -- and cropping is the
    default, since figures are saved with ``bbox_inches="tight"``.

    Args:
        cell: Token of the subplot, from :func:`cell_token`.

    Returns:
        The ``gid`` to set on ``ax.patch``.
    """
    return _SEP.join(["axes", cell])


class InteractiveFigure:
    """A matplotlib figure whose artists can be switched on and off from its legends.

    The figure is complete before this wrapper sees it; all the wrapper adds is the rendering.
    :attr:`figure` is the plain matplotlib :class:`~matplotlib.figure.Figure`, so anything
    matplotlib can do to a figure can still be done to this one.

    In a notebook, displaying the object renders it.  In a script, use :meth:`savefig`.  To place
    one in a page built by something else, use :meth:`to_html`.

    Args:
        figure: The figure to wrap.
        dimensions: The dimensions a trace's visibility is resolved along, one per legend, in the
            order their tokens appear in a trace's ``gid``.  An artist is shown only while none of
            its keys is switched off, which is what lets the legends compose.  Each figure names
            its own, so that what a legend switches can be what that figure is actually about -- a
            decay plot switches paths and layers, a topology gates and kinds of activity -- and so
            that the browser can learn the set from the figure instead of both sides having to
            agree on one.
        cells: Mapping from cell token (see :func:`cell_token`) to the subplot it names.  Each
            subplot's background patch is tagged so the browser can find it, and its axis limits are
            read at render time rather than now, so adjusting the figure afterwards cannot leave the
            hover readout pointing at stale coordinates.
        traces: Mapping from a trace ``gid`` (see :func:`trace_gid`) to the data behind it, for the
            hover readout.  Each value is ``{"label": str, "points": [[x, y], ...]}`` in data
            coordinates, optionally with ``"stds"`` and ``"texts"``: arrays parallel to ``"points"``
            giving a standard deviation to show alongside a point, and a readout string replacing
            the default one.  Traces absent from the mapping have no readout.  Coordinates are kept
            bare rather than pre-rendered per point, so the payload stays proportionate to the SVG
            even for a series of tens of thousands of points.
        hidden: Mapping from one of ``dimensions`` to the tokens the figure opens with switched off,
            so a figure too dense to read all at once can start on part of itself and still offer
            the rest.  Only the interactive rendering honours this; a static export shows
            everything, having no way to offer the rest back.
        container_id: Identifier for the ``<div>`` wrapping the SVG, unique per page.  Defaults to a
            fresh random one, which is what makes several figures on a page independent; pass a
            value only when reproducible output matters, as in tests.

    Raises:
        ValueError: If ``hidden`` names a dimension the figure does not have, or if a ``gid`` in
            ``traces`` does not name exactly one token per dimension.
    """

    @HAS_MATPLOTLIB.require_in_call
    def __init__(
        self,
        figure: "Figure",
        *,
        dimensions: Sequence[str] = (),
        cells: Mapping[str, "Axes"] | None = None,
        traces: Mapping[str, Mapping[str, Any]] | None = None,
        hidden: Mapping[str, Sequence[str]] | None = None,
        container_id: str | None = None,
    ):
        self._dimensions = list(dimensions)

        unknown = set(hidden or {}) - set(self._dimensions)
        if unknown:
            raise ValueError(
                f"Invalid dimension(s) in hidden: {sorted(unknown)}. "
                f"Must be among this figure's dimensions {tuple(self._dimensions)}."
            )

        # An ``id`` naming the wrong number of keys is the one mistake here that leaves a figure
        # looking finished: the browser cannot tell which token belongs to which dimension, so it
        # skips the artist, and the result is a picture whose legends do nothing.
        expected = len(self._dimensions) + _TRACE_GID_FIXED_FIELDS
        wrong = [gid for gid in (traces or {}) if len(gid.split(_SEP)) != expected]
        if wrong:
            raise ValueError(
                f"Trace gid(s) {sorted(wrong)} do not name one token per dimension. A figure with "
                f"dimensions {tuple(self._dimensions)} builds gids of {expected} fields."
            )

        self._figure = figure
        self._cells = dict(cells or {})
        self._traces = dict(traces or {})
        self._hidden = {
            dimension: list(tokens) for dimension, tokens in (hidden or {}).items() if tokens
        }
        self._container_id = container_id or f"qnl-figure-{uuid.uuid4().hex[:12]}"

        for cell, ax in self._cells.items():
            ax.patch.set_gid(axes_gid(cell))

    @property
    def figure(self) -> "Figure":
        """The wrapped matplotlib figure."""
        return self._figure

    @property
    def dimensions(self) -> tuple[str, ...]:
        """The dimensions this figure's artists can be hidden along, one per legend."""
        return tuple(self._dimensions)

    @property
    def container_id(self) -> str:
        """Identifier of the ``<div>`` wrapping the SVG in :meth:`to_html`."""
        return self._container_id

    def savefig(self, *args: Any, **kwargs: Any) -> None:
        """Save the figure, forwarding every argument to :meth:`matplotlib.figure.Figure.savefig`.

        The format follows from the filename as usual.  Vector formats keep the labels as text
        paths, so mathematics stays sharp at any zoom.
        """
        kwargs.setdefault("bbox_inches", "tight")
        self._figure.savefig(*args, **kwargs)

    def to_svg(self) -> str:
        """Return the figure as a standalone SVG document.

        Text is rendered as paths, so the SVG needs no fonts to be present wherever it is shown.
        """
        return self._render("svg").decode()

    def to_png(self, dpi: int = 150) -> bytes:
        """Return the figure as PNG bytes.

        Args:
            dpi: Resolution to rasterize at.

        Returns:
            The PNG data.
        """
        return self._render("png", dpi=dpi)

    def to_html(self, *, full_html: bool = False) -> str:
        """Return HTML embedding the figure, interactivity included.

        The result is self-contained: the SVG, the data behind it, and the JavaScript are all
        inline, with no requests to a CDN and no files to copy alongside.

        The figure starts at the size it was laid out for, shrinks to fit if that is wider than the
        space available, and can be resized from its bottom-right corner by the reader.

        Args:
            full_html: If ``True``, wrap the figure in a complete HTML document; otherwise return a
                fragment to drop into an existing page.

        Returns:
            The HTML.
        """
        svg, size = _sized_for_a_page(_strip_xml_prologue(self.to_svg()))
        payload = _json_for_script(self._sidecar())

        fragment = (
            f'<div class="qnl-figure" id="{self._container_id}"'
            f' style="{_container_style(size)}">\n'
            f"{svg}\n"
            f'<script type="application/json" class="qnl-figure-data">{payload}</script>\n'
            f"</div>\n"
            f"<script>{_javascript()}</script>\n"
            f'<script>window.qnlInteractive.init("{self._container_id}");</script>'
        )
        if not full_html:
            return fragment
        return (
            "<!DOCTYPE html>\n"
            '<html>\n<head><meta charset="utf-8"></head>\n<body>\n'
            f"{fragment}\n"
            "</body>\n</html>\n"
        )

    def _repr_mimebundle_(
        self, include: Sequence[str] | None = None, exclude: Sequence[str] | None = None
    ) -> dict[str, Any]:
        """Return the figure as both interactive HTML and a static PNG.

        Offering both is what makes the figure degrade rather than break: a front end that will not
        run the script takes the PNG and still gets the whole figure, just fixed.
        """
        bundle = {"text/html": self.to_html(), "image/png": self.to_png()}
        if include is not None:
            bundle = {key: value for key, value in bundle.items() if key in include}
        if exclude is not None:
            bundle = {key: value for key, value in bundle.items() if key not in exclude}
        return bundle

    def _render(self, fmt: str, **kwargs: Any) -> bytes:
        """Render the figure to ``fmt``, with the settings the interactive path depends on."""
        import matplotlib

        with matplotlib.rc_context(
            {
                # Glyphs as paths, so an SVG carries its own type and cannot fall back to whatever
                # the viewer happens to have installed.
                "svg.fonttype": "path",
                # A figure's clip paths and reused marker shapes are named from a hash, so two
                # figures on one page would arrive at the same names for different things, leaving
                # the second pointing at the first one's definitions.  Salting per figure keeps them
                # apart.  Glyph outlines are named for font and character instead, and sharing those
                # between figures is harmless.
                "svg.hashsalt": self._container_id,
            }
        ):
            buffer = io.BytesIO()
            self._figure.savefig(buffer, format=fmt, bbox_inches="tight", **kwargs)
        return buffer.getvalue()

    def _sidecar(self) -> dict[str, Any]:
        """Return the JSON-serializable data the browser needs beyond the SVG itself.

        Axis limits are read here rather than being stored up front, so they reflect the figure as
        it is actually being rendered.
        """
        cells = {}
        for cell, ax in self._cells.items():
            xlim, ylim = ax.get_xlim(), ax.get_ylim()
            cells[cell] = {
                "axes_gid": axes_gid(cell),
                "xlim": [float(xlim[0]), float(xlim[1])],
                "ylim": [float(ylim[0]), float(ylim[1])],
                "xlog": ax.get_xscale() == "log",
                "ylog": ax.get_yscale() == "log",
            }
        return {
            "dimensions": self._dimensions,
            "cells": cells,
            "traces": self._traces,
            "hidden": self._hidden,
        }


def _javascript() -> str:
    """Return the vendored interactivity script.

    Inlined into every figure rather than once per page: a figure cannot see the page it lands on,
    so one that assumed a sibling had supplied the script would break whenever it turned out to be
    alone.  The script guards against running twice, so repeating it costs only its size.
    """
    return _JAVASCRIPT_PATH.read_text(encoding="utf-8")


def _sized_for_a_page(svg: str) -> tuple[str, tuple[float, float] | None]:
    """Return ``svg`` freed from its fixed size, along with what that size was in CSS pixels.

    matplotlib writes the size onto the root element in points and a browser honours it literally,
    so a figure laid out for a wide screen goes on demanding that width in a narrow column, where it
    is clipped rather than fitted.  Replacing the attributes with percentages hands the decision to
    the container; the ``viewBox`` matplotlib also writes keeps the drawing undistorted at whatever
    size results.

    Returns:
        The rewritten SVG, and its natural size in CSS pixels, or ``None`` if the root element did
        not carry one to convert.
    """
    match = _SVG_ROOT_PATTERN.search(svg)
    if match is None:
        return svg, None

    root = match.group(0)
    sizes = {
        name: float(length) * 96.0 / _CSS_UNITS_PER_INCH[unit or "px"]
        for name, length, unit in _SVG_LENGTH_PATTERN.findall(root)
    }
    if "width" not in sizes or "height" not in sizes:
        return svg, None

    rewritten = (
        _SVG_LENGTH_PATTERN.sub("", root).removesuffix(">").rstrip()
        + ' style="width:100%;height:100%;display:block">'
    )
    return svg[: match.start()] + rewritten + svg[match.end() :], (sizes["width"], sizes["height"])


def _container_style(size: tuple[float, float] | None) -> str:
    """Return the ``style`` for the ``<div>`` holding a figure of natural size ``size``."""
    style = [
        # What the hover readout is positioned against.
        "position:relative",
        # However wide the figure was laid out to be, never wider than what it was dropped into.
        "max-width:100%",
        # ``resize`` only applies to a scroll container.  Nothing is ever actually scrolled to,
        # since the figure scales to this box rather than spilling out of it.
        "overflow:hidden",
        # Let the reader settle the size; only they can see how much room they have.
        "resize:both",
    ]
    if size is not None:
        width, height = size
        style += [
            f"width:{width:.0f}px",
            # Height follows width, so clamping the width scales the whole figure rather than
            # leaving it adrift in a box of its original height.  Dragging the corner sets both and
            # so takes over from this.
            f"aspect-ratio:{width:.0f}/{height:.0f}",
            f"min-width:{_MIN_FIGURE_PX[0]}px",
            f"min-height:{_MIN_FIGURE_PX[1]}px",
        ]
    return ";".join(style)


def _strip_xml_prologue(svg: str) -> str:
    """Return ``svg`` without its XML declaration and doctype.

    Both are valid in a standalone ``.svg`` file and invalid partway through an HTML document, so
    they have to come off before the SVG is inlined into a page.
    """
    start = svg.find("<svg")
    return svg[start:] if start != -1 else svg


def _json_for_script(data: Any) -> str:
    """Serialize ``data`` for embedding in a ``<script>`` element.

    An HTML parser ends a ``<script>`` at the first ``</script``, wherever it appears, so a ``<``
    anywhere in the payload could truncate the document.  Escaping every one as ``\\u003c`` is safe
    because JSON can only contain the character inside a string, where the escape is equivalent.
    """
    return json.dumps(data, separators=(",", ":")).replace("<", "\\u003c")
