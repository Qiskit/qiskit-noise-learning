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

"""Embedding this package's plotly figures in a static HTML page.

The figures produced by this subpackage label their curves with math: a legend entry is a
``$...$``-delimited LaTeX string built by :func:`~.path_math_label` (labelled ``\\xrightarrow``
arrows, subscripted Paulis, bracketed fragment-depth exponents).  Plotly renders that math by
handing it to MathJax in the browser at page-view time, which makes any HTML embedding of these
figures — a Sphinx site, a Jupyter Book, an ``nbconvert`` report — subject to the same three
constraints:

1. The page must load MathJax **2 or 3**.  Plotly's converter checks the major version and silently
   bails out on anything else (leaving labels as raw LaTeX), so a page on MathJax 4 renders no math
   however it is configured.
2. The page must load **exactly one** MathJax.  MathJax 2 and 3 both take ownership of
   ``window.MathJax``, so whichever loads second breaks the first.  A figure must therefore not
   bring its own copy along.
3. A figure needs a redraw once MathJax is ready, and a re-measure when its container changes size.

:func:`html_page_renderer` returns a plotly renderer that satisfies (2) and (3) for a page that
supplies its own MathJax.  Constraint (1) stays the page's to keep, since only the page controls
which MathJax it loads; the renderer cannot enforce it, but the JavaScript it attaches warns on the
browser console when the MathJax it finds is one plotly will refuse to use.
"""

from collections.abc import Sequence
from functools import cache
from typing import TYPE_CHECKING, Any

from ..optionals import HAS_PLOTLY

if TYPE_CHECKING:
    from plotly.io.base_renderers import MimetypeRenderer

#: Suggested key to register :func:`html_page_renderer` under in ``plotly.io.renderers``.  Plotly
#: keys renderers by string and resolves ``pio.renderers.default`` through the same dictionary, so
#: registration and selection have to agree on a name; using this constant for both keeps them from
#: drifting apart.
RENDERER_NAME = "qiskit-noise-learning"

# Runs after each figure's ``Plotly.newPlot`` resolves, with ``{plot_id}`` replaced by that figure's
# div id.  Scoped to the one figure it is attached to, so a page can mix these figures with others.
_FIGURE_SUPPORT_JS = r"""
(function () {
    var gd = document.getElementById('{plot_id}');
    if (!gd || !window.Plotly) {
        // Unreachable as long as plotly.js is loaded by a blocking script tag ahead of the figure,
        // which is what ``include_plotlyjs`` emits either way.  Say something anyway, so that this
        // is not the one way the figure can come out wrong in silence.
        if (window.console && window.console.warn) {
            window.console.warn(
                'qiskit-noise-learning: no plotly.js or no figure element for {plot_id}; leaving ' +
                'this figure as drawn.'
            );
        }
        return;
    }

    // Re-measure the container.  Plotly's own ``responsive`` config only listens for *window*
    // resizes, so it misses a sidebar toggle or a theme settling its layout after first paint.
    function resize() {
        if (gd.data) {
            window.Plotly.Plots.resize(gd);
        }
    }

    // Re-run text layout, LaTeX conversion included.  Needed because plotly draws from an inline
    // script in the body, which on many pages runs before MathJax has finished starting up; that
    // first draw finds no MathJax and leaves every label as raw source.
    function redraw() {
        if (gd.data) {
            // Size the figure whichever way the redraw goes.  Passing ``resize`` as the rejection
            // handler too keeps the sizing guarantee independent of the math, and keeps a failed
            // redraw from surfacing only as an unhandled rejection.
            window.Plotly.redraw(gd).then(resize, resize);
        }
    }

    // Plotly typesets only against MathJax 2 or 3 and silently gives up on any other version, which
    // on the page looks identical to a figure that never had math in it.  Say so once, from the
    // figure that is affected, so the cause is visible to whoever is embedding it.
    function warnIfUnusable() {
        var version = window.MathJax && window.MathJax.version;
        var major = version ? parseInt(version.split('.')[0], 10) : null;
        if (major === 2 || major === 3) {
            return;
        }
        if (window.console && window.console.warn) {
            window.console.warn(
                'qiskit-noise-learning: this figure labels its curves with LaTeX, which plotly ' +
                'can only typeset against MathJax 2 or 3; this page loads ' +
                (version ? 'MathJax ' + version : 'no MathJax') +
                ', so the labels will stay as raw LaTeX source.'
            );
        }
    }

    // Plotly re-enters MathJax's startup while converting, so guard against running twice.
    var done = false;
    function start() {
        if (done) {
            return;
        }
        done = true;
        warnIfUnusable();
        redraw();
        if (window.ResizeObserver && gd.parentElement) {
            // Observe the parent, not the figure: resizing the figure changes its own rendered
            // size, which would feed straight back into the observer.
            var timer = null;
            var observer = new window.ResizeObserver(function () {
                window.clearTimeout(timer);
                timer = window.setTimeout(resize, 150);
            });
            observer.observe(gd.parentElement);
        }
    }

    // MathJax 2 and 3 signal readiness in entirely different ways, and ``window.MathJax`` stays
    // undefined until the loader runs, so neither can be hooked up front.
    function whenMathJaxReady(attemptsLeft) {
        var mathJax = window.MathJax;
        if (mathJax && mathJax.startup && mathJax.startup.promise) {
            mathJax.startup.promise.then(start);
        } else if (mathJax && mathJax.Hub && mathJax.Hub.Register) {
            // Runs immediately if startup already finished.  ``Hub.Register`` is checked as well as
            // ``Hub``: MathJax 2 is loaded asynchronously and publishes ``window.MathJax`` before
            // its API is fully built, and a throw in here would take the retry chain below down
            // with it, leaving the figure neither redrawn nor complained about.
            mathJax.Hub.Register.StartupHook('End', start);
        } else if (attemptsLeft > 0) {
            window.setTimeout(function () {
                whenMathJaxReady(attemptsLeft - 1);
            }, 100);
        } else {
            // No MathJax on the page at all: the labels stay as raw LaTeX, but at least size the
            // figure to its container.
            start();
        }
    }

    // A hundred attempts, 100 ms apart: ten seconds for a slow CDN, then give up on the math.
    whenMathJaxReady(100);
})();
"""


@cache
def _renderer_class() -> "type[MimetypeRenderer]":
    """Return the renderer class, building it on first use.

    The class cannot live at module scope, because plotly is an optional dependency and the base
    class is only importable once plotly is installed.  Caching means every renderer this module
    hands out shares one type, rather than each call minting a fresh, mutually unrecognized class.
    """
    from plotly.io.base_renderers import MimetypeRenderer

    class HtmlPageRenderer(MimetypeRenderer):
        """Plotly renderer emitting figure HTML for a page that supplies its own MathJax.

        Constructor arguments are as documented on :func:`html_page_renderer`, which is the only
        thing that instantiates this class.
        """

        def __init__(
            self,
            connected: bool = True,
            config: dict[str, Any] | None = None,
            post_script: str | Sequence[str] | None = None,
            height: int = 525,
        ):
            # Every parameter has to be stored as an attribute of the same name: plotly's
            # ``BaseRenderer.__repr__`` reads this signature and looks each parameter up in
            # ``__dict__``, and its ``__hash__`` is defined in terms of that repr.
            self.connected = connected
            self.config = {"responsive": True, **(config or {})}
            self.post_script = post_script
            self.height = height

        def __reduce__(self) -> tuple:
            # Pickle cannot look this class up by name, because it is built inside a function.
            # Round-trip through the module-level factory instead, so that instances copy and
            # pickle like plotly's own renderers, whose classes do live at module scope.
            return (
                _unpickle_renderer,
                (self.connected, self.config, self.post_script, self.height),
            )

        def to_mimebundle(self, fig_dict: dict[str, Any]) -> dict[str, str]:
            """Return the ``text/html`` rendering of one figure, script included."""
            from plotly.io import to_html

            post_script = [_FIGURE_SUPPORT_JS]
            if self.post_script:
                if isinstance(self.post_script, str):
                    post_script.append(self.post_script)
                else:
                    post_script.extend(self.post_script)

            html = to_html(
                fig_dict,
                # A copy: ``to_html`` writes into the config it is handed, and this renderer is
                # registered once and reused for every figure on the page.
                config=dict(self.config),
                # A static page has nothing to gain from starting an animation on load, and a
                # figure with no frames is unaffected either way.
                auto_play=False,
                include_plotlyjs="cdn" if self.connected else True,
                # The page owns MathJax; see constraint (2) in the module docstring.
                include_mathjax=False,
                post_script=post_script,
                # A figure goes inside the page's own document, not in place of it.
                full_html=False,
                # Track the container's width.  The height has to be fixed, since a figure has no
                # content height to derive one from.
                default_width="100%",
                default_height=self.height,
                # ``fig_dict`` comes from a Figure plotly has already validated.
                validate=False,
            )

            return {"text/html": html}

    return HtmlPageRenderer


def _unpickle_renderer(
    connected: bool,
    config: dict[str, Any] | None,
    post_script: str | Sequence[str] | None,
    height: int,
) -> "MimetypeRenderer":
    """Rebuild a renderer from the state captured by its ``__reduce__``."""
    return html_page_renderer(
        connected=connected, config=config, post_script=post_script, height=height
    )


@HAS_PLOTLY.require_in_call
def html_page_renderer(
    *,
    connected: bool = True,
    config: dict[str, Any] | None = None,
    post_script: str | Sequence[str] | None = None,
    height: int = 525,
) -> "MimetypeRenderer":
    """Return a plotly renderer for embedding figures in a static HTML page.

    The renderer differs from plotly's stock ``"notebook"`` renderers in two ways, both of which
    matter for a page that is not a live notebook:

    * It does **not** emit a MathJax ``<script>`` tag.  The stock renderers hard-code
      ``include_mathjax="cdn"``, which on a page that already loads MathJax gives two copies
      fighting over ``window.MathJax``.  The page is left to supply a MathJax 2 or 3 itself.
    * It attaches per-figure JavaScript that redraws the figure once MathJax is ready and keeps it
      sized to its container, in place of the stock notebook-cell teardown script (which watches for
      DOM containers that only exist in the classic notebook).

    Register it with plotly and make it the default before producing any figures::

        import plotly.io as pio

        from qiskit_noise_learning.visualizations import RENDERER_NAME, html_page_renderer

        pio.renderers[RENDERER_NAME] = html_page_renderer()
        pio.renderers.default = RENDERER_NAME

    Args:
        connected: Whether to load plotly.js from its CDN rather than inlining a copy in every
            figure.  Inlining costs roughly three megabytes per figure, so it is only worth it for
            pages that must render offline.
        config: Plotly.js configuration options applied to every figure, merged over this
            renderer's own default of ``{"responsive": True}``.
        post_script: Extra JavaScript to run after each figure is drawn, appended after this
            renderer's own script.  Occurrences of ``{plot_id}`` are replaced by the figure's
            div id.
        height: Pixel height of every figure.  Widths are always the full width of the containing
            element.

    Returns:
        A plotly renderer emitting the ``text/html`` mime type.
    """
    return _renderer_class()(
        connected=connected, config=config, post_script=post_script, height=height
    )
