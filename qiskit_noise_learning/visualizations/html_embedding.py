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

"""Embedding this package's plotly figures in a host HTML page.

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

:func:`host_page_renderer` returns a plotly renderer that satisfies (2) and (3) for a host page that
supplies its own MathJax, leaving only (1) to the page itself.
"""

from typing import TYPE_CHECKING

from ..optionals import HAS_PLOTLY

if TYPE_CHECKING:
    from plotly.io._base_renderers import MimetypeRenderer

# Runs after each figure's ``Plotly.newPlot`` resolves, with ``{plot_id}`` replaced by that figure's
# div id.  Scoped to the one figure it is attached to, so a page can mix these figures with others.
_FIGURE_SUPPORT_JS = r"""
(function () {
    var gd = document.getElementById('{plot_id}');
    if (!gd || !window.Plotly) {
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
            window.Plotly.redraw(gd).then(resize);
        }
    }

    // Plotly re-enters MathJax's startup while converting, so guard against running twice.
    var done = false;
    function start() {
        if (done) {
            return;
        }
        done = true;
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
        } else if (mathJax && mathJax.Hub) {
            // Runs immediately if startup already finished.
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


@HAS_PLOTLY.require_in_call
def host_page_renderer(*, connected: bool = True) -> "MimetypeRenderer":
    """Return a plotly renderer for embedding figures in a host HTML page.

    The renderer differs from plotly's stock ``"notebook"`` renderers in two ways, both of which
    matter for a page that is not a live notebook:

    * It does **not** emit a MathJax ``<script>`` tag.  The stock renderers hard-code
      ``include_mathjax="cdn"``, which on a page that already loads MathJax gives two copies
      fighting over ``window.MathJax``.  The host page is left to supply a MathJax 2 or 3 itself.
    * It attaches per-figure JavaScript that redraws the figure once MathJax is ready and keeps it
      sized to its container, in place of the stock notebook-cell teardown script (which watches for
      DOM containers that only exist in the classic notebook).

    Register it with plotly and make it the default before producing any figures::

        import plotly.io as pio

        from qiskit_noise_learning.visualizations import host_page_renderer

        pio.renderers["qiskit-noise-learning"] = host_page_renderer()
        pio.renderers.default = "qiskit-noise-learning"

    Args:
        connected: Whether to load plotly.js from its CDN rather than inlining a copy in every
            figure.  Inlining costs roughly three megabytes per figure, so it is only worth it for
            pages that must render offline.

    Returns:
        A plotly renderer emitting the ``text/html`` mime type.
    """
    from plotly.io._base_renderers import HtmlRenderer

    class _HostPageRenderer(HtmlRenderer):
        """Plotly HTML renderer for a page that supplies its own MathJax."""

        def to_mimebundle(self, fig_dict):
            from plotly.io import to_html

            post_script = [_FIGURE_SUPPORT_JS]
            if self.post_script:
                if isinstance(self.post_script, list | tuple):
                    post_script.extend(self.post_script)
                else:
                    post_script.append(self.post_script)

            return {
                "text/html": to_html(
                    fig_dict,
                    config=self.config,
                    auto_play=self.auto_play,
                    include_plotlyjs=self.include_plotlyjs,
                    include_mathjax=False,
                    post_script=post_script,
                    full_html=self.full_html,
                    animation_opts=self.animation_opts,
                    default_width="100%",
                    default_height=525,
                    validate=False,
                )
            }

    return _HostPageRenderer(connected=connected, config={"responsive": True})
