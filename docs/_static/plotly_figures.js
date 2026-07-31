// Make the plotly figures in the tutorials lay themselves out correctly.
//
// Two things go wrong with a plotly figure embedded in a Sphinx page, and one redraw at the right
// moment fixes both.
//
// 1. MathJax is not there yet.  Plotly draws each figure from an inline <script> in the body, which
//    runs while the page is still being parsed, whereas Sphinx loads MathJax from the very end of
//    the body.  So on the first draw plotly finds no MathJax, logs "No MathJax version: undefined",
//    and leaves every ``$...$`` label as raw source.  (Plotly only recognizes MathJax 2 and 3 -- it
//    bails out on any other major version -- which is why ``mathjax_path`` in conf.py pins a
//    version, using the bundle that carries the SVG output jax plotly switches to while converting.)
//
// 2. The figure measures its container too early.  The figures set no explicit width, so plotly
//    autosizes them; it takes that measurement on the first draw, before the theme's layout has
//    settled, and comes up several hundred pixels too wide.  ``responsive: true`` only re-measures
//    on a *window* resize, which never happens, so the stale width sticks.  A figure wider than its
//    container is clipped on the right -- which is where these figures put their legend.
//
// So once MathJax has finished starting up, redraw each figure -- which re-runs its text layout,
// LaTeX conversion included -- and then resize it, which re-measures the container.  Both are
// needed: a resize is a no-op on a figure that was given an explicit width (as the gate-set topology
// plot is), so those would keep their raw labels.  A ResizeObserver keeps the figures in step with
// the container afterwards, so a window resize or a sidebar toggle cannot reintroduce the mismatch.
(function () {
  "use strict";

  function figures() {
    // A div that never got plotted has no data, and resizing it would throw.
    return Array.prototype.filter.call(
      document.querySelectorAll(".plotly-graph-div"),
      function (div) {
        return div.data;
      }
    );
  }

  function resizeAll() {
    figures().forEach(function (div) {
      window.Plotly.Plots.resize(div);
    });
  }

  function redrawAll() {
    figures().forEach(function (div) {
      window.Plotly.redraw(div);
    });
  }

  // Plotly's resize is itself debounced, but the observer can fire in bursts mid-layout.
  function debounce(fn, wait) {
    var timer = null;
    return function () {
      window.clearTimeout(timer);
      timer = window.setTimeout(fn, wait);
    };
  }

  function observeContainers() {
    if (!window.ResizeObserver) {
      return;
    }
    var observer = new window.ResizeObserver(debounce(resizeAll, 150));
    figures().forEach(function (div) {
      // Observe the parent, not the figure: the figure's own height is fixed by the plotting code,
      // but resizing it changes its rendered size, which would feed back into the observer.
      if (div.parentElement) {
        observer.observe(div.parentElement);
      }
    });
  }

  var done = false;

  function start() {
    // Plotly re-enters MathJax's startup while converting, which would otherwise land us here again.
    if (done || !window.Plotly) {
      return;
    }
    done = true;
    redrawAll();
    resizeAll();
    observeContainers();
  }

  // MathJax 2 and MathJax 3 signal readiness in entirely different ways, and ``window.MathJax``
  // stays undefined until the loader runs, so neither can be hooked up front.
  function whenMathJaxReady(attemptsLeft) {
    var mathJax = window.MathJax;
    if (mathJax && mathJax.startup && mathJax.startup.promise) {
      mathJax.startup.promise.then(start);
    } else if (mathJax && mathJax.Hub) {
      // Runs immediately if startup already finished.
      mathJax.Hub.Register.StartupHook("End", start);
    } else if (attemptsLeft > 0) {
      window.setTimeout(function () {
        whenMathJaxReady(attemptsLeft - 1);
      }, 100);
    } else {
      // No MathJax at all: the labels will stay as raw LaTeX, but at least size the figures right.
      start();
    }
  }

  // A hundred attempts, 100 ms apart: ten seconds for a slow CDN, then give up on the math.
  whenMathJaxReady(100);
})();
