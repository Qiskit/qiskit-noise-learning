/*
 * This code is a Qiskit project.
 *
 * (C) Copyright IBM 2026.
 *
 * This code is licensed under the Apache License, Version 2.0. You may
 * obtain a copy of this license in the LICENSE.txt file in the root directory
 * of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
 *
 * Any modifications or derivative works of this code must retain this
 * copyright notice, and modified files need to carry a notice indicating
 * that they have been altered from the originals.
 */

/*
 * Interactivity for the SVG figures produced by ``qiskit_noise_learning.visualizations``.
 *
 * The figure arrives as finished SVG; nothing here draws anything.  This file only does the two
 * things a static image cannot:
 *
 *   1. Toggling.  Every data artist carries an ``id`` naming its key in each dimension, and every
 *      legend entry an ``id`` naming one key.  Dimensions are tracked independently and an artist is
 *      shown only while none of its keys is hidden, so the legends compose.
 *   2. Hover readout.  The data behind each artist travels alongside the SVG as JSON, in data
 *      coordinates, and is placed against the live on-screen rectangle of the axes -- which the
 *      figure marks for the purpose -- so the readout survives any cropping or scaling of the SVG.
 *
 * Loading this file twice on one page is harmless: the guard below keeps the first copy, and every
 * figure then registers itself against that copy.
 */

(function () {
  "use strict";

  if (window.qnlInteractive) {
    return;
  }

  // Opacity of a legend entry whose key is currently hidden.  Faded rather than removed, since it
  // stays the control you click to bring the data back.
  var HIDDEN_ENTRY_OPACITY = "0.3";

  // How near the pointer has to be, in CSS pixels, for a point to claim the readout.
  var HIT_RADIUS = 14;

  // How far the readout sits from the pointer, in CSS pixels.
  var READOUT_OFFSET = 12;

  // How far outside a legend entry's ink its click target extends, horizontally and vertically, in
  // SVG user units.  The vertical figure stays small on purpose: legend rows sit close together, and
  // a box tall enough to reach its neighbour's row would start answering for the wrong key.
  var HIT_PADDING = [4, 2];

  var SVG_NS = "http://www.w3.org/2000/svg";

  // Separator between the fields of an ``id``.  Must agree with ``_SEP`` in ``interactive_svg.py``,
  // which builds the ids this file takes apart.
  var SEP = "|";

  // The dimensions a trace can be hidden along, one per legend.  Visibility is the conjunction over
  // all of them.  Must agree with ``DIMENSIONS`` in ``interactive_svg.py``.
  var DIMENSIONS = ["path", "layer"];

  /* Segments of an artist's ``id``, or ``[]`` for an element without one.
   *
   * The ids are built in ``interactive_svg.py``; the two shapes read here are
   * ``trace|<cell>|<path>|<layer>|<n>`` and ``key|<dimension>|<token>|<part>``.
   */
  function segments(element) {
    var id = element.getAttribute("id");
    return id ? id.split(SEP) : [];
  }

  function QnlFigure(root, data) {
    this.root = root;
    this.data = data || {};
    this.traces = [];
    this.entries = [];
    // dimension -> Set of tokens currently switched off.  Seeded from the figure, which may open with
    // part of itself hidden; hidden artists are still in the SVG, one click away.
    this.hidden = {};
    // dimension -> every token that dimension's legend offers, for isolate-on-double-click.
    this.known = {};
    var initiallyHidden = this.data.hidden || {};
    for (var i = 0; i < DIMENSIONS.length; i++) {
      this.hidden[DIMENSIONS[i]] = new Set(initiallyHidden[DIMENSIONS[i]] || []);
      this.known[DIMENSIONS[i]] = new Set();
    }

    this._collect();
    this._addHitBoxes();
    this._wireLegends();
    this._wireReadout();
    this._apply();
  }

  /* Index the data artists and legend entries the SVG contains. */
  QnlFigure.prototype._collect = function () {
    var self = this;

    // Attribute selectors, not ``#id``: the ids contain ``|``, which is not valid in a CSS id
    // selector without escaping.
    this.root.querySelectorAll('[id^="trace|"]').forEach(function (element) {
      var parts = segments(element);
      if (parts.length < 4) {
        return;
      }
      self.traces.push({
        element: element,
        gid: parts.join(SEP),
        cell: parts[1],
        keys: { path: parts[2], layer: parts[3] },
      });
    });

    this.root.querySelectorAll('[id^="key|"]').forEach(function (element) {
      var parts = segments(element);
      if (parts.length < 4) {
        return;
      }
      var dimension = parts[1];
      if (!(dimension in self.hidden)) {
        return;
      }
      self.entries.push({ element: element, dimension: dimension, token: parts[2] });
      self.known[dimension].add(parts[2]);
    });
  };

  /* The bounding box of one element in its parent's coordinate system.
   *
   * ``getBBox`` reports a box in the element's *own* space, so an element carrying a transform has to
   * have it applied before boxes from two elements can be put side by side.  Returns ``null`` when
   * the browser cannot measure the element, which is what happens when the figure is not being
   * displayed -- a collapsed section, a background tab on first paint.
   */
  function boxInParentSpace(element) {
    if (!element.getBBox) {
      return null;
    }
    var box;
    try {
      box = element.getBBox();
    } catch (error) {
      return null;
    }
    var transform = element.transform && element.transform.baseVal.consolidate();
    if (!transform) {
      return { x: box.x, y: box.y, width: box.width, height: box.height };
    }
    var m = transform.matrix;
    var xs = [];
    var ys = [];
    [
      [box.x, box.y],
      [box.x + box.width, box.y],
      [box.x, box.y + box.height],
      [box.x + box.width, box.y + box.height],
    ].forEach(function (corner) {
      xs.push(m.a * corner[0] + m.c * corner[1] + m.e);
      ys.push(m.b * corner[0] + m.d * corner[1] + m.f);
    });
    var left = Math.min.apply(null, xs);
    var top = Math.min.apply(null, ys);
    return {
      x: left,
      y: top,
      width: Math.max.apply(null, xs) - left,
      height: Math.max.apply(null, ys) - top,
    };
  }

  /* Give each legend key one rectangular click target covering its handle, its label, and the gap
   * between the two.
   *
   * Legend ink is thin -- a label is glyph outlines, so the middle of an "o" is a hole that misses --
   * and without this a click has to land on it exactly.  The rectangle draws nothing and goes
   * underneath the ink, changing neither how the figure looks nor what the ink answers for.
   *
   * Skipped silently if the figure cannot be measured: clickable-on-the-ink-only is the behaviour
   * this improves on, not a broken one.
   */
  QnlFigure.prototype._addHitBoxes = function () {
    var byKey = {};
    this.entries.forEach(function (entry) {
      var name = entry.dimension + SEP + entry.token;
      (byKey[name] = byKey[name] || []).push(entry);
    });

    var boxes = [];
    Object.keys(byKey).forEach(function (name) {
      var entries = byKey[name];
      var parent = entries[0].element.parentNode;
      var box = null;
      for (var i = 0; i < entries.length; i++) {
        // A single rectangle can only stand for entries drawn in a single coordinate system, and
        // siblings are the only elements guaranteed to share one.
        if (entries[i].element.parentNode !== parent) {
          return;
        }
        var part = boxInParentSpace(entries[i].element);
        if (!part) {
          return;
        }
        box = box === null ? part : union(box, part);
      }
      // A handle on its own is a horizontal line, whose box is legitimately of zero height; only a
      // box with no extent at all means there was nothing to measure.
      if (!box || (!box.width && !box.height)) {
        return;
      }

      var rect = document.createElementNS(SVG_NS, "rect");
      rect.setAttribute("x", box.x - HIT_PADDING[0]);
      rect.setAttribute("y", box.y - HIT_PADDING[1]);
      rect.setAttribute("width", box.width + 2 * HIT_PADDING[0]);
      rect.setAttribute("height", box.height + 2 * HIT_PADDING[1]);
      // ``fill: none`` is what keeps it invisible; ``pointer-events: all`` is what makes it catch a
      // pointer anyway, which an unfilled shape otherwise would not.
      rect.setAttribute("style", "fill:none;pointer-events:all");
      rect.setAttribute("id", ["key", entries[0].dimension, entries[0].token, "box"].join(SEP));
      // First child, so the box sits below the ink of every entry rather than over the top of the
      // next row's.
      parent.insertBefore(rect, parent.firstChild);
      boxes.push({ element: rect, dimension: entries[0].dimension, token: entries[0].token });
    });

    this.entries = this.entries.concat(boxes);
  };

  /* The smallest box containing two boxes. */
  function union(first, second) {
    var x = Math.min(first.x, second.x);
    var y = Math.min(first.y, second.y);
    return {
      x: x,
      y: y,
      width: Math.max(first.x + first.width, second.x + second.width) - x,
      height: Math.max(first.y + first.height, second.y + second.height) - y,
    };
  }

  QnlFigure.prototype._wireLegends = function () {
    var self = this;
    this.entries.forEach(function (entry) {
      entry.element.style.cursor = "pointer";
      // Every part of an entry shares one token, so any of them can be clicked to drive its key.
      entry.element.addEventListener("click", function (event) {
        event.preventDefault();
        self.toggle(entry.dimension, entry.token);
      });
      entry.element.addEventListener("dblclick", function (event) {
        event.preventDefault();
        self.isolate(entry.dimension, entry.token);
      });
    });
  };

  /* Show or hide one key. */
  QnlFigure.prototype.toggle = function (dimension, token) {
    var hidden = this.hidden[dimension];
    if (hidden.has(token)) {
      hidden.delete(token);
    } else {
      hidden.add(token);
    }
    this._apply();
  };

  /* Hide every *other* key in this dimension, or restore them all if it is already alone.
   *
   * A double click also delivers two clicks, so ``toggle`` has already run twice by the time this
   * does; because that leaves the key as it started, isolating from here is unambiguous.
   */
  QnlFigure.prototype.isolate = function (dimension, token) {
    var known = this.known[dimension];
    var hidden = this.hidden[dimension];
    var alreadyAlone = hidden.size === known.size - 1 && !hidden.has(token);
    hidden.clear();
    if (!alreadyAlone) {
      known.forEach(function (other) {
        if (other !== token) {
          hidden.add(other);
        }
      });
    }
    this._apply();
  };

  QnlFigure.prototype._isVisible = function (trace) {
    for (var i = 0; i < DIMENSIONS.length; i++) {
      var dimension = DIMENSIONS[i];
      if (this.hidden[dimension].has(trace.keys[dimension])) {
        return false;
      }
    }
    return true;
  };

  /* Push the current key state onto the SVG. */
  QnlFigure.prototype._apply = function () {
    var self = this;
    this.traces.forEach(function (trace) {
      trace.visible = self._isVisible(trace);
      trace.element.style.display = trace.visible ? "" : "none";
    });
    this.entries.forEach(function (entry) {
      var off = self.hidden[entry.dimension].has(entry.token);
      entry.element.style.opacity = off ? HIDDEN_ENTRY_OPACITY : "";
    });
    this._hideReadout();
  };

  /* A number at a length a reader can take in, without trailing floating-point noise. */
  function format(value) {
    return String(Number(value.toPrecision(6)));
  }

  /* A trace label as text rather than as its source.
   *
   * Labels are math-mode LaTeX, since that is what the legend typesets them from, and the readout is
   * a plain DOM node, where the delimiters would show up literally.
   */
  function plainLabel(label) {
    return label ? label.replace(/\$/g, "") : "";
  }

  /* The text for one hit: whatever the figure supplied for that point, else its label and place. */
  function readoutText(hit) {
    if (hit.trace.texts && hit.trace.texts[hit.index]) {
      return hit.trace.texts[hit.index];
    }
    var lines = [];
    var label = plainLabel(hit.trace.label);
    if (label) {
      lines.push(label);
    }
    var place = "(" + format(hit.point[0]) + ", " + format(hit.point[1]) + ")";
    var stds = hit.trace.stds;
    if (stds && typeof stds[hit.index] === "number") {
      place += " ± " + format(stds[hit.index]);
    }
    lines.push(place);
    return lines.join("\n");
  }

  /* Fraction of the way from ``lo`` to ``hi``, on a linear or logarithmic axis. */
  function fraction(value, lo, hi, logarithmic) {
    if (logarithmic) {
      if (!(value > 0) || !(lo > 0) || !(hi > 0)) {
        return NaN;
      }
      return (Math.log(value) - Math.log(lo)) / (Math.log(hi) - Math.log(lo));
    }
    return (value - lo) / (hi - lo);
  }

  /* The on-screen rectangle of one subplot, or ``null`` if it cannot be located.
   *
   * Read fresh on every pointer move rather than cached: the page may have resized, scrolled, or
   * restyled the figure since the last one, and a stale rectangle would silently misplace the
   * readout instead of failing visibly.
   */
  QnlFigure.prototype._cellRect = function (cell) {
    var spec = (this.data.cells || {})[cell];
    if (!spec) {
      return null;
    }
    var element = this.root.querySelector('[id="' + spec.axes_gid + '"]');
    if (!element) {
      return null;
    }
    var rect = element.getBoundingClientRect();
    if (!rect.width || !rect.height) {
      return null;
    }
    return { spec: spec, rect: rect };
  };

  /* The visible point nearest the pointer, if one is close enough to claim the readout. */
  QnlFigure.prototype._nearest = function (clientX, clientY) {
    var traceData = this.data.traces || {};
    var best = null;
    var rects = {};

    for (var i = 0; i < this.traces.length; i++) {
      var trace = this.traces[i];
      if (!trace.visible) {
        continue;
      }
      var spec = traceData[trace.gid];
      if (!spec || !spec.points) {
        continue;
      }
      if (!(trace.cell in rects)) {
        rects[trace.cell] = this._cellRect(trace.cell);
      }
      var cell = rects[trace.cell];
      if (!cell) {
        continue;
      }

      for (var j = 0; j < spec.points.length; j++) {
        var point = spec.points[j];
        var fx = fraction(point[0], cell.spec.xlim[0], cell.spec.xlim[1], cell.spec.xlog);
        var fy = fraction(point[1], cell.spec.ylim[0], cell.spec.ylim[1], cell.spec.ylog);
        if (isNaN(fx) || isNaN(fy)) {
          continue;
        }
        var dx = cell.rect.left + fx * cell.rect.width - clientX;
        // SVG y grows downwards while the axis grows upwards, hence measuring from the bottom.
        var dy = cell.rect.bottom - fy * cell.rect.height - clientY;
        var distance = Math.sqrt(dx * dx + dy * dy);
        if (distance <= HIT_RADIUS && (best === null || distance < best.distance)) {
          best = { distance: distance, trace: spec, point: point, index: j };
        }
      }
    }
    return best;
  };

  QnlFigure.prototype._wireReadout = function () {
    var self = this;

    this.readout = document.createElement("div");
    this.readout.className = "qnl-readout";
    // Inline styles rather than a stylesheet: a figure has to look the same in a Sphinx page, a
    // notebook, and a bare HTML file, none of which share a theme.
    this.readout.setAttribute(
      "style",
      [
        "position:absolute",
        "display:none",
        "z-index:10",
        "pointer-events:none",
        "padding:4px 7px",
        "border:1px solid rgba(0,0,0,0.25)",
        "border-radius:3px",
        "background:rgba(255,255,255,0.95)",
        "color:#222",
        "font:12px/1.35 -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif",
        "white-space:pre",
        "box-shadow:0 1px 4px rgba(0,0,0,0.2)",
      ].join(";")
    );
    this.root.appendChild(this.readout);

    this.root.addEventListener("mousemove", function (event) {
      var hit = self._nearest(event.clientX, event.clientY);
      if (!hit) {
        self._hideReadout();
        return;
      }
      var rootRect = self.root.getBoundingClientRect();
      self.readout.textContent = readoutText(hit);
      self.readout.style.display = "block";
      // Measured only once it has its text and is displayed, then placed on whichever side of the
      // pointer keeps it inside the figure, whose container clips what leaves it.
      var size = self.readout.getBoundingClientRect();
      var left = event.clientX - rootRect.left + READOUT_OFFSET;
      var top = event.clientY - rootRect.top + READOUT_OFFSET;
      if (left + size.width > rootRect.width) {
        left = Math.max(0, left - size.width - 2 * READOUT_OFFSET);
      }
      if (top + size.height > rootRect.height) {
        top = Math.max(0, top - size.height - 2 * READOUT_OFFSET);
      }
      self.readout.style.left = left + "px";
      self.readout.style.top = top + "px";
    });
    this.root.addEventListener("mouseleave", function () {
      self._hideReadout();
    });
  };

  QnlFigure.prototype._hideReadout = function () {
    if (this.readout) {
      this.readout.style.display = "none";
    }
  };

  /* Attach behaviour to the figure in the container with the given id.
   *
   * Safe to call before the element exists -- the call is deferred to ``DOMContentLoaded`` -- and
   * safe to call twice for the same container, so a duplicated script tag is a no-op rather than a
   * figure with two sets of handlers.
   */
  function init(containerId) {
    var root = document.getElementById(containerId);
    if (!root) {
      if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", function () {
          init(containerId);
        });
      }
      return;
    }
    if (root.__qnlFigure) {
      return;
    }
    var payload = root.querySelector("script.qnl-figure-data");
    var data = {};
    if (payload) {
      try {
        data = JSON.parse(payload.textContent);
      } catch (error) {
        // A malformed sidecar costs the hover readout, not the figure, so keep going.
        data = {};
      }
    }
    root.__qnlFigure = new QnlFigure(root, data);
  }

  window.qnlInteractive = { init: init, Figure: QnlFigure };
})();
