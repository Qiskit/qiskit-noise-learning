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

"""Tests for the interactive SVG layer.

The interactivity itself lives in JavaScript and cannot be exercised here.  What *can* be checked --
and is, because it is where the coupling between the two sides sits -- is the contract the script
relies on: that every ``gid`` the script looks for is present in the SVG, is unique, and resolves
against a legend entry and a subplot that also exist.  A figure that violates any of those still
renders as a picture, which is exactly why the failure needs a test rather than an eye.
"""

import json
import re
import xml.etree.ElementTree as ET

import matplotlib
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402 - must follow the backend selection above

from qiskit_noise_learning.visualizations.interactive_svg import (  # noqa: E402
    DIMENSIONS,
    InteractiveFigure,
    TokenMap,
    axes_gid,
    cell_token,
    legend_gid,
    trace_gid,
)


def _svg_ids(svg):
    """Every ``id`` attribute in an SVG document, as a list so duplicates are visible."""
    return [
        element.get("id") for element in ET.fromstring(svg).iter() if element.get("id") is not None
    ]


def _definitions(svg):
    """Serialized form of every element the SVG points at by id.

    Only these matter across figures: an id nothing refers to is a label, but a *referenced* id
    shared by two figures on one page resolves to whichever came first in the document.
    """
    referenced = set(re.findall(r"url\(#([^)]+)\)", svg)) | set(re.findall(r'href="#([^"]+)"', svg))
    root = ET.fromstring(svg)
    # Indentation lands in each element's ``tail``, which would otherwise make two identical
    # definitions compare unequal purely because of where they sit in the document.
    for element in root.iter():
        element.tail = None
    return {
        element.get("id"): ET.tostring(element)
        for element in root.iter()
        if element.get("id") in referenced
    }


@pytest.fixture()
def figure():
    """A two-cell figure with two paths in two layers, plus both legends.

    Deliberately the smallest thing that can go wrong in the ways that matter: the same
    ``(path, layer)`` pair recurs in both cells, which is what makes ``id`` uniqueness a real
    question rather than a formality.
    """
    fig, axes = plt.subplots(1, 2, figsize=(7, 3))
    paths, layers = TokenMap("p"), TokenMap("l")
    traces = {}

    for cell_index, ax in enumerate(axes):
        cell = cell_token(cell_index)
        for path_key, offset in (("path-a", 0.0), ("path-b", 0.2)):
            for layer_index, (layer_key, style) in enumerate(
                (("points", "o"), ("fit", "-")),
            ):
                xs = [1.0, 2.0, 3.0]
                ys = [0.9 - offset - 0.1 * i for i in range(3)]
                (line,) = ax.plot(xs, ys, style)
                gid = trace_gid(cell, paths.token(path_key), layers.token(layer_key), layer_index)
                line.set_gid(gid)
                traces[gid] = {
                    "label": f"{path_key} / {layer_key}",
                    "points": [[x, y] for x, y in zip(xs, ys)],
                }
                # Only measured points have a spread; a fitted curve is exact where it is evaluated.
                if layer_key == "points":
                    traces[gid]["stds"] = [0.01 * (i + 1) for i in range(len(xs))]

    # ``fig.legend`` accumulates into ``fig.legends``, so calling it twice gives two legends without
    # any extra bookkeeping, and it draws its own copy of each handle -- hence tagging
    # ``legend_handles`` rather than the proxies passed in, which are never drawn.
    for dimension, tokens, loc in (("path", paths, "center left"), ("layer", layers, "lower left")):
        keyed = list(tokens.items())
        legend = fig.legend(
            [plt.Line2D([], [], label=f"{dimension}:{key}") for key, _ in keyed],
            [f"{dimension}:{key}" for key, _ in keyed],
            loc=loc,
        )
        for (_, token), handle, text in zip(keyed, legend.legend_handles, legend.get_texts()):
            handle.set_gid(legend_gid(dimension, token, "handle"))
            text.set_gid(legend_gid(dimension, token, "text"))

    yield InteractiveFigure(
        fig,
        cells={cell_token(i): ax for i, ax in enumerate(axes)},
        traces=traces,
        container_id="qnl-figure-test",
    )
    plt.close(fig)


class TestTokenMap:
    def test_one_token_per_distinct_key(self):
        """Equal keys must share a token and distinct keys must not, or artists that should toggle
        together come apart."""
        tokens = TokenMap("p")
        assert tokens.token(("fit", 3)) == tokens.token(("fit", 3))
        assert tokens.token(("fit", 3)) != tokens.token(("fit", 4))
        assert len(tokens) == 2

    def test_tokens_are_xml_safe(self):
        """Keys unusable as XML ids must still yield usable tokens: ids is what they become."""
        tokens = TokenMap("p")
        for key in (r"$X_{1} \to Z_{1}$", "with space", "1leading-digit", "sep|arator"):
            assert re.fullmatch(r"p\d+", tokens.token(key))


class TestGids:
    def test_legend_gid_rejects_unknown_dimension(self):
        with pytest.raises(ValueError):
            legend_gid("colour", "p0", "handle")

    @pytest.mark.parametrize("dimension", DIMENSIONS)
    def test_trace_gid_carries_every_dimension(self, dimension):
        """The script resolves a trace against both legends, so both keys must be recoverable."""
        gid = trace_gid(cell_token(0), "p3", "l1", 2)
        entry = legend_gid(dimension, {"path": "p3", "layer": "l1"}[dimension], "handle")
        assert entry.split("|")[2] in gid.split("|")

    def test_trace_gids_differ_by_index(self):
        assert trace_gid("c0", "p0", "l0", 0) != trace_gid("c0", "p0", "l0", 1)

    def test_trace_gids_differ_by_cell(self):
        """The same path and layer appear in every subplot; only the cell keeps their ids apart."""
        assert trace_gid("c0", "p0", "l0") != trace_gid("c1", "p0", "l0")


class TestSvgContract:
    def test_ids_are_unique(self, figure):
        """Duplicate ids are invalid markup, and make the script act on the wrong element."""
        ids = _svg_ids(figure.to_svg())
        duplicates = {value for value in ids if ids.count(value) > 1}
        assert not duplicates

    def test_every_trace_resolves_to_both_legends(self, figure):
        """A trace whose key has no legend entry can never be switched back on once hidden."""
        ids = _svg_ids(figure.to_svg())
        entry_tokens = {
            parts[1]: set() for parts in (i.split("|") for i in ids if i.startswith("key|"))
        }
        for value in ids:
            parts = value.split("|")
            if parts[0] == "key":
                entry_tokens[parts[1]].add(parts[2])

        traces = [value.split("|") for value in ids if value.startswith("trace|")]
        assert traces, "figure produced no tagged data artists"
        for _, _, path_token, layer_token, _ in traces:
            assert path_token in entry_tokens["path"]
            assert layer_token in entry_tokens["layer"]

    def test_no_orphan_legend_entries(self, figure):
        """A legend entry switching nothing is a control the reader can click to no effect."""
        ids = _svg_ids(figure.to_svg())
        traces = [value.split("|") for value in ids if value.startswith("trace|")]
        used = {
            "path": {parts[2] for parts in traces},
            "layer": {parts[3] for parts in traces},
        }
        for value in ids:
            parts = value.split("|")
            if parts[0] == "key":
                assert parts[2] in used[parts[1]]

    def test_both_handle_and_text_are_tagged(self, figure):
        """Clicking the label has to work, not just clicking the little line beside it."""
        ids = {value for value in _svg_ids(figure.to_svg()) if value.startswith("key|")}
        for dimension in DIMENSIONS:
            tokens = {v.split("|")[2] for v in ids if v.split("|")[1] == dimension}
            assert tokens
            for token in tokens:
                assert legend_gid(dimension, token, "handle") in ids
                assert legend_gid(dimension, token, "text") in ids

    def test_every_cell_patch_is_tagged(self, figure):
        """The hover readout locates a subplot by its patch; an untagged one gets no readout."""
        ids = set(_svg_ids(figure.to_svg()))
        for cell in figure._sidecar()["cells"]:  # noqa: SLF001 - the mapping has no public accessor
            assert axes_gid(cell) in ids

    def test_carries_no_font_references(self, figure):
        """Rendering text as paths is what lets the mathematics survive without the fonts."""
        assert "font-family" not in figure.to_svg()

    def test_alike_figures_do_not_collide_on_definition_ids(self, figure):
        """Two figures drawn alike must not name their definitions identically.

        matplotlib names the things a figure refers to by id -- its clip paths, its reused marker
        shapes -- from a hash of their content, so two figures that contain the same shapes arrive
        at the same names and a page holding both carries duplicate ids.  Nothing *renders* wrong
        when that happens, precisely because equal names there imply equal content, but the markup
        is invalid; salting the hash per figure keeps them apart.  Two renderings of one figure is
        the worst case, and the case a grid of similar subplots most resembles.

        Glyph outlines are excluded deliberately: matplotlib names those for a font and a character
        rather than by hash, and the same name is always the same outline.
        """
        twin = InteractiveFigure(figure.figure, container_id="qnl-figure-twin")
        hashed = re.compile(r"^[pm][0-9a-f]{8,}$")
        mine = {gid for gid in _definitions(figure.to_svg()) if hashed.match(gid)}
        theirs = {gid for gid in _definitions(twin.to_svg()) if hashed.match(gid)}
        assert (
            mine and theirs
        ), "no hash-named definitions; has matplotlib changed how it names them?"
        assert not mine & theirs


class TestVendoredScript:
    """Checks on the JavaScript asset that do not require a browser to run it.

    What the script *does* can only be confirmed by loading a figure in one, which the migration's
    manual verification step covers.  What can be confirmed here is that it is present, that it will
    be installed alongside the package rather than only existing in a checkout, and that the one
    constant it shares with the Python side still agrees -- a disagreement there would leave a
    legend silently inert, which is the failure least likely to be noticed by eye.
    """

    def test_is_installed_with_the_package(self):
        from qiskit_noise_learning.visualizations import interactive_svg

        assert interactive_svg._JAVASCRIPT_PATH.is_file()  # noqa: SLF001 - checking the asset path

    def test_dimensions_agree_across_the_language_boundary(self):
        from qiskit_noise_learning.visualizations import interactive_svg

        source = interactive_svg._JAVASCRIPT_PATH.read_text()  # noqa: SLF001 - reading the asset
        declared = re.search(r"var DIMENSIONS = \[(.*?)\]", source)
        assert declared
        assert tuple(re.findall(r'"([^"]+)"', declared.group(1))) == DIMENSIONS

    def test_reads_the_initially_hidden_keys(self):
        """The other field the two sides have to agree on by name, with the same failure mode."""
        from qiskit_noise_learning.visualizations import interactive_svg

        source = interactive_svg._JAVASCRIPT_PATH.read_text()  # noqa: SLF001 - reading the asset
        assert "this.data.hidden" in source

    def test_the_id_separator_agrees_across_the_language_boundary(self):
        """Every id is written with it here and taken apart on it there; a disagreement inerts both
        legends at once."""
        from qiskit_noise_learning.visualizations import interactive_svg

        source = interactive_svg._JAVASCRIPT_PATH.read_text()  # noqa: SLF001 - reading the asset
        declared = re.search(r'var SEP = "(.*?)";', source)
        assert declared
        assert declared.group(1) == interactive_svg._SEP  # noqa: SLF001 - the shared constant


class TestSidecar:
    def test_limits_track_the_axes(self, figure):
        """Read at render time, so a figure adjusted after construction still reads out right."""
        list(figure._cells.values())[0].set_xlim(0.0, 42.0)  # noqa: SLF001 - no public accessor
        assert figure._sidecar()["cells"]["c0"]["xlim"] == [0.0, 42.0]  # noqa: SLF001

    def test_records_axis_scales(self, figure):
        ax = figure._cells["c0"]  # noqa: SLF001 - no public accessor
        ax.set_yscale("log")
        cell = figure._sidecar()["cells"]["c0"]  # noqa: SLF001
        assert cell["ylog"] is True
        assert cell["xlog"] is False

    def test_carries_the_keys_the_figure_opens_switched_off(self, figure):
        """A figure may open on part of itself; the tokens for the rest travel in the sidecar."""
        partial = InteractiveFigure(figure.figure, hidden={"layer": ["l1"]})
        assert partial._sidecar()["hidden"] == {"layer": ["l1"]}  # noqa: SLF001 - no accessor

    def test_rejects_hiding_along_an_unknown_dimension(self, figure):
        """Silently ignoring the name would leave a figure that hides nothing and says nothing."""
        with pytest.raises(ValueError, match="hidden"):
            InteractiveFigure(figure.figure, hidden={"colour": ["x0"]})

    def test_trace_data_is_keyed_by_gid(self, figure):
        """The script joins the SVG to the data by id, so the keys have to be the same ids."""
        sidecar = figure._sidecar()  # noqa: SLF001 - no public accessor
        ids = set(_svg_ids(figure.to_svg()))
        assert sidecar["traces"]
        for gid in sidecar["traces"]:
            assert gid in ids


class TestHtml:
    def test_embeds_svg_without_an_xml_prologue(self, figure):
        """A doctype partway through a page is invalid, and derails the parser after it."""
        html = figure.to_html()
        assert "<svg" in html
        assert "<?xml" not in html
        assert "<!DOCTYPE svg" not in html

    def test_sidecar_json_is_parseable(self, figure):
        payload = re.search(
            r'<script type="application/json" class="qnl-figure-data">(.*?)</script>',
            figure.to_html(),
            re.DOTALL,
        )
        assert payload
        assert json.loads(payload.group(1))["cells"]

    def test_sidecar_cannot_close_the_script_early(self, figure):
        """A ``<`` in a label would otherwise be able to truncate the document."""
        gid = next(iter(figure._sidecar()["traces"]))  # noqa: SLF001 - no public accessor
        figure._traces[gid] = {"label": "</script><img src=x>", "points": []}  # noqa: SLF001
        html = figure.to_html()
        assert "</script><img" not in html
        # Still valid JSON, and still says what it said.
        payload = re.search(
            r'<script type="application/json" class="qnl-figure-data">(.*?)</script>',
            html,
            re.DOTALL,
        )
        assert json.loads(payload.group(1))["traces"][gid]["label"] == "</script><img src=x>"

    def test_initializes_its_own_container(self, figure):
        html = figure.to_html()
        assert f'id="{figure.container_id}"' in html
        assert f'window.qnlInteractive.init("{figure.container_id}")' in html

    def test_script_is_inlined_not_fetched(self, figure):
        """No CDN, and no file to copy alongside the page."""
        html = figure.to_html()
        assert "window.qnlInteractive" in html
        assert "<script src=" not in html

    def test_script_tolerates_being_included_twice(self, figure):
        """Two figures on a page each bring the script; the second copy must stand down."""
        html = figure.to_html()
        script = re.search(r"<script>(.*?window\.qnlInteractive = .*?)</script>", html, re.DOTALL)
        assert script
        assert "if (window.qnlInteractive)" in script.group(1)

    def test_full_html_is_a_document(self, figure):
        html = figure.to_html(full_html=True)
        assert html.startswith("<!DOCTYPE html>")
        assert html.rstrip().endswith("</html>")

    def test_container_ids_are_unique_by_default(self, figure):
        first = InteractiveFigure(figure.figure)
        second = InteractiveFigure(figure.figure)
        assert first.container_id != second.container_id


class TestSizing:
    """How large the figure asks to be.

    Whether it *looks* right in a given page is for a browser to say, so only the unit conversion is
    checked here: matplotlib declares a size in points and a browser lays out in pixels, at 96 to
    the inch against 72, so a figure that skips the conversion silently asks for a third more room
    than it needs.
    """

    @staticmethod
    def _root(svg_or_html):
        """The opening ``<svg>`` tag, which is where a size would be declared."""
        root = re.search(r"<svg\b[^>]*>", svg_or_html)
        assert root
        return root.group(0)

    def test_starts_at_its_natural_size_converted_to_pixels(self, figure):
        points = {
            name: float(length)
            for name, length in re.findall(
                r'\s(width|height)="([0-9.]+)pt"', self._root(figure.to_svg())
            )
        }
        style = re.search(r'<div class="qnl-figure" id="[^"]+" style="([^"]*)"', figure.to_html())
        assert style
        width = re.search(r"width:([0-9]+)px", style.group(1))
        assert width
        assert abs(int(width.group(1)) - points["width"] * 96 / 72) <= 1

    def test_standalone_svg_keeps_the_size_it_was_drawn_at(self, figure):
        """Only the embedded copy answers to a container; a ``.svg`` file has no container."""
        root = self._root(figure.to_svg())
        assert re.search(r'\swidth="[0-9.]+pt"', root)
        assert re.search(r'\sheight="[0-9.]+pt"', root)


class TestMimebundle:
    def test_offers_html_and_a_static_fallback(self, figure):
        bundle = figure._repr_mimebundle_()  # noqa: SLF001 - the IPython display protocol
        assert set(bundle) == {"text/html", "image/png"}
        assert bundle["image/png"].startswith(b"\x89PNG")

    def test_honours_include_and_exclude(self, figure):
        assert set(figure._repr_mimebundle_(include=["image/png"])) == {"image/png"}  # noqa: SLF001
        assert set(figure._repr_mimebundle_(exclude=["image/png"])) == {"text/html"}  # noqa: SLF001


class TestSavefig:
    @pytest.mark.parametrize("suffix", ["png", "pdf", "svg"])
    def test_writes_a_non_empty_file(self, figure, tmp_path, suffix):
        target = tmp_path / f"figure.{suffix}"
        figure.savefig(target)
        assert target.stat().st_size > 0

    def test_pdf_keeps_the_mathematics_as_vectors(self, figure, tmp_path):
        """A rasterized export would lose the sharpness the labels are typeset for.

        Checks for an actual image object; every matplotlib PDF *declares* image support in its
        ``/ProcSet`` whether or not it uses any.
        """
        target = tmp_path / "figure.pdf"
        figure.savefig(target)
        assert b"/Subtype /Image" not in target.read_bytes()
