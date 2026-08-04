"""Sphinx configuration for the qiskit-noise-learning documentation."""

import qiskit_noise_learning

# -- Project information -----------------------------------------------------

project = "Qiskit Noise Learning"
copyright = "2026, IBM"
author = "IBM"

release = qiskit_noise_learning.__version__
version = release

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    # myst_nb bundles myst_parser, so it also handles the plain Markdown pages.
    "myst_nb",
    "sphinxcontrib.bibtex",
    "sphinx_proof",
    "qiskit_sphinx_theme",
]

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

# -- autodoc / autosummary ---------------------------------------------------

# Only the class docstring is used for classes -- constructor parameters are
# documented in the class docstring, not in __init__.
autoclass_content = "class"
autodoc_member_order = "bysource"
autosummary_generate = True
# The per-subpackage API pages list names re-exported into each package __init__;
# this lets autosummary resolve and generate stubs for those imported names.
autosummary_imported_members = True

autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
}

# -- napoleon (Google-style docstrings) --------------------------------------

napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = False

# -- intersphinx -------------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "qiskit": ("https://quantum.cloud.ibm.com/docs/api/qiskit", None),
    "qiskit-ibm-runtime": ("https://quantum.cloud.ibm.com/docs/api/qiskit-ibm-runtime", None),
}

# -- MyST (Markdown) ---------------------------------------------------------

# amsmath: support LaTeX align/aligned environments; dollarmath: $...$ / $$...$$ math;
# colon_fence: ``:::{directive}`` fences, which unlike backtick fences can nest a code block
# without the enclosing fence having to grow an extra backtick.
myst_enable_extensions = ["amsmath", "colon_fence", "dollarmath"]

# By default MyST writes its own ``mathjax3_config`` and marks each page's top-level section
# ``tex2jax_ignore``, re-enabling math node by node.  Neither suits this site, which runs MathJax 2
# (see "Math" below) and would find a v3-shaped ``window.MathJax`` and fail to start.  Switching the
# mechanism off takes the ignore classes with it, which is what the note on ``mathjax2_config``
# below is about.
myst_update_mathjax = False

# -- myst-nb (executable tutorials) ------------------------------------------

# The tutorials in docs/tutorials are MyST Markdown notebooks with no stored outputs;
# every build runs them.  "cache" keeps local rebuilds fast by reusing outputs whenever
# the source is unchanged.
nb_execution_mode = "cache"
# Never publish a tutorial whose code raised.
nb_execution_raise_on_error = True
# The tutorials simulate whole learning experiments, which takes minutes, not seconds.
nb_execution_timeout = 900
# Render stderr inline rather than raising it as a Sphinx warning, which -W would turn into
# a build failure.  Anything a tutorial warns about should be visible to the reader instead.
nb_output_stderr = "show"


# -- sphinxcontrib-bibtex ----------------------------------------------------

bibtex_bibfiles = ["refs.bib"]

# -- Math --------------------------------------------------------------------
#
# All of the settings in this section, and the ``myst_update_mathjax`` above, follow from a single
# decision, so they are kept together: **the whole site is on MathJax 2 because two tutorial pages
# embed plotly figures whose curve labels are LaTeX.**
#
# Plotly typesets those labels by handing them to whatever MathJax the page loaded, and its
# converter checks the major version and silently gives up on anything but 2 or 3 -- so a figure on
# Sphinx's default MathJax 4 renders its legend as raw LaTeX source.  Of the two versions plotly
# accepts it has to be 2, even though it is end-of-life: under MathJax 3, Firefox mismeasures the
# ``ex``-sized nested ``<svg>`` that MathJax emits, and a legend comes out with its entries
# overlapping each other and its title, running off the right edge of the figure.  MathJax 2 is the
# path plotly actually tests, and it lays out correctly in both Firefox and Chromium.
#
# Sphinx has no per-page math configuration, so this choice is site-wide, and most of what it costs
# is paid by pages that have nothing to do with plotly -- above all ``formalism/index.md``, which is
# by far the largest body of math here.  Anything that changes about how prose math renders on this
# site starts with the paragraph above.  Reverting to MathJax 3 or 4 means translating the macros
# below back into the ``mathjax3_config`` schema and finding another way to render figure labels.
#
# The path must name an SVG bundle, because plotly switches the output jax to SVG while converting.
mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@2.7.9/MathJax.js?config=TeX-AMS-MML_SVG"

# Pin the entry point by content hash.  This only covers ``MathJax.js`` itself -- MathJax 2 goes on
# to pull its config bundle, output jax and fonts from the same CDN directory at runtime, and those
# cannot carry hashes -- so it is a partial guarantee.  The full alternative is vendoring MathJax 2
# into the repository, which its dynamic font loading makes a tens-of-megabytes commit for an
# end-of-life dependency; jsdelivr serves npm tarball contents verbatim and npm does not delete
# published versions, so the URL is stable enough not to be worth that.
mathjax_options = {
    "integrity": "sha384-vi9R4hb1goLJPJDHY+dOmXxcY3HGv6tJIwHxy5JunOTxJGHbsSuubPgl++SNxYYi",
    "crossorigin": "anonymous",
}

# MathJax 2 configuration, in ``MathJax.Hub.Config`` form.  Sphinx emits this as a
# ``text/x-mathjax-config`` block and switches the loader from ``defer`` to ``async``.
#
# There is deliberately no ``tex2jax`` entry.  MyST's ignore-class mechanism -- which would need a
# matching ``processClass`` here to re-enable math -- only runs when ``myst_update_mathjax`` is
# true, and it is false above, so no page carries an ignore class and MathJax 2's own defaults
# already process the whole page.  The flip side is that nothing shields the code cells: MathJax 2
# typesets ``$$...$$`` and ``\[...\]`` out of the box, so a tutorial that *prints* one of those
# would have it typeset as math.  If that comes up, add an ``ignoreClass`` naming myst-nb's
# ``cell_input`` and ``cell_output`` containers, and check in a browser that the plotly figures
# still typeset -- plotly hands its labels to MathJax directly rather than relying on the page scan,
# but it does so through an element whose position in the DOM is plotly's business, not ours.
mathjax2_config = {
    "TeX": {
        "Macros": {
            # No-argument macros.
            "Z": r"\mathbb{Z}",
            "E": r"\mathcal{E}",
            "P": r"\mathcal{P}",
            "U": r"\mathcal{U}",
            # Macros with arguments: [replacement, number-of-args].
            "ip": [r"\langle #1, #2 \rangle", 2],
            "bra": [r"\langle #1 |", 1],
            "ket": [r"| #1 \rangle", 1],
            "opbra": [r"\langle\!\langle #1 |", 1],
            "opket": [r"| #1 \rangle\!\rangle", 1],
        }
    },
}


def setup(app):
    """Register documentation-only Sphinx hooks."""
    # By default MathJax is only loaded on pages Sphinx found math on.  A plotly figure's LaTeX
    # lives inside a JSON blob, which is invisible to that check, so a tutorial whose only math is
    # inside a figure would load no MathJax at all and render the labels as raw source.  The cost is
    # that pages with no math load MathJax too.
    app.set_html_assets_policy("always")


# -- HTML output -------------------------------------------------------------

html_theme = "qiskit-ecosystem"
html_title = f"{project} {release}"
