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

# By default MyST writes its own ``mathjax3_config``.  This project runs MathJax 2 (see the MathJax
# section below), which would then find a v3-shaped ``window.MathJax`` and fail to start; MyST's
# ignore-class handling is folded into ``mathjax2_config`` instead.
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


# -- Plotly ------------------------------------------------------------------

# The tutorials' figures are emitted through the package's own plotly renderer
# (``qiskit_noise_learning.visualizations.host_page_renderer``), which knows not to bring a second
# MathJax along and carries the per-figure redraw/resize script with it.  Nothing is needed here.


def setup(app):
    """Register documentation-only Sphinx hooks."""
    # By default MathJax is only loaded on pages Sphinx found math on.  A plotly figure's LaTeX
    # is invisible to that check, so a tutorial whose only math is inside a figure would load no
    # MathJax at all and silently render the figure's labels as raw source.
    app.set_html_assets_policy("always")


# -- sphinxcontrib-bibtex ----------------------------------------------------

bibtex_bibfiles = ["refs.bib"]

# -- MathJax -----------------------------------------------------------------

# Sphinx defaults to MathJax 4, but plotly refuses to typeset against anything other than MathJax
# 2 or 3, so figure labels would stay as raw LaTeX.  Of those two it has to be 2, even though it is
# end-of-life: with MathJax 3, Firefox mismeasures the ``ex``-sized nested ``<svg>`` that MathJax
# emits, and a figure's legend comes out with its entries overlapping each other and its title and
# running off the right edge.  MathJax 2 is the path plotly actually tests, and it lays out
# correctly in both Firefox and Chromium.  The bundle must be an SVG one, because plotly switches
# the output jax to SVG while converting.
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
mathjax2_config = {
    "tex2jax": {
        # MyST marks a notebook page's whole section ``tex2jax_ignore``, so that a stray ``$`` in a
        # code cell is not typeset.  Naming the classes that Sphinx and MyST put on actual math
        # re-enables it inside that section, without re-enabling the code cells.
        "processClass": "math|tex2jax_process|mathjax_process|output_area",
    },
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

# -- HTML output -------------------------------------------------------------

html_theme = "qiskit-ecosystem"
html_title = f"{project} {release}"
