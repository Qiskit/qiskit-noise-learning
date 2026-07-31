"""Sphinx configuration for the qiskit-noise-learning documentation."""

import re

from docutils import nodes

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

# amsmath: support LaTeX align/aligned environments; dollarmath: $...$ / $$...$$ math.
myst_enable_extensions = ["amsmath", "dollarmath"]

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

# Plotly's HTML renderers unconditionally pull in MathJax 2.7.5 alongside every figure, but
# Sphinx already loads a much newer MathJax for the page.  Drop plotly's copy so the two do
# not fight over ``window.MathJax``; the figures keep their own plotly.js script tag.
_PLOTLY_MATHJAX_SCRIPT = re.compile(r'<script src="[^"]*mathjax[^"]*"></script>', re.IGNORECASE)


def _strip_plotly_mathjax(_app, doctree):
    for node in doctree.findall(nodes.raw):
        if node.get("format") != "html":
            continue
        html = node.astext()
        stripped = _PLOTLY_MATHJAX_SCRIPT.sub("", html)
        if stripped != html:
            node.children = [nodes.Text(stripped)]


def setup(app):
    """Register documentation-only Sphinx hooks."""
    app.connect("doctree-read", _strip_plotly_mathjax)


# -- sphinxcontrib-bibtex ----------------------------------------------------

bibtex_bibfiles = ["refs.bib"]

# -- MathJax -----------------------------------------------------------------
mathjax3_config = {
    "tex": {
        "macros": {
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
    }
}

# -- HTML output -------------------------------------------------------------

html_theme = "qiskit-ecosystem"
html_title = f"{project} {release}"
