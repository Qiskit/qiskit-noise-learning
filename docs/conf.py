"""Sphinx configuration for the qiskit-noise-learning documentation."""

import sys
from pathlib import Path

import qiskit_noise_learning

# Local extensions, which reshape the build output for the published documentation site.
sys.path.insert(0, str(Path(__file__).parent / "_ext"))

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
    # The published site ingests this build's HTML rather than building its own, and
    # supports neither TeX macros nor a bibliography shared between pages.
    "expand_math_macros",
    "auto_references",
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

# Number the references in order of first citation, as a paper does.  "unsrt" leaves the label
# and sorting styles unset, which resolve to pybtex's "number" and "none" plugins.
bibtex_default_style = "unsrt"

# Every page carries its own bibliography (see the auto_references extension), so a work cited
# on two pages legitimately appears in two of them.  sphinxcontrib-bibtex reports that as a
# duplicate, and -W would turn it into a build failure.  A key duplicated *within* one page is a
# separate subtype, "duplicate_local_citation", and still fails.
suppress_warnings = ["bibtex.duplicate_citation", "bibtex.duplicate_label"]

# -- Math --------------------------------------------------------------------

# Expanded into the math itself by the expand_math_macros extension, so no definitions reach
# the HTML.  Deliberately not also declared to MathJax: a local build then renders exactly what
# the published page will, so anything the expansion misses is visible here as a MathJax error
# rather than only after the site ingests the build.
math_macros = {
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

# -- HTML output -------------------------------------------------------------

html_theme = "qiskit-ecosystem"
html_title = f"{project} {release}"
