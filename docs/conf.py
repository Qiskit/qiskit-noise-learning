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
    "myst_parser",
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
}

# -- MyST (Markdown) ---------------------------------------------------------

# amsmath: support LaTeX align/aligned environments; dollarmath: $...$ / $$...$$ math.
myst_enable_extensions = ["amsmath", "dollarmath"]

# -- sphinxcontrib-bibtex ----------------------------------------------------

bibtex_bibfiles = ["formalism/refs.bib"]

# -- MathJax -----------------------------------------------------------------

# Custom macros ported from the formalism document's LaTeX preamble, so that the
# equation bodies can be transferred nearly verbatim.
mathjax3_config = {
    "tex": {
        "macros": {
            # No-argument macros.
            "real": r"\mathbb{R}",
            "Z": r"\mathbb{Z}",
            "symE": r"\mathcal{E}",
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
