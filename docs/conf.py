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
    "qiskit_sphinx_theme",
]

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

# -- autodoc / autosummary ---------------------------------------------------

# Only the class docstring is used for classes -- constructor parameters are
# documented in the class docstring, not in __init__.
autoclass_content = "class"
autodoc_member_order = "bysource"
autosummary_generate = True

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

# -- HTML output -------------------------------------------------------------

html_theme = "qiskit-ecosystem"
html_title = f"{project} {release}"
