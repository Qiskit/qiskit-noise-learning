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

"""Tests that the hand-maintained API reference pages stay in sync with the public API.

Each ``docs/apidocs/*.rst`` page enumerates the public classes and functions of one subpackage by
hand in its ``autosummary`` blocks. These tests guard against drift: a public name added to a
subpackage but forgotten in its documentation page, a stale entry left behind after a rename or
removal, and a page missing from (or spuriously listed in) the API reference table of contents.
"""

import importlib
import re
from pathlib import Path

import pytest

APIDOCS = Path(__file__).resolve().parents[2] / "docs" / "apidocs"

_CURRENTMODULE = re.compile(r"\.\.\s+currentmodule::\s+(\S+)")
_AUTOOBJECT = re.compile(r"\.\.\s+auto(?:data|class|function)::\s+(\S+)")
_AUTOSUMMARY = re.compile(r"\.\.\s+autosummary::")
_DIRECTIVE = re.compile(r"\.\.\s+\w[\w-]*::")
_IDENTIFIER = re.compile(r"[A-Za-z_]\w*")


def _api_pages():
    """Return the API reference pages to check (every ``apidocs`` page except the index)."""
    pages = sorted(page for page in APIDOCS.glob("*.rst") if page.name != "index.rst")
    assert pages, f"no API reference pages found under {APIDOCS}"
    return pages


def _parse_page(path):
    """Return ``(module_name, documented_names)`` parsed from an ``apidocs`` page.

    ``documented_names`` collects the members listed in ``autosummary`` blocks together with any
    ``autodata``/``autoclass``/``autofunction`` targets, mirroring exactly what the page documents.
    """
    module_name = None
    documented = set()
    in_autosummary = False
    for line in path.read_text().splitlines():
        if match := _CURRENTMODULE.match(line):
            module_name = match.group(1)
            in_autosummary = False
        elif match := _AUTOOBJECT.match(line):
            documented.add(match.group(1))
            in_autosummary = False
        elif _AUTOSUMMARY.match(line):
            in_autosummary = True
        elif _DIRECTIVE.match(line):
            in_autosummary = False  # any other directive ends the block
        elif in_autosummary:
            stripped = line.strip()
            if not stripped:
                continue  # blank lines separate the options from the entries
            if not line[0].isspace():
                in_autosummary = False  # a dedented, non-blank line ends the block
            elif not stripped.startswith(":") and _IDENTIFIER.fullmatch(stripped):
                documented.add(stripped)
    assert module_name is not None, f"{path.name} has no `.. currentmodule::` directive"
    return module_name, documented


def _public_names(module):
    """Return the public names that originate in ``module``, excluding re-imports.

    A name belongs to ``module``'s own public API when it does not start with an underscore and its
    ``__module__`` lies within ``module``. Names re-imported from other subpackages (helpers such as
    :class:`~.LinearMap`) and module-level type aliases (whose ``__module__`` points at the aliased
    class) are excluded, matching what ``autosummary`` generates for the page.
    """
    prefix = module.__name__
    names = set()
    for name, value in vars(module).items():
        if name.startswith("_"):
            continue
        origin = getattr(value, "__module__", None)
        if isinstance(origin, str) and (origin == prefix or origin.startswith(f"{prefix}.")):
            names.add(name)
    return names


@pytest.mark.parametrize("page", _api_pages(), ids=lambda page: page.stem)
def test_public_names_are_documented(page):
    """Every public class/function originating in a subpackage appears on its API page."""
    module_name, documented = _parse_page(page)
    module = importlib.import_module(module_name)
    missing = sorted(_public_names(module) - documented)
    assert not missing, (
        f"{page.name} is missing public name(s) {missing} exported by {module_name}; add them to "
        "the page's autosummary block (or drop them from the subpackage's public API)."
    )


@pytest.mark.parametrize("page", _api_pages(), ids=lambda page: page.stem)
def test_documented_names_exist(page):
    """Every name listed on an API page still resolves on its module."""
    module_name, documented = _parse_page(page)
    module = importlib.import_module(module_name)
    stale = sorted(name for name in documented if not hasattr(module, name))
    assert not stale, (
        f"{page.name} documents name(s) {stale} that no longer exist on {module_name}; "
        "remove or rename the stale entries."
    )


def test_index_lists_every_api_page():
    """The API reference index toctree lists exactly the existing subpackage pages."""
    index = APIDOCS / "index.rst"
    listed = {
        stripped
        for line in index.read_text().splitlines()
        if line[:1].isspace() and _IDENTIFIER.fullmatch(stripped := line.strip())
    }
    pages = {page.stem for page in _api_pages()}
    assert listed == pages, (
        f"apidocs/index.rst toctree {sorted(listed)} does not match the page files "
        f"{sorted(pages)}; keep the table of contents in sync with the pages."
    )
