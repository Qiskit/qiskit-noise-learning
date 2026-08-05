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

"""Shared helpers for visualization tests.

Every figure here is read back the way the browser reads it: ``id``\\ s out of the rendered SVG, and
data out of the JSON sidecar in the rendered HTML.  Going through the rendering rather than through
the artists is the point -- an artist whose ``gid`` never made it out through the backend would
toggle nothing, and no assertion about the artist would notice.
"""

import json
import re

import pytest

from qiskit_noise_learning.visualizations.interactive_figure import trace_gid

#: Separator between the fields of a ``gid``.  Recovered from the public builder rather than
#: imported, so these helpers take an id apart exactly as the browser does.
_SEP = trace_gid("cell", ("token",)).removeprefix("trace")[0]


@pytest.fixture()
def svg_ids():
    """Return a reader ``(figure, prefix="") -> list[str]``.

    The ``id`` attributes of the rendered SVG that start with ``prefix``, in document order.
    Duplicates are kept, since a repeated id is invalid markup and would be invisible in a set.
    """

    def _svg_ids(figure, prefix=""):
        return [
            found
            for found in re.findall(r'id="([^"]+)"', figure.to_svg())
            if found.startswith(prefix)
        ]

    return _svg_ids


@pytest.fixture()
def sidecar():
    """Return a reader ``figure -> dict``: the figure's JSON sidecar, as the browser receives it."""

    def _sidecar(figure):
        payload = re.search(
            r'class="qnl-figure-data">(.*?)</script>', figure.to_html(), re.DOTALL
        ).group(1)
        return json.loads(payload)

    return _sidecar


@pytest.fixture()
def key_tokens(svg_ids):
    """Return a reader ``(figure, dimension) -> set[str]``: the tokens that legend offers."""

    def _key_tokens(figure, dimension):
        return {
            found.split(_SEP)[2]
            for found in svg_ids(figure, "key|")
            if found.split(_SEP)[1] == dimension
        }

    return _key_tokens


@pytest.fixture()
def trace_tokens(svg_ids):
    """Return a reader ``figure -> set[tuple[str, ...]]``.

    The dimension tokens each rendered data artist carries, one per dimension, in id order.
    """

    def _trace_tokens(figure):
        return {tuple(found.split(_SEP)[2:-1]) for found in svg_ids(figure, "trace|")}

    return _trace_tokens


@pytest.fixture()
def gid_separator():
    """The separator the public ``gid`` builders write between fields."""
    return _SEP
