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

"""Tests for the TeX macro expansion the documentation build applies to every math node.

These cover the substitution rules, including the ones that keep ordinary TeX from being mistaken
for a macro use. The module under test is deliberately free of Sphinx and docutils imports, so
these run without the ``docs`` extra installed.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "docs" / "_ext"))

from tex_macros import MacroError, expand  # noqa: E402

# The macros the documentation actually defines, in the shape ``docs/conf.py`` gives them.
MACROS = {
    "Z": r"\mathbb{Z}",
    "E": r"\mathcal{E}",
    "P": r"\mathcal{P}",
    "U": r"\mathcal{U}",
    "ip": [r"\langle #1, #2 \rangle", 2],
    "bra": [r"\langle #1 |", 1],
    "ket": [r"| #1 \rangle", 1],
    "opbra": [r"\langle\!\langle #1 |", 1],
    "opket": [r"| #1 \rangle\!\rangle", 1],
}


@pytest.mark.parametrize(
    ("tex", "expected"),
    [
        # Zero-argument macros, including where a subscript or superscript follows directly.
        (r"\P^S", r"\mathcal{P}^S"),
        (r"\E_m = \U_mG", r"\mathcal{E}_m = \mathcal{U}_mG"),
        # One- and two-argument macros.
        (r"\opket{X}", r"| X \rangle\!\rangle"),
        (r"\ip{A}{B}", r"\langle A, B \rangle"),
        # An argument holding further macro uses expands all the way down.
        (r"\opket{\ket{m}\bra{m}}", r"| | m \rangle\langle m | \rangle\!\rangle"),
        # An argument group ends at its own closing brace, not the first one.
        (r"\opbra{G^\dagger(Q \otimes Z^{x})}", r"\langle\!\langle G^\dagger(Q \otimes Z^{x}) |"),
        # A control sequence runs to the end of its letters, so this is not a use of ``\P``.
        (r"\Phi", r"\Phi"),
        # Undefined macros are left exactly as written.
        (r"\frac{1}{2} \otimes \delta_{y,b}", r"\frac{1}{2} \otimes \delta_{y,b}"),
    ],
)
def test_expands_macro_uses(tex, expected):
    """Defined macros are replaced by their bodies and everything else is left alone."""
    assert expand(tex, MACROS) == expected


def test_line_break_does_not_start_a_macro_name():
    r"""The second backslash of a ``\\`` line break cannot begin a macro name.

    The formalism page's aligned environments contain ``\\`` followed by a macro, so reading the
    escape as one token rather than scanning for a pattern is what keeps the two apart.
    """
    assert expand(r"a \\ \P^N", MACROS) == r"a \\ \mathcal{P}^N"
    # Without the intervening space, ``P`` is literal text following the line break.
    assert expand(r"a \\P", MACROS) == r"a \\P"


def test_reads_an_argument_given_without_braces():
    r"""A single token is a complete argument, as in ``\ket m``."""
    assert expand(r"\ket m", MACROS) == r"| m \rangle"
    assert expand(r"\ket\psi", MACROS) == r"| \psi \rangle"


def test_rejects_a_use_with_a_missing_argument():
    """A macro whose argument never arrives is an error rather than silent output."""
    with pytest.raises(MacroError, match="ran out of input"):
        expand(r"x + \ket", MACROS)


def test_rejects_an_unclosed_argument_group():
    """An argument group that is never closed is an error."""
    with pytest.raises(MacroError, match="unclosed argument group"):
        expand(r"\ket{m", MACROS)


def test_rejects_a_definition_referring_to_an_argument_it_does_not_take():
    """A body using more arguments than the definition declares is an error."""
    with pytest.raises(MacroError, match="argument #2 but takes 1"):
        expand(r"\pair{x}", {"pair": [r"#1 and #2", 1]})


def test_rejects_a_macro_that_expands_to_itself():
    """Expansion that cannot terminate fails the build instead of looping."""
    with pytest.raises(MacroError, match="did not terminate"):
        expand(r"\loop", {"loop": r"\loop"})
