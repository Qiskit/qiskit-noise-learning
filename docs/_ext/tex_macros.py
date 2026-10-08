"""Expand TeX macros in a string, the way MathJax would expand them in a browser.

This module is plain string handling with no Sphinx or docutils involvement, so it can be
tested without the documentation toolchain installed.
"""

import re
from collections.abc import Mapping, Sequence
from typing import Any

# A macro's arguments are spliced in unexpanded and only rewritten on the pass after, so
# expansion repeats until nothing changes. Exceeding this many passes means a macro
# expands to itself, whether directly or by way of others.
MAX_PASSES = 16

_ARGUMENT = re.compile(r"#([1-9])")


class MacroError(ValueError):
    """A macro definition or use that cannot be expanded."""


def expand(tex: str, macros: Mapping[str, Any]) -> str:
    """Return ``tex`` with every macro in ``macros`` replaced by its definition.

    Args:
        tex: TeX source, as it appears in the body of a math node.
        macros: Macro definitions, each either a body or a ``[body, n_args]`` pair,
            matching the shape of MathJax's own ``macros`` configuration.

    Returns:
        The source with all defined macros expanded, including inside the arguments of
        other macros.

    Raises:
        MacroError: A macro is used without enough arguments, a definition refers to an
            argument it does not take, or expansion does not terminate.
    """
    result = tex
    for _ in range(MAX_PASSES):
        result, changed = _expand_once(result, macros)
        if not changed:
            return result
    raise MacroError(f"macro expansion did not terminate after {MAX_PASSES} passes: {tex!r}")


def _expand_once(tex: str, macros: Mapping[str, Any]) -> tuple[str, bool]:
    """Expand every macro use in ``tex`` once, returning the result and whether it changed.

    The scan reads a token at a time rather than matching a pattern, so that an escape
    sequence consumes both of its characters. That is what keeps the ``Z`` of a ``\\\\``
    line break followed by ``Z`` from reading as the ``\\Z`` macro.
    """
    out: list[str] = []
    changed = False
    i = 0
    while i < len(tex):
        if tex[i] != "\\":
            out.append(tex[i])
            i += 1
            continue
        token, i = _read_token(tex, i)
        # A control sequence runs to the end of its letters, so ``\Phi`` is a token of its
        # own rather than a use of ``\P``. An escape such as ``\\`` or ``\%`` names no
        # macro either, so both fall through to being copied unchanged.
        name = token[1:]
        if name not in macros:
            out.append(token)
            continue
        definition = macros[name]
        body, n_args = (definition, 0) if isinstance(definition, str) else definition
        args, i = _read_args(tex, i, n_args, name)
        out.append(_substitute(body, args, name))
        changed = True
    return "".join(out), changed


def _read_token(tex: str, i: int) -> tuple[str, int]:
    """Read the single TeX token at ``i``, returning it and the position after it.

    A token is a control sequence -- a backslash and the letters following it, or a
    backslash escaping one other character -- or else a single character.
    """
    if tex[i] != "\\":
        return tex[i], i + 1
    end = i + 1
    while end < len(tex) and tex[end].isalpha():
        end += 1
    if end == i + 1:
        # No letters followed, so the backslash escapes the single character after it --
        # or nothing at all, at the very end of the source.
        end = min(end + 1, len(tex))
    return tex[i:end], end


def _read_args(tex: str, start: int, n_args: int, name: str) -> tuple[list[str], int]:
    """Read ``n_args`` arguments of ``\\name`` from ``start``, returning them and the end."""
    args = []
    i = start
    for _ in range(n_args):
        while i < len(tex) and tex[i].isspace():
            i += 1
        if i >= len(tex):
            raise MacroError(f"macro {name!r} takes {n_args} argument(s) but ran out of input")
        if tex[i] == "{":
            arg, i = _read_group(tex, i, name)
        else:
            # An unbraced argument is one token, as in ``\ket m`` or ``\ket\psi``.
            arg, i = _read_token(tex, i)
        args.append(arg)
    return args, i


def _read_group(tex: str, start: int, name: str) -> tuple[str, int]:
    """Read the balanced ``{...}`` group at ``start``, returning its content and the end."""
    depth = 0
    i = start
    while i < len(tex):
        if tex[i] == "\\":
            # An escaped brace is content, not a delimiter.
            i += 2
            continue
        if tex[i] == "{":
            depth += 1
        elif tex[i] == "}":
            depth -= 1
            if depth == 0:
                return tex[start + 1 : i], i + 1
        i += 1
    raise MacroError(f"macro {name!r} has an unclosed argument group")


def _substitute(body: str, args: Sequence[str], name: str) -> str:
    """Return ``body`` with ``#1``, ``#2``, ... replaced by ``args``."""

    def replace(match: re.Match) -> str:
        index = int(match[1])
        if index > len(args):
            raise MacroError(f"macro {name!r} refers to argument #{index} but takes {len(args)}")
        return args[index - 1]

    return _ARGUMENT.sub(replace, body)
