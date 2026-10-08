"""Expand TeX macros in math nodes so the built HTML carries no macro definitions.

The published documentation site does not build this project's HTML; it ingests it. The
ingestion extracts the ``role="main"`` element and nothing else, which drops the
``window.MathJax`` configuration block that ``sphinx.ext.mathjax`` writes the macro
definitions into. Math itself is taken verbatim out of the ``span.math``/``div.math``
markup, so a macro that survives into the content arrives at the reader undefined.

This extension rewrites every math node to its fully expanded form at build time, leaving
HTML that reads as if each symbol had been written out by hand. Macros are defined once in
the ``math_macros`` configuration value, in the same shape ``mathjax4_config`` used:
either ``name: body`` or ``name: [body, number_of_arguments]``.
"""

from collections.abc import Mapping, Sequence
from typing import Any

from docutils import nodes
from sphinx.application import Sphinx
from sphinx.transforms.post_transforms import SphinxPostTransform
from sphinx.util import logging

logger = logging.getLogger(__name__)

# Expansion repeats until nothing changes, because a macro's arguments are spliced in
# unexpanded and only rewritten on the following pass. A handful of passes covers any
# nesting depth the documentation uses; exceeding this means a macro expands to itself.
MAX_PASSES = 16


class MacroError(ValueError):
    """A macro definition or use that cannot be expanded."""


def expand(tex: str, macros: Mapping[str, Any]) -> str:
    """Return ``tex`` with every macro in ``macros`` replaced by its definition.

    Args:
        tex: TeX source, as it appears in the body of a math node.
        macros: Macro definitions, each either a body string or a ``[body, n_args]``
            pair, matching the shape of MathJax's own ``macros`` configuration.

    Returns:
        The source with all defined macros expanded, including inside the arguments of
        other macros.

    Raises:
        MacroError: A macro is used without enough arguments, or expansion does not
            terminate.
    """
    definitions = _definitions(macros)
    result = tex
    for _ in range(MAX_PASSES):
        result, changed = _expand_once(result, definitions)
        if not changed:
            return result
    raise MacroError(f"macro expansion did not terminate after {MAX_PASSES} passes: {tex!r}")


def _definitions(macros: Mapping[str, Any]) -> dict[str, tuple[str, int]]:
    """Normalize the configured macros to ``{name: (body, n_args)}``."""
    definitions = {}
    for name, definition in macros.items():
        if isinstance(definition, str):
            definitions[name] = (definition, 0)
        elif isinstance(definition, Sequence) and len(definition) == 2:
            body, n_args = definition
            definitions[name] = (str(body), int(n_args))
        else:
            raise MacroError(f"macro {name!r} is neither a body nor a [body, n_args] pair")
    return definitions


def _expand_once(tex: str, definitions: Mapping[str, tuple[str, int]]) -> tuple[str, bool]:
    """Expand every macro use in ``tex`` once, returning the result and whether it changed.

    The scan walks the source rather than matching a pattern, so that an escape sequence
    consumes both of its characters. That is what keeps the ``Z`` in a line break
    followed by ``\\Z`` from reading as the ``\\Z`` macro.
    """
    out: list[str] = []
    changed = False
    i, end = 0, len(tex)
    while i < end:
        if tex[i] != "\\":
            out.append(tex[i])
            i += 1
            continue
        name_start = i + 1
        if name_start >= end or not tex[name_start].isalpha():
            # An escape sequence such as ``\\``, ``\{`` or ``\%``: both characters are
            # literal, and the second one never starts a macro name.
            out.append(tex[i : name_start + 1])
            i = name_start + 1
            continue
        name_end = name_start
        while name_end < end and tex[name_end].isalpha():
            name_end += 1
        # A macro name runs to the end of the letters, so ``\Ztest`` is its own name and
        # not a use of ``\Z``.
        name = tex[name_start:name_end]
        if name not in definitions:
            out.append(tex[i:name_end])
            i = name_end
            continue
        body, n_args = definitions[name]
        args, i = _read_args(tex, name_end, n_args, name)
        out.append(_substitute(body, args))
        changed = True
    return "".join(out), changed


def _read_args(tex: str, start: int, n_args: int, name: str) -> tuple[list[str], int]:
    """Read ``n_args`` arguments of ``\\name`` from ``start``, returning them and the end."""
    args: list[str] = []
    i = start
    for _ in range(n_args):
        while i < len(tex) and tex[i].isspace():
            i += 1
        if i >= len(tex):
            raise MacroError(f"macro {name!r} takes {n_args} argument(s) but ran out of input")
        if tex[i] == "{":
            arg, i = _read_group(tex, i, name)
        elif tex[i] == "\\":
            # A single-token argument given as a control sequence, as in ``\ket\psi``.
            token_end = i + 1
            while token_end < len(tex) and tex[token_end].isalpha():
                token_end += 1
            arg = tex[i : max(token_end, i + 2)]
            i = max(token_end, i + 2)
        else:
            arg, i = tex[i], i + 1
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


def _substitute(body: str, args: Sequence[str]) -> str:
    """Return ``body`` with ``#1``, ``#2``, ... replaced by ``args``."""
    out: list[str] = []
    i = 0
    while i < len(body):
        if body[i] == "\\":
            out.append(body[i : i + 2])
            i += 2
            continue
        if body[i] == "#" and i + 1 < len(body):
            following = body[i + 1]
            if following == "#":
                out.append("#")
                i += 2
                continue
            if following.isdigit() and following != "0" and int(following) <= len(args):
                out.append(args[int(following) - 1])
                i += 2
                continue
        out.append(body[i])
        i += 1
    return "".join(out)


class ExpandMathMacros(SphinxPostTransform):
    """Rewrite every math node in the document to its macro-free form."""

    default_priority = 900

    def run(self, **kwargs: Any) -> None:
        macros = self.config.math_macros
        if not macros:
            return
        for node in list(self.document.findall(nodes.math)) + list(
            self.document.findall(nodes.math_block)
        ):
            try:
                expanded = expand(node.astext(), macros)
            except MacroError as exc:
                logger.warning(str(exc), location=node, type="math", subtype="macro")
                continue
            if expanded != node.astext():
                node.children = [nodes.Text(expanded)]


def setup(app: Sphinx) -> dict[str, Any]:
    app.add_config_value("math_macros", {}, "env")
    app.add_post_transform(ExpandMathMacros)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
