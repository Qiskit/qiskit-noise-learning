"""Expand TeX macros in math nodes so the built HTML carries no macro definitions.

This extension rewrites every math node to its fully expanded form at build time, leaving
HTML that reads as if each symbol had been written out by hand. Macros are defined once in
the ``math_macros`` configuration value, in the shape ``tex_macros`` documents.
"""

from typing import Any

from docutils import nodes
from sphinx.application import Sphinx
from sphinx.transforms.post_transforms import SphinxPostTransform
from sphinx.util import logging
from tex_macros import MacroError, expand

logger = logging.getLogger(__name__)


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
            tex = node.astext()
            try:
                expanded = expand(tex, macros)
            except MacroError as exc:
                logger.warning(str(exc), location=node, type="math", subtype="macro")
                continue
            if expanded != tex:
                node.children = [nodes.Text(expanded)]


def setup(app: Sphinx) -> dict[str, Any]:
    app.add_config_value("math_macros", {}, "env")
    app.add_post_transform(ExpandMathMacros)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
