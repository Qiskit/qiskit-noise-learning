"""Give every document that cites something its own "References" section.

A single ``bibliography`` directive serving the whole site makes each other page's
citation a cross-document link, and the published site has no central bibliography to
point at. The alternative -- a bibliography on every page -- cannot be written by hand,
because citations also belong in docstrings, and the autodoc pages those render onto are
generated and cannot carry a directive.

So this extension appends a References section to every document's source, then drops it
again from the documents that turn out to cite nothing. The removal happens while the
document is read, before Sphinx collects the page's table of contents, so an uncited page
is left with no trace of the section.

The section is a real heading rather than a rubric: the ingestion promotes only
``Methods``, ``Methods Defined Here`` and ``Attributes`` rubrics to headings, and renders
every other rubric as bold text.
"""

from pathlib import Path
from typing import Any

from docutils import nodes
from sphinx import addnodes
from sphinx.application import Sphinx
from sphinxcontrib.bibtex.nodes import bibliography as bibliography_node

# Only entries cited in this very document, which is what makes each section page-local.
_FILTER = "docname in docnames"

_MYST = f"""

## References

```{{bibliography}}
:filter: {_FILTER}
```
"""

# ``=`` titles every page under docs/apidocs, so ``-`` is the second level throughout.
_RST = f"""

References
----------

.. bibliography::
   :filter: {_FILTER}
"""

# Environment collectors read the doctree at priority 500, the page's table of contents
# among them. Pruning has to come first or an uncited page keeps a "References" entry.
_BEFORE_COLLECTORS = 400


def inject(app: Sphinx, docname: str, source: list[str]) -> None:
    """Append a References section to the source of ``docname``."""
    suffix = Path(str(app.env.doc2path(docname))).suffix
    if suffix == ".md":
        source[0] += _MYST
    elif suffix == ".rst":
        source[0] += _RST


def prune(app: Sphinx, doctree: nodes.document) -> None:
    """Remove the injected References section if the document cites nothing."""
    if any(node.get("refdomain") == "cite" for node in doctree.findall(addnodes.pending_xref)):
        return
    injected = list(doctree.findall(bibliography_node))
    if not injected:
        return
    # The injected directive is the last one in the document, and sits alone under its
    # own heading.
    node = injected[-1]
    parent = node.parent
    if isinstance(parent, nodes.section) and len(parent) == 2:
        parent.parent.remove(parent)
    else:
        parent.remove(node)


def setup(app: Sphinx) -> dict[str, Any]:
    app.connect("source-read", inject)
    app.connect("doctree-read", prune, priority=_BEFORE_COLLECTORS)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
