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

"""HTML representation utilities."""

from collections.abc import Iterable, Sequence
from typing import Self

_CSS_CLASS = "qp-html-table"

_STYLE_BLOCK = """
<style>
.qp-html-table {
    border-collapse: collapse;
    font-family: monospace;
    font-size: 0.9em;
}
.qp-html-table caption {
    text-align: left;
    font-weight: bold;
    padding-bottom: 4px;
}
.qp-html-table th, .qp-html-table td {
    border: 1px solid #bbb;
    padding: 4px 8px;
    text-align: left;
}
.qp-html-table th {
    background: #f0f0f0;
    color: #111;
}
.qp-html-table tbody tr:hover td {
    background: rgba(0, 0, 0, 0.15);
}
@media (prefers-color-scheme: dark) {
    .qp-html-table th, .qp-html-table td {
        border-color: #555;
    }
    .qp-html-table th {
        background: #2a2a2a;
        color: #eee;
    }
    .qp-html-table tbody tr:hover td {
        background: rgba(255, 255, 255, 0.4);
    }
}
</style>
""".strip()


class HTMLTable:
    """A builder for HTML table representations.

    Each method returns ``self`` to allow chaining. The object itself implements
    ``_repr_html_`` so it can be returned directly from a class's ``_repr_html_`` method
    or displayed inline in a Jupyter notebook.

    .. code:: python

        table = (
            HTMLTable()
            .set_caption("My Table")
            .set_columns(["Name", "Value"])
            .add_row(["foo", "1"])
            .add_row(["bar", "2"])
        )
        # In a notebook, ``table`` renders as an HTML table.
        # Or: html_str = table._repr_html_()
    """

    def __init__(self):
        self._caption: str | None = None
        self._columns: list[str] = []
        self._rows: list[list[str]] = []

    def set_caption(self, caption: str) -> Self:
        """Set the table caption.

        Args:
            caption: The caption text shown above the table.

        Returns:
            This instance.
        """
        self._caption = caption
        return self

    def extend_columns(self, columns: Iterable[str]) -> Self:
        """Extend the column header names.

        Args:
            columns: The column header labels to extend with.

        Returns:
            This instance.
        """
        if self._rows:
            raise ValueError("The columns cannot be extended after rows have been added.")
        self._columns.extend(columns)
        return self

    def add_row(self, cells: Sequence[str], empty_char: str = "&mdash;") -> Self:
        """Append a row of pre-formatted cell strings.

        Args:
            cells: One string per column. HTML entities (e.g. ``&mdash;``) are allowed.
                Must have the same length as the columns set via :meth:`set_columns`.
            empty_char: Char to substitute when a cell is empty.

        Returns:
            This instance.

        Raises:
            ValueError: If ``cells`` has a different length than the number of columns.
        """
        if self._columns and len(cells) != len(self._columns):
            raise ValueError(f"Expected {len(self._columns)} cells per row, got {len(cells)}.")
        self._rows.append([empty_char if not val else val for val in cells])
        return self

    def _repr_html_(self) -> str:
        parts = [_STYLE_BLOCK, f"<table class='{_CSS_CLASS}'>"]

        if self._caption is not None:
            parts.append(f"<caption>{self._caption}</caption>")

        if self._columns:
            ths = "".join(f"<th>{c}</th>" for c in self._columns)
            parts.append(f"<thead><tr>{ths}</tr></thead>")

        if self._rows:
            rows_html = "".join(
                "<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in self._rows
            )
            parts.append(f"<tbody>{rows_html}</tbody>")

        parts.append("</table>")
        return "".join(parts)
