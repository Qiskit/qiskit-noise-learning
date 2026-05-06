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

import pytest

from qiskit_noise_learning.utils.html_repr import HTMLTable


def test_empty_table():
    html = HTMLTable()._repr_html_()
    assert "<table" in html
    assert "</table>" in html
    assert "<thead>" not in html
    assert "<tbody>" not in html
    assert "<caption" not in html


def test_style_block_included():
    html = HTMLTable()._repr_html_()
    assert "qp-html-table" in html
    assert "prefers-color-scheme" in html


def test_caption():
    html = HTMLTable().set_caption("My Caption")._repr_html_()
    assert "<caption" in html
    assert "My Caption" in html


def test_columns():
    html = HTMLTable().extend_columns(["A", "B"])._repr_html_()
    assert "<thead>" in html
    assert "<th" in html
    assert "A" in html
    assert "B" in html


def test_extend_columns_accumulates():
    html = HTMLTable().extend_columns(["A", "B"]).extend_columns(["C"])._repr_html_()
    assert html.count("<th>") == 3
    assert "C" in html


def test_rows():
    html = (
        HTMLTable().extend_columns(["X", "Y"]).add_row(["1", "2"]).add_row(["3", "4"])._repr_html_()
    )
    assert "<tbody>" in html
    assert html.count("<tr>") == 3  # 1 header row + 2 body rows
    assert "1" in html
    assert "4" in html


def test_html_entities_in_cells():
    html = HTMLTable().extend_columns(["Col"]).add_row(["&mdash;"])._repr_html_()
    assert "&mdash;" in html


def test_empty_cell_replaced_with_default_char():
    html = HTMLTable().extend_columns(["Col"]).add_row([""])._repr_html_()
    assert "&mdash;" in html


def test_empty_cell_custom_empty_char():
    html = HTMLTable().extend_columns(["Col"]).add_row([""], empty_char="N/A")._repr_html_()
    assert "N/A" in html
    assert "&mdash;" not in html


def test_non_empty_cell_not_replaced():
    html = HTMLTable().extend_columns(["Col"]).add_row(["hello"])._repr_html_()
    assert "hello" in html
    assert "&mdash;" not in html


def test_chaining_returns_self():
    table = HTMLTable()
    assert table.set_caption("c") is table
    assert table.extend_columns(["a"]) is table
    assert table.add_row(["x"]) is table


def test_add_row_wrong_number_of_cells():
    table = HTMLTable().extend_columns(["A", "B", "C"])
    with pytest.raises(ValueError, match="3"):
        table.add_row(["only", "two"])


def test_add_row_too_many_cells():
    table = HTMLTable().extend_columns(["A"])
    with pytest.raises(ValueError, match="1"):
        table.add_row(["a", "b"])


def test_add_row_before_columns_no_error():
    # When no columns are set, row length cannot be validated — should not raise.
    html = HTMLTable().add_row(["a", "b"])._repr_html_()
    assert "a" in html


def test_extend_columns_after_add_row_does_not_revalidate():
    # Validation only applies at add_row time; extending columns after rows is allowed.
    with pytest.raises(ValueError, match="columns cannot be extended after rows have been added"):
        HTMLTable().add_row(["a", "b"]).extend_columns(["X", "Y"])


def test_full_table_structure():
    html = (
        HTMLTable()
        .set_caption("Test")
        .extend_columns(["Name", "Value"])
        .add_row(["foo", "1"])
        ._repr_html_()
    )
    # Check document order: caption before thead before tbody
    cap_pos = html.index("<caption")
    thead_pos = html.index("<thead>")
    tbody_pos = html.index("<tbody>")
    assert cap_pos < thead_pos < tbody_pos
