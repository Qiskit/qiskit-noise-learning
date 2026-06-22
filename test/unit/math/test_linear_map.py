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

from qiskit_noise_learning.math import (
    EnumeratedParameterSpace,
    IndexedMatrix,
    IndexedVector,
    LinearMap,
)


class _ExampleMap(LinearMap):
    """A linear map defined by an explicit mapping from output index to sparse row."""

    def __init__(self, rows: dict[str, IndexedVector]):
        self._rows = rows
        inputs = frozenset(idx for row in rows.values() for idx in row)
        super().__init__(
            input_space=EnumeratedParameterSpace(inputs),
            output_space=EnumeratedParameterSpace(frozenset(rows)),
        )

    def row(self, output_index: str) -> IndexedVector:
        return self._rows[output_index]


def test_to_indexed_matrix():
    example = _ExampleMap(
        {
            "o0": IndexedVector({"i0": 1.0, "i1": 2.0}),
            "o1": IndexedVector({"i1": 3.0}),
        }
    )

    matrix = example.to_indexed_matrix(["o0", "o1"])

    assert isinstance(matrix, IndexedMatrix)
    assert matrix.row_index_map.keys() == {"o0", "o1"}
    assert matrix.column_index_map.keys() == {"i0", "i1"}
    # __getitem__ fills omitted columns with zeros
    assert matrix["o0"] == IndexedVector({"i0": 1.0, "i1": 2.0})
    assert matrix["o1"] == IndexedVector({"i0": 0.0, "i1": 3.0})


def test_to_indexed_matrix_subset_of_rows():
    example = _ExampleMap(
        {
            "o0": IndexedVector({"i0": 1.0}),
            "o1": IndexedVector({"i1": 3.0}),
        }
    )

    matrix = example.to_indexed_matrix(["o1"])

    assert matrix.row_index_map.keys() == {"o1"}
    assert matrix.column_index_map.keys() == {"i1"}


def test_to_indexed_matrix_drops_zero_rows():
    # An all-zero row contributes no columns and is not added (matching add_rows semantics).
    example = _ExampleMap(
        {
            "o0": IndexedVector({"i0": 1.0}),
            "zero": IndexedVector(),
        }
    )

    matrix = example.to_indexed_matrix(["o0", "zero"])

    assert matrix.row_index_map.keys() == {"o0"}


def test_to_indexed_matrix_applies_forward_via_matmul():
    # The materialized matrix reproduces forward application: (M @ x)[o] == evaluate(o, x).
    example = _ExampleMap(
        {
            "o0": IndexedVector({"i0": 1.0, "i1": 2.0}),
            "o1": IndexedVector({"i1": 3.0}),
        }
    )
    x = IndexedVector({"i0": 5.0, "i1": 7.0})

    matrix = example.to_indexed_matrix(["o0", "o1"])
    result = matrix @ x

    assert result["o0"] == example.evaluate("o0", x)
    assert result["o1"] == example.evaluate("o1", x)
