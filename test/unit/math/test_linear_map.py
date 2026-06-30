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

from qiskit_noise_learning.math import (
    ComposedLinearMap,
    EnumeratedIndexedSpace,
    IndexedMatrix,
    IndexedVector,
    LinearMap,
)


class _ExampleMap(LinearMap):
    """A linear map defined by an explicit mapping from output index to its sparse row."""

    def __init__(self, row_map: dict[str, IndexedVector]):
        self._row_map = row_map
        inputs = frozenset(index for row in row_map.values() for index in row)
        super().__init__(
            input_space=EnumeratedIndexedSpace(inputs),
            output_space=EnumeratedIndexedSpace(frozenset(row_map)),
        )

    def rows(self, output_indices):
        output_indices = list(output_indices)
        return IndexedMatrix.from_rows(output_indices, [self._row_map[o] for o in output_indices])


@pytest.fixture()
def example():
    return _ExampleMap(
        {
            "o0": IndexedVector({"i0": 1.0, "i1": 2.0}),
            "o1": IndexedVector({"i1": 3.0}),
        }
    )


def test_spaces(example):
    assert "o0" in example.output_space
    assert "i0" in example.input_space
    assert example.output_space.dim == 2


def test_rows(example):
    matrix = example.rows(["o0", "o1"])

    assert isinstance(matrix, IndexedMatrix)
    assert matrix.row_index_map.keys() == {"o0", "o1"}
    assert matrix.column_index_map.keys() == {"i0", "i1"}
    assert matrix["o0"] == IndexedVector({"i0": 1.0, "i1": 2.0})
    assert matrix["o1"] == IndexedVector({"i0": 0.0, "i1": 3.0})

    matrix = example.rows(["o1"])

    assert matrix.row_index_map.keys() == {"o1"}
    assert matrix.column_index_map.keys() == {"i1"}


def test_projected_output(example):
    result = example.projected_output(["o0", "o1"], {"i0": 5.0, "i1": 7.0})

    assert result == IndexedVector({"o0": 19.0, "o1": 21.0})


def test_projected_output_strict_on_missing_index(example):
    with pytest.raises(KeyError):
        example.projected_output(["o0"], {"i1": 7.0})


def test_left_multiply(example):
    # a matrix whose columns are output indices of the map
    selector = IndexedMatrix.from_rows(["x"], [IndexedVector({"o0": 1.0, "o1": 1.0})])

    result = example.left_multiply(selector)

    assert result.row_index_map.keys() == {"x"}
    # x = o0 row + o1 row = {i0:1, i1:2} + {i1:3}
    assert result["x"] == IndexedVector({"i0": 1.0, "i1": 5.0})


def test_matmul():
    inner = _ExampleMap({"b0": IndexedVector({"a0": 1.0})})
    outer = _ExampleMap({"c0": IndexedVector({"b0": 1.0})})

    composed = outer @ inner

    assert isinstance(composed, ComposedLinearMap)
    # application order: inner first, outer last
    assert composed.maps == [inner, outer]


def test_compose_flattens_chain():
    a = _ExampleMap({"b0": IndexedVector({"a0": 1.0})})
    b = _ExampleMap({"c0": IndexedVector({"b0": 1.0})})
    c = _ExampleMap({"d0": IndexedVector({"c0": 1.0})})

    chain = a.compose(b).compose(c)

    assert isinstance(chain, ComposedLinearMap)
    assert chain.maps == [a, b, c]


def test_composed_rows():
    inner = _ExampleMap(
        {
            "o0": IndexedVector({"i0": 1.0, "i1": 2.0}),
            "o1": IndexedVector({"i1": 3.0}),
        }
    )
    outer = _ExampleMap(
        {
            "c0": IndexedVector({"o0": 1.0, "o1": 1.0}),
            "c1": IndexedVector({"o0": 2.0}),
        }
    )

    composed = outer @ inner
    matrix = composed.rows(["c0", "c1"])

    # c0 = o0 + o1 = {i0:1, i1:2} + {i1:3}
    assert matrix["c0"] == IndexedVector({"i0": 1.0, "i1": 5.0})
    # c1 = 2 * o0
    assert matrix["c1"] == IndexedVector({"i0": 2.0, "i1": 4.0})


def test_composed_rows_single_map():
    inner = _ExampleMap({"o0": IndexedVector({"i0": 1.0, "i1": 2.0})})

    composed = ComposedLinearMap([inner])

    assert composed.rows(["o0"])["o0"] == IndexedVector({"i0": 1.0, "i1": 2.0})


def test_composed_projected_output():
    inner = _ExampleMap(
        {
            "o0": IndexedVector({"i0": 1.0, "i1": 2.0}),
            "o1": IndexedVector({"i1": 3.0}),
        }
    )
    outer = _ExampleMap({"c0": IndexedVector({"o0": 1.0, "o1": 1.0})})

    composed = outer @ inner

    # c0 row = {i0:1, i1:5}; dotted with {i0:5, i1:7} = 5 + 35
    assert composed.projected_output(["c0"], {"i0": 5.0, "i1": 7.0}) == IndexedVector({"c0": 40.0})
