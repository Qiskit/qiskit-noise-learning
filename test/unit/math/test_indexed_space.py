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

import math

from qiskit_noise_learning.math import EnumeratedIndexedSpace, IndexedSpace


def test_enumerated_dim_and_contains():
    space = EnumeratedIndexedSpace(frozenset({"a", "b", "c"}))

    assert space.dim == 3
    assert "a" in space
    assert "z" not in space


def test_enumerated_iter_and_len():
    space = EnumeratedIndexedSpace(frozenset({"a", "b"}))

    assert len(space) == 2
    assert set(space) == {"a", "b"}


def test_enumerated_empty():
    space = EnumeratedIndexedSpace(frozenset())

    assert space.dim == 0
    assert len(space) == 0
    assert "a" not in space


def test_dim_allows_infinite():
    # the dim contract admits math.inf for infinite-dimensional spaces
    class _InfiniteSpace(IndexedSpace):
        @property
        def dim(self) -> float:
            return math.inf

        def __contains__(self, index: object) -> bool:
            return True

    space = _InfiniteSpace()
    assert space.dim == math.inf
    assert "anything" in space
