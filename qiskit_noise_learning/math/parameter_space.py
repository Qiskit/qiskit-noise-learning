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

"""ParameterSpace"""

from abc import ABC, abstractmethod
from collections.abc import Hashable, Iterator
from typing import Generic, TypeVar

Index = TypeVar("Index", bound=Hashable)


class ParameterSpace(Generic[Index], ABC):
    """Abstract description of vector space with arbitrary basis index types."""

    @property
    @abstractmethod
    def dim(self) -> int | float:
        """The dimension (cardinality) of the space.

        May be ``math.inf`` for infinite-dimensional spaces.
        """

    @abstractmethod
    def __contains__(self, index: object) -> bool:
        """Check whether an index is a valid member of this space."""


class EnumeratedParameterSpace(ParameterSpace[Index]):
    """A parameter space backed by an explicit finite collection of indices.

    Args:
        indices: The collection of valid indices in this space.
    """

    def __init__(self, indices: frozenset[Index]):
        self._indices = indices

    @property
    def dim(self) -> int:
        return len(self._indices)

    def __contains__(self, index: object) -> bool:
        return index in self._indices

    def __iter__(self) -> Iterator[Index]:
        return iter(self._indices)

    def __len__(self) -> int:
        return len(self._indices)
