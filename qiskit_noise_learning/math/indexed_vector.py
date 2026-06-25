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

"""IndexedVector"""

from collections.abc import Hashable
from typing import Self, TypeVar

Index = TypeVar("BasisIndex", bound=Hashable)


class IndexedVector(dict[Index, float]):
    """A vector of floats with arbitrary index, or axis label, data."""

    def add(self, other: Self) -> Self:
        """Return a new indexed vector that is the sum of self and other."""

        new_vector = self.copy()

        for idx, value in other.items():
            new_vector[idx] = new_vector.get(idx, 0.0) + value

        return IndexedVector[Index](new_vector)

    def mul(self, const: float) -> Self:
        """Return a new indexed vector by multipling self with a constant."""
        return IndexedVector[Index]({k: const * v for k, v in self.items()})

    def __add__(self, other: Self) -> Self:
        return self.add(other)

    def __mul__(self, const: float) -> Self:
        return self.mul(const)

    def __rmul__(self, const: float) -> Self:
        return self.mul(const)
