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

"""BasePattern"""

from abc import ABC
from collections.abc import Iterable
from itertools import chain
from typing import Generic, Self, TypeVar

T = TypeVar("T")


class BasePattern(ABC, Generic[T]):
    """Abstract base class representing a sequence of generic elements.

    Specified as a starting fragment, a repeatable middle fragment, and an ending fragment.

    Args:
        start_fragment: The start of the sequence.
        repeatable_fragment: The repeatable middle of the sequence.
        end_fragment: The end of the sequence.
    """

    def __init__(
        self,
        start_fragment: Iterable[T],
        repeatable_fragment: Iterable[T],
        end_fragment: Iterable[T],
    ):
        self._start_fragment = list(start_fragment)
        self._repeatable_fragment = list(repeatable_fragment)
        self._end_fragment = list(end_fragment)

    @property
    def start_fragment(self) -> list[T]:
        """The starting fragment."""
        return self._start_fragment

    @property
    def repeatable_fragment(self) -> list[T]:
        """The repeatable fragment."""
        return self._repeatable_fragment

    @property
    def end_fragment(self) -> list[T]:
        """The ending fragment."""
        return self._end_fragment

    @property
    def _chain(self) -> Iterable[T]:
        # many methods end up needing this chain
        return chain(self._start_fragment, self._repeatable_fragment, self._end_fragment)

    def __repr__(self) -> str:
        s = f"{type(self).__name__}(\n"
        s += f"    start_fragment={self.start_fragment},\n"
        s += f"    repeatable_fragment={self.repeatable_fragment},\n"
        s += f"    end_fragment={self.end_fragment},\n"
        s += ")"
        return s

    def __eq__(self, other: Self) -> bool:
        return (
            isinstance(other, self.__class__)
            and self.start_fragment == other.start_fragment
            and self.repeatable_fragment == other.repeatable_fragment
            and self.end_fragment == other.end_fragment
        )
