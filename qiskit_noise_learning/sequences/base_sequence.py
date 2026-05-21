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

"""BaseSequence"""

from abc import ABC
from collections.abc import Iterable, Iterator
from itertools import chain
from typing import Generic, Self, TypeVar

T = TypeVar("T")


class BaseSequence(ABC, Generic[T]):
    """Abstract base class representing a sequence of generic elements.

    Specified as a starting fragment, a repeatable middle fragment, an ending fragment, and an
    optional depth indicating the number of repetitions of the middle fragment. When ``depth`` is
    ``None``, the instance represents a variable-depth sequence whose structure is defined but whose
    concrete length is not yet determined. Variable-depth sequences cannot be iterated, indexed, or
    queried for length.

    Args:
        start_fragment: The start of the sequence.
        repeatable_fragment: The repeatable middle of the sequence.
        end_fragment: The end of the sequence.
        depth: The number of repetitions of the repeatable fragment. If ``None``, the sequence
            represents a variable-depth sequence.
    """

    def __init__(
        self,
        start_fragment: Iterable[T],
        repeatable_fragment: Iterable[T],
        end_fragment: Iterable[T],
        depth: int | None = None,
    ):
        self._start_fragment = list(start_fragment)
        self._repeatable_fragment = list(repeatable_fragment)
        self._end_fragment = list(end_fragment)
        self._depth = depth

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
    def depth(self) -> int | None:
        """The number of repetitions of the repeatable fragment, or ``None`` for variable-depth."""
        return self._depth

    @property
    def _fragment_chain(self) -> Iterable[T]:
        return chain(self._start_fragment, self._repeatable_fragment, self._end_fragment)

    def __iter__(self) -> Iterator[T]:
        if self._depth is None:
            raise ValueError("Cannot iterate over a variable-depth sequence (depth is None).")
        yield from self._start_fragment
        for _ in range(self._depth):
            yield from self._repeatable_fragment
        yield from self._end_fragment

    def __getitem__(self, idx) -> T:
        if self._depth is None:
            raise ValueError("Cannot index a variable-depth sequence (depth is None).")
        if idx < 0:
            raise IndexError("No negative indices allowed.")

        if idx < (num_start := len(self._start_fragment)):
            return self._start_fragment[idx]

        idx -= num_start
        if idx < (num_repeat := len(self._repeatable_fragment) * self._depth):
            return self._repeatable_fragment[idx % len(self._repeatable_fragment)]

        idx -= num_repeat
        if idx >= len(self._end_fragment):
            raise IndexError("Index exceeds length of sequence.")

        return self._end_fragment[idx]

    def __len__(self) -> int:
        if self._depth is None:
            raise ValueError("Cannot compute length of a variable-depth sequence (depth is None).")
        return (
            len(self._start_fragment)
            + len(self._repeatable_fragment) * self._depth
            + len(self._end_fragment)
        )

    def __repr__(self) -> str:
        s = f"{type(self).__name__}(\n"
        s += f"    start_fragment={self.start_fragment},\n"
        s += f"    repeatable_fragment={self.repeatable_fragment},\n"
        s += f"    end_fragment={self.end_fragment},\n"
        if self._depth is not None:
            s += f"    depth={self._depth},\n"
        s += ")"
        return s

    def __eq__(self, other: Self) -> bool:
        return (
            isinstance(other, self.__class__)
            and self.start_fragment == other.start_fragment
            and self.repeatable_fragment == other.repeatable_fragment
            and self.end_fragment == other.end_fragment
            and self.depth == other.depth
        )
