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
from collections.abc import Iterator
from typing import Any, Generic, Self, TypeVar

from .base_pattern import BasePattern

# Ideally, we'd be able to get away with one TypeVar here, but Python typing is not yet powerful
# enough to do things like "PatternT.T". As such, children must specify a pattern type and an
# instruction type, and make them agree.
InstrT = TypeVar("InstrT")
PatternT = TypeVar("PatternT", bound="BasePattern[Any]")


class BaseSequence(ABC, Generic[PatternT, InstrT]):
    """A :class:`~.BasePattern` with a specified depth.

    Args:
        pattern: The underlying pattern.
        depth: The number of repetitions of the repeatable fragment in ``pattern``.
    """

    def __init__(self, pattern: PatternT, depth: int):
        self._pattern = pattern
        self._depth = depth

    @property
    def depth(self) -> int:
        """The number of repetitions of the repeatable fragment of the :attr:`pattern`."""
        return self._depth

    @property
    def pattern(self) -> PatternT:
        """The underlying pattern."""
        return self._pattern

    def __iter__(self) -> Iterator[InstrT]:
        yield from self.pattern.start_fragment
        for _ in range(self.depth):
            yield from self.pattern.repeatable_fragment
        yield from self.pattern.end_fragment

    def __getitem__(self, idx) -> InstrT:
        if idx < 0:
            raise IndexError("No negative indices allowed.")

        if idx < (num_start := len(self.pattern.start_fragment)):
            return self.pattern.start_fragment[idx]

        idx -= num_start
        if idx < (num_repeat := len(self.pattern.repeatable_fragment) * self.depth):
            return self.pattern.repeatable_fragment[idx % len(self.pattern.repeatable_fragment)]

        idx -= num_repeat
        if idx >= len(self.pattern.end_fragment):
            raise IndexError("Index exceeds length of sequence.")

        return self.pattern.end_fragment[idx]

    def __len__(self) -> int:
        return (
            len(self.pattern.start_fragment)
            + len(self.pattern.repeatable_fragment) * self.depth
            + len(self.pattern.end_fragment)
        )

    def __repr__(self) -> str:
        s = f"{type(self).__name__}(\n"
        s += f"    {self.pattern.start_fragment} +\n"
        s += f"    {self.pattern.repeatable_fragment} * {self.depth} +\n"
        s += f"    {self.pattern.end_fragment},\n"
        s += ")"
        return s

    def __eq__(self, other: Self) -> bool:
        return (
            isinstance(other, self.__class__)
            and self.pattern == other.pattern
            and self.depth == other.depth
        )
