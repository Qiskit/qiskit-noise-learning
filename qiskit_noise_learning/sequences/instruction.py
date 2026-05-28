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

"""Instruction"""

from abc import abstractmethod
from typing import Self


class Instruction:
    """Abstract base class for instructions in instruction sequences."""

    @abstractmethod
    @property
    def is_complete(self) -> bool:
        """Whether or not this instruction is completely specified."""

    @abstractmethod
    def complete(self) -> Self:
        """Return a completely specified instruction that also implement self."""

    @abstractmethod
    def is_mergeable_with(self, other: "Instruction") -> bool:
        """Whether or not this instruction is mergeable with another one.

        Two instructions are mergeable if a third instruction exists that simultaneously
        implements both of their actions. The trivial case is when the instructions are equal:
        the third instruction can be a third instance of the same instruction. However, non-trivial
        cases are possible because some instruction types, notably
        :class:`~PartialPauliPermutation`, do not necessarily fully specify their own action, so
        that unequal instances can nevertheless still have their constraints simultaneously
        satisfied by a single third instance.

        If this method returns ``True``, then the method :meth:`~merge` should succeed.

        Args:
            other: The other instruction to check mergeablitity with.

        Returns:
            Whether this instruction is mergeable with the other.
        """

    @abstractmethod
    def merge(self, other: Self) -> Self:
        """Merge self and other into a single instruction.

        Args:
            other: The other instruction to merge with.

        Returns:
            Some instruction (possibly the same instance) that simultaneously implements the
            action of this instruction and the other instruction.
        """

    @abstractmethod
    def __eq__(self, other):
        pass
