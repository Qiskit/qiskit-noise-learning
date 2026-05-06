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

"""InstructionSequence"""

from typing import Self

from .base_sequence import BaseSequence
from .instruction import Instruction
from .instruction_pattern import InstructionPattern


class InstructionSequence(BaseSequence[InstructionPattern, Instruction]):
    """A sequence of instructions for a given gate set."""

    @property
    def is_complete(self) -> bool:
        r"""Whether all contained :class:`PartialPauliPermutation`\s are complete."""
        return self.pattern.is_complete

    def complete(self) -> Self:
        """Return a new instance whose data is the same as ``self`` except that all contained
        :class:`PartialPauliPermutation` instances are completed.

        Returns:
            A new :class:`InstructionPattern` instance.
        """
        return InstructionSequence(pattern=self.pattern.complete(), depth=self.depth)

    def has_same_structure_as(self, other: "InstructionSequence") -> bool:
        """Return whether this instruction sequence shares the same circuit structure as another.

        Here, sharing the same structure means that this instruction sequence has the same depth
        as the other instruction sequence, and that the patterns they own also have the same
        structure, see :meth:`.InstructionPattern.has_same_structure_as`.

        Args:
            other: Another :class:`.InstructionSequence`.

        Returns:
            Whether this instruction sequence shares the same structure as the other.
        """
        return self.depth == other.depth and self.pattern.has_same_structure_as(other.pattern)

    def is_mergeable_with(self, other: Self) -> bool:
        r"""Check if this instruction sequence is mergeable with another instruction sequence.

        Two instruction sequences are mergeable if they have the same depth, and if their
        patterns are mergable, see :meth:`~InstructionPattern.is_mergeable_with`.

        Args:
            other: The other :class:`.InstructionSequence`.

        Returns:
            Whether this instance is mergeable with another.
        """
        return self.pattern.is_mergeable_with(other.pattern) and self.depth == other.depth

    def merge(self, other: Self) -> Self:
        r"""Merge this instruction sequence with another instruction sequence.

        Args:
            other: The other :class:`.InstructionSequence`.

        Returns:
            The merged sequence of self and other.

        Raises:
            ValueError: If the patterns are not mergeable, or if ``self.depth != other.depth``.
        """
        if self.depth != other.depth:
            raise ValueError("self is not mergeable with other due to depth mismatch.")

        return InstructionSequence(pattern=self.pattern.merge(other.pattern), depth=self.depth)
