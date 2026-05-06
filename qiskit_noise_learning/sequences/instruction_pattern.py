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

"""InstructionPattern"""

from collections.abc import Iterator
from typing import Self

from .apply_gate import ApplyGate
from .base_pattern import BasePattern
from .instruction import Instruction
from .partial_pauli_permutation import PartialPauliPermutation


def _complete_fragment(fragment: list[Instruction]) -> Iterator[Instruction]:
    # turn each given instruction into a "complete" one
    for instruction in fragment:
        if isinstance(instruction, PartialPauliPermutation):
            yield instruction.complete()
        else:
            yield instruction


def _filter_to_gates(fragment: list[Instruction]) -> Iterator[ApplyGate]:
    for instruction in fragment:
        if isinstance(instruction, ApplyGate):
            yield instruction


class InstructionPattern(BasePattern[Instruction]):
    """A sequence of fragments of ordered instructions."""

    @property
    def is_complete(self) -> bool:
        r"""Whether all contained :class:`PartialPauliPermutation`\s are complete."""
        return not any(
            isinstance(instr, PartialPauliPermutation) and not instr.is_complete
            for instr in self._chain
        )

    def complete(self) -> Self:
        """Return a new instance where every :class:`PartialPauliPermutation` is completed.

        Returns:
            A new :class:`InstructionPattern` instance.
        """

        return InstructionPattern(
            start_fragment=_complete_fragment(self.start_fragment),
            repeatable_fragment=_complete_fragment(self.repeatable_fragment),
            end_fragment=_complete_fragment(self.end_fragment),
        )

    def has_same_structure_as(self, other: "InstructionPattern") -> bool:
        """Return whether this instruction pattern shares the same circuit structure as another.

        Here, sharing the same structure means that all fragments have the same gate applications
        and in the same order, but they might differ on single-qubit gate rounds. This concept is
        relevant when deciding which instruction patterns and sequences can share the same circuit
        templates at execution time.

        Args:
            other: Another :class:`.InstructionPattern`.

        Returns:
            Whether this instruction pattern shares the same structure as the other.
        """
        self_gates = (
            list(_filter_to_gates(self.start_fragment)),
            list(_filter_to_gates(self.repeatable_fragment)),
            list(_filter_to_gates(self.end_fragment)),
        )
        other_gates = (
            list(_filter_to_gates(other.start_fragment)),
            list(_filter_to_gates(other.repeatable_fragment)),
            list(_filter_to_gates(other.end_fragment)),
        )

        return all(
            len(self_fragment) == len(other_fragment)
            and all(
                gate0.is_mergeable_with(gate1)
                for gate0, gate1 in zip(self_fragment, other_fragment)
            )
            for self_fragment, other_fragment in zip(self_gates, other_gates)
        )

    def is_mergeable_with(self, other: Self) -> bool:
        r"""Check if this instruction pattern is mergeable with another instruction pattern.

        Two instruction patterns are mergeable if there exists a third instruction pattern that
        simultaneously implements the action of both patterns. This can be determined on a
        fragment-by-fragment basis. For a given pair of fragments, ``x``, ``y``, mergeabliity
        requires that ``len(x) == len(y)``, and ``x[idx]`` and ``y[idx]`` are mergable (see
        :meth:`.Instruction.is_mergable_with`\) instructions for all ``idx``.

        Args:
            other: The other :class:`.InstructionPattern`.

        Returns:
            Whether this instance is mergeable with another.
        """
        return (
            len(self.start_fragment) == len(other.start_fragment)
            and len(self.repeatable_fragment) == len(other.repeatable_fragment)
            and len(self.end_fragment) == len(other.end_fragment)
            and all(
                instr0.is_mergeable_with(instr1)
                for instr0, instr1 in zip(self._chain, other._chain)  # noqa: SLF001
            )
        )

    def merge(self, other: Self) -> Self:
        r"""Merge this instruction pattern with another instruction pattern.

        Assuming ``self.is_mergeable_with(other)``, the returned merged pattern is constructed by
        merging each corresponding fragment. Merging two compatible fragments ``x``, ``y`` consists
        of building a new list of the same length based on element-wise merging.

        Args:
            other: The other :class:`.InstructionPattern`.

        Returns:
            The merged pattern of self and other.

        Raises:
            ValueError: If the patterns are not mergeable.
        """

        if len(self.start_fragment) != len(other.start_fragment):
            raise ValueError(
                f"Cannot merge InstructionPatterns {self} and {other} due to "
                "start_fragments of different lengths."
            )
        if len(self.repeatable_fragment) != len(other.repeatable_fragment):
            raise ValueError(
                f"Cannot merge InstructionPatterns {self} and {other} due to "
                "repeatable_fragments of different lengths."
            )
        if len(self.end_fragment) != len(other.end_fragment):
            raise ValueError(
                f"Cannot merge InstructionPatterns {self} and {other} due to "
                "end_fragments of different lengths."
            )

        return InstructionPattern(
            start_fragment=[x.merge(y) for x, y in zip(self.start_fragment, other.start_fragment)],
            repeatable_fragment=[
                x.merge(y) for x, y in zip(self.repeatable_fragment, other.repeatable_fragment)
            ],
            end_fragment=[x.merge(y) for x, y in zip(self.end_fragment, other.end_fragment)],
        )
