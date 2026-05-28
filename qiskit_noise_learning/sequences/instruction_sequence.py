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

from collections.abc import Iterator
from typing import Self

from .apply_gate import ApplyGate
from .base_sequence import BaseSequence
from .instruction import Instruction
from .partial_pauli_permutation import PartialPauliPermutation


def _complete_fragment(fragment: list[Instruction]) -> Iterator[Instruction]:
    for instruction in fragment:
        if isinstance(instruction, PartialPauliPermutation):
            yield instruction.complete()
        else:
            yield instruction


def _filter_to_gates(fragment: list[Instruction]) -> Iterator[ApplyGate]:
    for instruction in fragment:
        if isinstance(instruction, ApplyGate):
            yield instruction


class InstructionSequence(BaseSequence[Instruction]):
    """A sequence of instructions.

    When ``depth`` is ``None``, represents a variable-depth instruction sequence whose structure is
    defined but whose concrete length is not yet determined.

    Args:
        start_fragment: The start of the sequence.
        repeatable_fragment: The repeatable middle of the sequence.
        end_fragment: The end of the sequence.
        depth: The number of repetitions of the repeatable fragment.
    """

    @property
    def is_complete(self) -> bool:
        r"""Whether all contained :class:`PartialPauliPermutation`\s are complete."""
        return not any(
            isinstance(instr, PartialPauliPermutation) and not instr.is_complete
            for instr in self._fragment_chain
        )

    def complete(self) -> Self:
        """Return a new instance whose data is the same as ``self`` except that all contained
        :class:`PartialPauliPermutation` instances are completed.

        Returns:
            A new :class:`InstructionSequence` instance.
        """
        return InstructionSequence(
            start_fragment=_complete_fragment(self.start_fragment),
            repeatable_fragment=_complete_fragment(self.repeatable_fragment),
            end_fragment=_complete_fragment(self.end_fragment),
            depth=self.depth,
        )

    def has_same_structure_as(self, other: "InstructionSequence") -> bool:
        """Return whether this instruction sequence shares the same circuit structure as another.

        Here, sharing the same structure means that the depths are the same (including both being
        ``None``), and all fragments have the same gate applications in the same order, but
        possibly differing in other instructions. The "structure" implied here is that, if the
        sequences agree on :class:`ApplyGate` instructions, then they can be implemented within
        the same template circuit.

        Args:
            other: Another :class:`.InstructionSequence`.

        Returns:
            Whether this instruction sequence shares the same structure as the other.
        """
        if self.depth != other.depth:
            return False

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
        r"""Check if this instruction sequence is mergeable with another instruction sequence.

        Two instruction sequences are mergeable if they have compatible depths (both ``None``, or
        equal integers), and if their fragments are element-wise mergeable.

        Args:
            other: The other :class:`.InstructionSequence`.

        Returns:
            Whether this instance is mergeable with another.
        """
        if self.depth != other.depth:
            return False

        return (
            len(self.start_fragment) == len(other.start_fragment)
            and len(self.repeatable_fragment) == len(other.repeatable_fragment)
            and len(self.end_fragment) == len(other.end_fragment)
            and all(
                instr0.is_mergeable_with(instr1)
                for instr0, instr1 in zip(self._fragment_chain, other._fragment_chain)  # noqa: SLF001
            )
        )

    def merge(self, other: Self) -> Self:
        r"""Merge this instruction sequence with another instruction sequence.

        Assuming ``self.is_mergeable_with(other)``, the returned merged sequence is constructed by
        merging each corresponding fragment element-wise.

        Args:
            other: The other :class:`.InstructionSequence`.

        Returns:
            The merged sequence of self and other.

        Raises:
            ValueError: If the sequences are not mergeable.
        """
        if self.depth != other.depth:
            raise ValueError("Cannot merge InstructionSequences with different depths.")
        if len(self.start_fragment) != len(other.start_fragment):
            raise ValueError(
                f"Cannot merge InstructionSequences {self} and {other} due to "
                "start_fragments of different lengths."
            )
        if len(self.repeatable_fragment) != len(other.repeatable_fragment):
            raise ValueError(
                f"Cannot merge InstructionSequences {self} and {other} due to "
                "repeatable_fragments of different lengths."
            )
        if len(self.end_fragment) != len(other.end_fragment):
            raise ValueError(
                f"Cannot merge InstructionSequences {self} and {other} due to "
                "end_fragments of different lengths."
            )

        return InstructionSequence(
            start_fragment=[x.merge(y) for x, y in zip(self.start_fragment, other.start_fragment)],
            repeatable_fragment=[
                x.merge(y) for x, y in zip(self.repeatable_fragment, other.repeatable_fragment)
            ],
            end_fragment=[x.merge(y) for x, y in zip(self.end_fragment, other.end_fragment)],
            depth=self.depth,
        )
