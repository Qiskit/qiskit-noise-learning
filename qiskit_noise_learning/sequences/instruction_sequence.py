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

from collections.abc import Hashable
from typing import Self

from .apply_gate import ApplyGate
from .base_sequence import BaseSequence
from .instruction import Instruction


def _gate_tokens(fragment: list[Instruction]) -> tuple[str, ...]:
    """Return the name of each gate applied in a fragment, in order.

    Args:
        fragment: The fragment to summarize.

    Returns:
        One gate name per gate application, every other instruction being ignored.
    """
    return tuple(instr.gate_name for instr in fragment if isinstance(instr, ApplyGate))


def _validate_instructions(fragment: list[Instruction]) -> None:
    """Validate that every element of a fragment is an instruction.

    Args:
        fragment: The fragment to validate.

    Raises:
        TypeError: If the fragment contains an object that is not an instruction.
    """
    for instr in fragment:
        if not isinstance(instr, Instruction):
            raise TypeError(
                "Cannot summarize the structure of instruction sequences containing "
                f"{type(instr).__name__} objects, which are not instructions."
            )


def _structure_tokens(fragment: list[Instruction]) -> tuple[Hashable, ...]:
    """Return the structure token of each instruction of a fragment, in order.

    Args:
        fragment: The fragment to summarize.

    Returns:
        One structure token per instruction.

    Raises:
        TypeError: If the fragment contains an object that is not an instruction.
    """
    _validate_instructions(fragment)
    return tuple(instr.structure_token for instr in fragment)


class InstructionSequence(BaseSequence[Instruction]):
    """A sequence of instructions.

    Args:
        start_fragment: The start of the sequence.
        repeatable_fragment: The repeatable middle of the sequence.
        end_fragment: The end of the sequence.
        fragment_depth: The number of repetitions of the repeatable fragment.
    """

    @property
    def is_complete(self) -> bool:
        r"""Whether all contained instructions are completely specified."""
        return all(instr.is_complete for instr in self._fragment_chain)

    def complete(self) -> Self:
        """Return a new instance whose data is the same as ``self`` except that all contained
        instructions are completed.

        Returns:
            A new :class:`InstructionSequence` instance.
        """
        return InstructionSequence(
            start_fragment=[x.complete() for x in self.start_fragment],
            repeatable_fragment=[x.complete() for x in self.repeatable_fragment],
            end_fragment=[x.complete() for x in self.end_fragment],
            fragment_depth=self.fragment_depth,
        )

    @property
    def gate_key(self) -> Hashable:
        """A hashable summary of the gate applications in this instruction sequence.

        Two instruction sequences have equal gate keys exactly when they have the same fragment
        depth and each of their fragments applies the same gates in the same order, however else
        they differ; every instruction that is not a gate application is ignored. The value itself
        is opaque, with only equality and hashability guaranteed.
        """
        return (
            self.fragment_depth,
            _gate_tokens(self.start_fragment),
            _gate_tokens(self.repeatable_fragment),
            _gate_tokens(self.end_fragment),
        )

    @property
    def structure_key(self) -> Hashable:
        """A hashable summary of the instruction structure of this sequence.

        This key comes with the following guarantees:

        * If two instruction sequences have different keys, they are not mergeable.
        * If two instruction sequences have the same key, they have the same fragment depth, and
          their fragments contain the same sequence of instruction types on the same qubits.

        Raises:
            TypeError: If this instruction sequence contains an object that is not an instruction.
        """
        return (
            self.fragment_depth,
            _structure_tokens(self.start_fragment),
            _structure_tokens(self.repeatable_fragment),
            _structure_tokens(self.end_fragment),
        )

    def is_mergeable_with(self, other: Self) -> bool:
        r"""Check if this instruction sequence is mergeable with another instruction sequence.

        Two instruction sequences are mergeable if they have compatible fragment depths and their
        fragments are element-wise mergeable.

        Args:
            other: The other :class:`.InstructionSequence`.

        Returns:
            Whether this instance is mergeable with another.
        """
        if self.fragment_depth != other.fragment_depth:
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

        Assuming this instance is mergeable with ``other``, the returned merged sequence is
        constructed by merging each corresponding fragment element-wise.

        Args:
            other: The instruction sequence to merge this with.

        Returns:
            The merged sequence of self and other.

        Raises:
            ValueError: If the sequences are not mergeable.
        """
        if self.fragment_depth != other.fragment_depth:
            raise ValueError("Cannot merge InstructionSequences with different fragment depths.")
        if (self_len := len(self.start_fragment)) != (other_len := len(other.start_fragment)):
            raise ValueError(
                f"Cannot merge InstructionSequences with start fragments of different "
                f"lengths: {self_len} and {other_len}."
            )
        if (self_len := len(self.repeatable_fragment)) != (
            other_len := len(other.repeatable_fragment)
        ):
            raise ValueError(
                f"Cannot merge InstructionSequences with repeatable fragments of different "
                f"lengths: {self_len} and {other_len}."
            )
        if (self_len := len(self.end_fragment)) != (other_len := len(other.end_fragment)):
            raise ValueError(
                f"Cannot merge InstructionSequences with end fragments of different "
                f"lengths: {self_len} and {other_len}."
            )

        return InstructionSequence(
            start_fragment=[x.merge(y) for x, y in zip(self.start_fragment, other.start_fragment)],
            repeatable_fragment=[
                x.merge(y) for x, y in zip(self.repeatable_fragment, other.repeatable_fragment)
            ],
            end_fragment=[x.merge(y) for x, y in zip(self.end_fragment, other.end_fragment)],
            fragment_depth=self.fragment_depth,
        )
