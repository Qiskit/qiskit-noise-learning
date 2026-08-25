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

from .apply_gate import ApplyGate
from .base_sequence import BaseSequence
from .instruction import Instruction
from .partial_pauli_permutation import PartialPauliPermutation


def _gate_tokens(fragment: list[Instruction]) -> tuple:
    """Return the name of each gate applied in a fragment, in order, ignoring other instructions."""
    return tuple(instr.gate_name for instr in fragment if isinstance(instr, ApplyGate))


def _structure_tokens(fragment: list[Instruction]) -> tuple:
    """Return a token summarizing each instruction of a fragment, in order.

    Args:
        fragment: The fragment to summarize.

    Returns:
        One token per instruction, naming the gate applied or the number of qubits permuted.

    Raises:
        TypeError: If the fragment contains an instruction that is neither a gate application nor a
            partial Pauli permutation.
    """
    tokens = []
    for instr in fragment:
        if isinstance(instr, ApplyGate):
            tokens.append(("gate", instr.gate_name))
        elif isinstance(instr, PartialPauliPermutation):
            tokens.append(("permutation", instr.num_qubits))
        else:
            raise TypeError(
                "Cannot summarize the structure of instruction sequences containing "
                f"{type(instr).__name__} instructions: only gate applications and partial "
                "Pauli permutations are supported."
            )
    return tuple(tokens)


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
    def gate_key(self) -> tuple:
        """A hashable summary of the gate applications in this instruction sequence.

        The key consists of the fragment depth together with the name of each gate applied in each
        fragment, in order, ignoring every other instruction. Two instruction sequences have equal
        gate keys exactly when :meth:`has_same_gates_as` holds between them.
        """
        return (
            self.fragment_depth,
            _gate_tokens(self.start_fragment),
            _gate_tokens(self.repeatable_fragment),
            _gate_tokens(self.end_fragment),
        )

    @property
    def structure_key(self) -> tuple:
        """A hashable summary of the instruction structure of this instruction sequence.

        This key refines :attr:`gate_key` by also recording the position and number of qubits of
        every partial Pauli permutation. Instruction sequences with different structure keys are
        never mergeable, so :meth:`is_mergeable_with` need only be considered among sequences
        sharing one. Equal structure keys further imply that the sequences specify Pauli mappings
        on the same numbers of qubits, in the same order.

        Raises:
            TypeError: If this instruction sequence contains an instruction that is neither a gate
                application nor a partial Pauli permutation.
        """
        return (
            self.fragment_depth,
            _structure_tokens(self.start_fragment),
            _structure_tokens(self.repeatable_fragment),
            _structure_tokens(self.end_fragment),
        )

    def has_same_gates_as(self, other: "InstructionSequence") -> bool:
        """Return whether this instruction sequence has the same gate applications as another.

        Here, having the same gates means that the fragment depths are the same, and all
        fragments contain the same gate applications in the same order, but possibly differing in
        other instructions.

        Args:
            other: Another :class:`.InstructionSequence`.

        Returns:
            Whether this instruction sequence has the same gate applications as the other.
        """
        return self.gate_key == other.gate_key

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
