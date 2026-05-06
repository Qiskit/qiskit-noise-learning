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

"""Path"""

from .base_sequence import BaseSequence
from .fidelity_index import FidelityIndex
from .instruction_sequence import InstructionSequence
from .path_pattern import PathPattern


class Path(BaseSequence[PathPattern, FidelityIndex]):
    """A sequence of fidelity indices."""

    def extend_permutations(
        self, instruction_sequence: InstructionSequence
    ) -> InstructionSequence | None:
        r"""Return an instruction sequence with extended permutations to traverse self.

        Attempts to construct a new :class:`InstructionSequence` that traverses a superset of paths
        including ``self`` and the paths traversed by the input sequence.

        See :meth:`.PathPattern.extend_permutations` for a detailed description of how the
        :class:`.InstructionPattern` is constructed.

        Args:
            instruction_sequence: The instruction sequence to extend.

        Returns:
            The extended instruction sequence, or ``None`` if an extension is not possible.
        """

        if self.depth != instruction_sequence.depth:
            return None

        extended_pattern = self.pattern.extend_permutations(instruction_sequence.pattern)
        if extended_pattern is None:
            return None

        return InstructionSequence(pattern=extended_pattern, depth=self.depth)

    def is_traversed_by(self, instruction_sequence: InstructionSequence) -> bool:
        """Whether or not this path is traversed by the instruction sequence.

        Returns:
            ``True`` if the depths are the same and the pattern of ``self`` is traversed by
            the instruction sequence pattern. See :meth:`PathPattern.is_traversed_by` for details.
        """
        return self.depth == instruction_sequence.depth and self.pattern.is_traversed_by(
            instruction_sequence.pattern
        )

    def sign(self, instruction_sequence: InstructionSequence) -> int:
        """Return the observable sign of the instruction sequence when traversing self.

        Args:
            instruction_sequence: The instruction sequence to check.

        Returns:
            The sign.

        Raises:
            ValueError: If the instruction sequence does not traverse the path.
        """
        if self.depth != instruction_sequence.depth:
            raise ValueError()

        start_flip, repeatable_flip = self.pattern.sign_flips(instruction_sequence.pattern)
        return (-1) ** (start_flip + repeatable_flip * self.depth)

    def to_instruction_sequence(self) -> InstructionSequence:
        r"""Return an instruction sequence that traverses this path.

        The returned sequence is minimally specified, in the sense that the layers of single qubit
        Cliffords between the gate set elements are given as :class:`.PartialPauliPermutation`\s
        which only specify the mappings required to traverse this path.

        See :meth:`.PathPattern.to_instruction_pattern` for a detailed description of how the
        :class:`.InstructionPattern` is constructed.

        Returns:
            The instruction sequence.
        """

        return InstructionSequence(pattern=self.pattern.to_instruction_pattern(), depth=self.depth)

    def __hash__(self) -> int:
        if not hasattr(self, "_hash"):
            self._hash = hash((self.pattern, self.depth))
        return self._hash
