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

"""ApplyGate"""

from typing import Self

from qiskit_noise_learning.gate_sets import ModelGate

from .instruction import Instruction


class ApplyGate(Instruction):
    """An instruction that applies a fixed gate."""

    def __init__(self, gate: ModelGate):
        self._gate = gate

    @property
    def gate(self) -> ModelGate:
        """The gate to apply."""
        return self._gate

    def is_mergeable_with(self, other: "Instruction") -> bool:
        """Whether or not this apply gate is mergeable with another one.

        Note that mergeability of two :class:`ApplyGate` instructions is based on equality
        of the names of the contained model gates.

        Args:
            other: The other instruction to check mergeablitity with.

        Returns:
            Whether this instruction is mergeable with the other.
        """
        return isinstance(other, ApplyGate) and self.gate.name == other.gate.name

    def merge(self, other):
        if not self.is_mergeable_with(other):
            raise ValueError("Cannot merge ApplyGate instructions with different gates.")

        return ApplyGate(self.gate)

    def __eq__(self, other: Self) -> bool:
        return isinstance(other, ApplyGate) and self.gate == other.gate

    def __repr__(self) -> str:
        return f"ApplyGate('{self.gate.name}')"
