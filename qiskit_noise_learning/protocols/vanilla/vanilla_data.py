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

"""The record the vanilla protocol leaves in a program for its analysis counterpart."""

from collections.abc import Sequence


class VanillaData:
    """Additional passthrough data needed by the vanilla protocol.

    Args:
        gate_names: The name of the gate each instruction given to
            :func:`~.prepare_vanilla_program` became, in the order the instructions were given.
    """

    def __init__(self, gate_names: Sequence[str]):
        self._gate_names = list(gate_names)

    @property
    def gate_names(self) -> list[str]:
        """The gate name of every submitted instruction, in submission order."""
        return self._gate_names
