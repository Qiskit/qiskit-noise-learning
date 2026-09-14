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

from dataclasses import dataclass


@dataclass(frozen=True)
class MeasurementRegister:
    """A description of a classical register.

    Args:
        name: The name of the classical register.
        qubit_idxs: The measured physical qubit indices, where entry ``j`` is the qubit whose
            outcome is stored in classical bit ``j`` of the register.
        measuring_gate_idx: The position of the register's measuring gate among the measuring
            gates of an instruction sequence. A value of ``-1`` indicates the register does not
            correspond to a result within an instruction sequence.
    """

    name: str
    qubit_idxs: tuple[int, ...]
    measuring_gate_idx: int

    @property
    def num_bits(self) -> int:
        """The number of classical bits in the register."""
        return len(self.qubit_idxs)
