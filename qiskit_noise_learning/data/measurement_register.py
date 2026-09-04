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
    """One classical register of an experiment, and what its bits mean.

    An ordered collection of these fully describes the ``"bit"`` dimension of a
    :class:`~.RawData` leaf: each register contributes ``num_bits`` bits, and every bit knows
    the register it belongs to, the physical qubit it holds the outcome for, and the measuring
    gate whose outcomes its register was created to hold.

    Note that a register need not measure in ascending qubit order, and that the same physical
    qubit may be measured by more than one register.

    Args:
        name: The name of the classical register.
        qubit_idxs: The measured physical qubit indices, where entry ``j`` is the qubit whose
            outcome is stored in classical bit ``j`` of the register.
        measuring_gate_idx: The position of the register's measuring gate among the measuring
            gates of the instruction sequence, in the order those gates are traversed, or ``-1``
            if the register has no corresponding measuring gate.

            Only measuring gates are counted, because a single :class:`~.RawData` leaf holds
            data from more than one fragment depth, so a position among *all* gates would not be
            a property of the register.
    """

    name: str
    qubit_idxs: tuple[int, ...]
    measuring_gate_idx: int

    @property
    def num_bits(self) -> int:
        """The number of classical bits in the register."""
        return len(self.qubit_idxs)
