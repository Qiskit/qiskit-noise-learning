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

"""LogFidelitySpace"""

from qiskit_noise_learning.gate_sets import ModelGateSet
from qiskit_noise_learning.math import ParameterSpace
from qiskit_noise_learning.sequences import FidelityIndex


class LogFidelitySpace(ParameterSpace[FidelityIndex]):
    """The log-fidelity parameter space for a gate set.

    Each coordinate of this space is the log fidelity of a gate transition, with the coordinates
    indexed by the :class:`~.FidelityIndex` instances valid for the gate set.

    Args:
        gate_set: The model gate set defining the valid fidelity indices.
    """

    def __init__(self, gate_set: ModelGateSet):
        self._gate_set = gate_set

    @property
    def dim(self) -> int:
        total = 0
        for gate in self._gate_set.values():
            num_unmeas_unreset = len(
                set(gate.qubit_idxs) - set(gate.meas_idxs) - set(gate.prep_idxs)
            )
            num_meas = len(gate.meas_idxs)
            num_meas_or_prep = len(set(gate.meas_idxs) | set(gate.prep_idxs))
            total += (4**num_unmeas_unreset) * (2**num_meas) * (2**num_meas_or_prep)
        return total

    @property
    def gate_set(self) -> ModelGateSet:
        """The gate set defining valid fidelity indices."""
        return self._gate_set

    def __contains__(self, index: object) -> bool:
        if not isinstance(index, FidelityIndex):
            return False
        return index.gate_name in self._gate_set
