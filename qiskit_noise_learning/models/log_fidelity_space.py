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

from qiskit_noise_learning.gate_sets import GateSet, ModelGateSet
from qiskit_noise_learning.math import IndexedSpace
from qiskit_noise_learning.sequences import FidelityIndex


class LogFidelitySpace(IndexedSpace[FidelityIndex]):
    r"""The space of log fidelities of a gate set.

    The basis indices are the :class:`~.FidelityIndex` objects of the gate set, excluding the
    trivial identity fidelity of each gate. For each gate, the fidelities are in bijection with a
    Pauli on the unmeasured and unreset qubits :math:`U`, a set of input bits on the measured qubits
    :math:`M`, and a set of output bits on the measured and reset qubits :math:`M \cup R` (see
    :class:`~.FidelityIndex`). The dimension is therefore :math:`\sum_{\text{gates}} \left(4^{|U|}
    \, 2^{|M|} \, 2^{|M \cup R|} - 1\right)`.

    Args:
        gate_set: The gate set whose fidelities the space describes. Converted to a
            :class:`~.ModelGateSet`.
    """

    def __init__(self, gate_set: GateSet):
        self._gate_set = gate_set.model_gate_set

    @property
    def gate_set(self) -> ModelGateSet:
        """The gate set whose fidelities this space describes."""
        return self._gate_set

    @property
    def dim(self) -> int | float:
        total = 0
        for gate in self._gate_set.values():
            measured = gate.meas_idxs
            measured_or_reset = measured | gate.prep_idxs
            num_pauli_qubits = len(set(gate.qubit_idxs) - measured_or_reset)
            gate_count = 4**num_pauli_qubits * 2 ** len(measured) * 2 ** len(measured_or_reset)
            # exclude the trivial identity fidelity of the gate
            total += gate_count - 1
        return total

    def __contains__(self, index: object) -> bool:
        if not isinstance(index, FidelityIndex) or index.gate_name not in self._gate_set:
            return False

        # reject the trivial identity fidelity
        if len(index.pauli.indices) == 0 and not index.in_bit_indices and not index.out_bit_indices:
            return False

        return FidelityIndex.is_valid_for_gate(
            gate=self._gate_set[index.gate_name],
            pauli=index.pauli,
            in_bit_indices=index.in_bit_indices,
            out_bit_indices=index.out_bit_indices,
        )
