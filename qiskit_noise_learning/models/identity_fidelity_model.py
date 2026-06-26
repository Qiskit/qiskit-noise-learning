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

"""IdentityFidelityModel"""

from collections.abc import Iterable

from qiskit_noise_learning.gate_sets import GateSet, ModelGateSet
from qiskit_noise_learning.math import IndexedMatrix, IndexedVector, LinearMap
from qiskit_noise_learning.sequences import FidelityIndex

from .log_fidelity_space import LogFidelitySpace


class IdentityFidelityModel(LinearMap[FidelityIndex, FidelityIndex]):
    r"""A fidelity model whose parameters are the log fidelities themselves.

    The parameterization matrix is the identity: the input and output spaces are the same
    :class:`~.LogFidelitySpace`, and the row of a fidelity index is the unit vector on that index.

    Args:
        gate_set: The gate set whose fidelities are being modelled. To be converted to a
            :class:`~.ModelGateSet`.
    """

    def __init__(self, gate_set: GateSet):
        space = LogFidelitySpace(gate_set)
        super().__init__(input_space=space, output_space=space)

    @property
    def gate_set(self) -> ModelGateSet:
        """The gate set whose fidelities are being modelled."""
        return self.output_space.gate_set

    def rows(
        self, output_indices: Iterable[FidelityIndex]
    ) -> IndexedMatrix[FidelityIndex, FidelityIndex]:
        """Construct the sub-matrix whose rows are the given fidelity indices.

        Each row is the unit vector on its fidelity index (the identity parameterization).

        Args:
            output_indices: The fidelity indices labelling the desired rows.

        Returns:
            An :class:`~.IndexedMatrix` whose rows and columns are both the requested fidelity
            indices, with identity data.
        """
        output_indices = list(output_indices)
        return IndexedMatrix.from_rows(
            output_indices,
            [IndexedVector[FidelityIndex]({fidelity: 1.0}) for fidelity in output_indices],
        )
