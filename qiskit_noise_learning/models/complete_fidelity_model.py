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

"""CompleteFidelityModel"""

from functools import cached_property

from qiskit_noise_learning.math import IndexedVector, ParameterSpace
from qiskit_noise_learning.sequences import FidelityIndex

from .fidelity_index_space import FidelityIndexSpace
from .fidelity_model import FidelityModel


class CompleteFidelityModel(FidelityModel[FidelityIndex]):
    r"""A fidelity model where the parameters are the log fidelities themselves.

    In this case, the parameterization matrix :math:`A` is simply the identity.

    Args:
        gate_set: The gate set for which the fidelities are being modelled.
    """

    @cached_property
    def input_space(self) -> ParameterSpace[FidelityIndex]:
        return FidelityIndexSpace(self._gate_set)

    @cached_property
    def output_space(self) -> ParameterSpace[FidelityIndex]:
        return FidelityIndexSpace(self._gate_set)

    def _core_row(self, fidelity_index):
        return IndexedVector[FidelityIndex]({fidelity_index: 1.0})
